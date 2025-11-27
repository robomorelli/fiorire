import torch.nn as nn
from collections import OrderedDict
from models.utils.layers import conv_block1D, deconv_block1D, bottleneck1D
import torch

from config import activation_dict
torch.manual_seed(0)

class Encoder1D(nn.Module):
    def __init__(self,
                 in_channels=1,
                 base_filters=32,
                 kernel_size=3,
                 num_layers=2,
                 padding=1,
                 dilation=1,
                 pool_ks=2,
                 pool_stride=2,
                 activation=nn.ReLU(),
                 bottleneck_act=None,
                 compression_factor=2,
                 seq_length=16,
                 flattened=True):
        super().__init__()
        self.in_channels = in_channels
        self.base_filters = base_filters
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.padding = padding
        self.pool_ks = pool_ks
        self.pool_stride = pool_stride
        self.compression_factor = compression_factor
        self.seq_length = seq_length
        self.act = activation
        self.bottleneck_act = bottleneck_act
        self.flattened = flattened

        layers = []
        in_f = in_channels
        for i in range(num_layers):
            out_f = base_filters * (2 ** i)
            layers.append((
                f'enc_lay_{i+1}',
                conv_block1D(
                    in_f, out_f,
                    kernel_size=self.kernel_size,
                    dilation=dilation,
                    pool_ks=self.pool_ks,
                    pool_stride=self.pool_stride,
                    activation=self.act,
                    padding=self.padding
                )
            ))
            in_f = out_f

        self.encoder = nn.Sequential(OrderedDict(layers))

        # Bottleneck: 1x1 conv doubling channels
        self.bottleneck_out_channels = in_f * 2
        self.bottleneck = bottleneck1D(in_f, self.bottleneck_out_channels, activation=self.act,
                                       bottleneck_activation=self.bottleneck_act, batch_norm=True)

        # Compute flattened size after bottleneck
        self.flattened_size, self.seq_enc = self._get_final_flattened_size()
        self.latent_dim = int(self.flattened_size // self.compression_factor)

        if self.flattened:
            self.flatten = nn.Flatten()
            self.encoder_layer = nn.Linear(self.flattened_size, self.latent_dim)

        self._init_weights()

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(1, self.in_channels, self.seq_length)
            x = self.encoder(x)
            x = self.bottleneck(x)
            _, c, l = x.size()
        return c * l, l

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        if self.flattened:
            x = self.flatten(x)
            x = self.encoder_layer(x)
        return x


# ==============================
# Decoder 1D
# ==============================

class Decoder1D(nn.Module):
    def __init__(self,
                 in_channels=1,
                 base_filters=32,
                 kernel_size=2,
                 num_layers=2,
                 stride=2,
                 latent_dim=None,
                 flattened_size=None,
                 seq_length=16,
                 seq_enc=2,
                 activation=nn.ReLU(),
                 flattened=True,
                 double_deconv=False,
                 conv_padding=0,
                 conv_kernel_size=3,
                 conv_stride=1,
                 conv_dilation=1,
                 bottleneck_out_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.base_filters = base_filters
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.seq_enc = seq_enc
        self.act = activation
        self.flattened = flattened
        self.stride = stride
        self.latent_dim = latent_dim
        self.conv_stride = conv_stride
        self.conv_dilation = conv_dilation
        self.conv_kernel_size = conv_kernel_size
        self.conv_padding = conv_padding
        self.double_deconv = double_deconv

        if bottleneck_out_channels is None:
            raise ValueError("Provide bottleneck_out_channels from encoder")
        self.bottleneck_out_channels = bottleneck_out_channels

        if flattened:
            self.reshape = nn.Linear(self.latent_dim, flattened_size)

        output_padding = self._compute_output_padding(seq_enc, seq_length, num_layers, stride)

        # Build decoder
        layers = []
        in_f = bottleneck_out_channels
        out_f = in_f // 2
        for i in range(num_layers):
            if i == num_layers - 1:
                out_f = base_filters
            layers.append((
                f'dec_lay_{i+1}',
                deconv_block1D(
                    in_f, out_f,
                    kernel_size=self.kernel_size,
                    stride=stride,
                    activation=self.act,
                    double_deconv=self.double_deconv,
                    output_padding=output_padding[i],
                    conv_kernel_size=self.conv_kernel_size,
                    conv_padding=self.conv_padding,
                    conv_stride=self.conv_stride,
                    conv_dilation=self.conv_dilation
                )
            ))
            in_f = out_f
            if i != num_layers - 1:
                out_f = max(in_f // 2, base_filters)

        self.decoder = nn.Sequential(OrderedDict(layers))
        self.decoder_out = nn.Conv1d(base_filters, in_channels, kernel_size=1)
        self._init_weights()

    def _compute_output_padding(self, Lb, L_target, num_layers, stride, kernel_size=2, padding=0, dilation=1):
        """
        Compute per-layer output_padding for ConvTranspose1d so that
        the final output length matches L_target.

        Lb: encoded sequence length (from encoder)
        L_target: original input sequence length
        num_layers: number of decoder layers
        stride, kernel_size, padding, dilation: ConvTranspose1d parameters
        """

        # Compute lengths forward without output_padding
        lengths = [Lb]
        for _ in range(num_layers):
            L_in = lengths[-1]
            L_out = (L_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1
            lengths.append(L_out)

        diff = L_target - lengths[-1]

        # Distribute output_padding from the last layer backwards
        ops = [0] * num_layers
        for i in reversed(range(num_layers)):
            if diff <= 0:
                break
            # Output padding can't exceed stride - 1
            op = min(diff, stride - 1)
            ops[i] = op
            # Each output padding at layer i increases output length by op * (stride ** i)
            diff -= op * (stride ** (num_layers - 1 - i))

        return ops

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        if self.flattened:
            x = self.reshape(x)
            x = x.view((-1, self.bottleneck_out_channels, self.seq_enc))
        x = self.decoder(x)
        out = self.decoder_out(x)
        return out


# ==============================
# Full CONV_AE1D
# ==============================


class CONV_AE1D(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        model_cfg = cfg.model

        self.in_channels = cfg.dataset.n_features
        self.kernel_size = model_cfg.kernel_size
        self.base_filters = model_cfg.base_filters
        self.double_deconv = model_cfg.double_deconv
        self.num_layers = model_cfg.num_layers
        self.act = activation_dict.get(model_cfg.get("activation", None), None)
        self.bottleneck_act = activation_dict.get(model_cfg.get("bottleneck_activation", None), None)
        self.stride = model_cfg.stride
        self.pool = model_cfg.pool
        self.increasing = model_cfg.increasing
        self.dilation = model_cfg.dilation
        self.flattened = model_cfg.flattened
        self.compression_factor = model_cfg.compression_factor
        self.seq_length = cfg.dataset.seq_in_length
        self.pool_ks = 2
        self.pool_stride = 2

        self.padding = (self.kernel_size - 1) // 2  # automatic "same" padding

        # Encoder
        self.encoder = Encoder1D(
            in_channels=self.in_channels,
            base_filters=self.base_filters,
            kernel_size=self.kernel_size,
            padding=self.padding,
            num_layers=self.num_layers,
            seq_length=self.seq_length,
            activation=self.act,
            flattened=self.flattened,
            compression_factor=self.compression_factor,
            bottleneck_act = self.bottleneck_act
        )

        self.flattened_size = self.encoder.flattened_size
        self.cfg.model.flattened_size = self.flattened_size
        self.latent_dim = int(self.flattened_size // self.compression_factor)

        # Decoder
        self.decoder = Decoder1D(
            in_channels=self.in_channels,
            base_filters=self.base_filters,
            num_layers=self.num_layers,
            kernel_size=self.pool_ks,
            stride=self.pool_stride,
            latent_dim=self.latent_dim,
            flattened_size=self.flattened_size,
            seq_length=self.seq_length,
            seq_enc=self.encoder.seq_enc,
            activation=self.act,
            flattened=self.flattened,
            double_deconv=self.double_deconv, conv_padding=self.padding,
            conv_kernel_size=self.kernel_size, conv_stride=self.stride, conv_dilation=self.dilation,
            bottleneck_out_channels=self.encoder.bottleneck_out_channels,
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out