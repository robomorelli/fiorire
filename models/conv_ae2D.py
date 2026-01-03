import torch.nn as nn
import torch
from config import activation_dict
from models.utils.layers import conv_block, deconv_block, bottleneck2D
from typing import Union, Tuple
import math
from collections import OrderedDict

torch.manual_seed(0)

class Encoder(nn.Module):
    def __init__(self, in_channels=1, base_filters=32, kernel_size=3, num_layers=2,
                 padding: Union[int, Tuple[int, int]] = 1,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 pool_ks: Union[int, Tuple[int, int]] = 2,
                 pool_stride: Union[int, Tuple[int, int]] = 2, activation=nn.ReLU(),
                 compression_factor=2,
                 img_heigth=16, img_width=16, flattened=True,  bottleneck_act=nn.Tanh(), compression_type='on_features'):
        super(Encoder, self).__init__()

        self.in_channels = in_channels
        self.base_filters = base_filters
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.num_layers = num_layers
        self.padding = padding
        self.pool_ks = pool_ks
        self.pool_stride = pool_stride
        self.compression_factor = compression_factor
        self.compression_type = compression_type
        self.h = img_heigth
        self.w = img_width
        self.flattened_inputs = self.h*self.w
        self.act = activation
        self.flattened = flattened
        self.bottleneck_act = bottleneck_act
        self.act_str = {'relu':'relu', 'lrelu':'leaky_relu', 'elu':'relu'}[{v: k for k, v in activation_dict.items()}.get(activation, 'relu').lower()]

        out_f = None
        encoder_layers = []
        in_f = self.in_channels
        for i in range(self.num_layers):
            out_f = self.base_filters * (2 ** i)
            encoder_layers.append((
                f'enc_lay_{i + 1}',
                conv_block(
                    in_f, out_f,
                    kernel_size=self.kernel_size,
                    dilation=self.dilation,
                    pool_ks=self.pool_ks,
                    pool_stride=self.pool_stride,
                    activation=self.act,
                    padding=self.padding
                )
            ))
            in_f = out_f

        self.encoder = nn.Sequential(OrderedDict(encoder_layers))

        # ✅ Add bottleneck: 1×1 conv doubling the channels
        #self.bottleneck_out_channels = in_f * 2  # <--- store output filter count
        #self.bottleneck = bottleneck2D(in_f, self.bottleneck_out_channels, activation=self.act, batch_norm=True)
        # compute flattened size *after* bottleneck
        self.last_layers_channels = out_f * 2
        bottleneck_conv = nn.Conv2d(in_f, self.last_layers_channels, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.flattened_size, self.h_enc, self.w_enc = self._get_final_flattened_size(bottleneck_conv)

        self.latent_dim = int(self.flattened_size // self.compression_factor) if  self.compression_type == 'on_features' else int((self.h*self.w // self.compression_factor))

        self.bottleneck = bottleneck2D(bottleneck_conv=bottleneck_conv,
                                       flattened=self.flattened, flattened_size= self.flattened_size,
                                       latent_dim=self.latent_dim, activation=self.act,
                                       bottleneck_activation=self.bottleneck_act, batch_norm=True)

        self.init_kaiming_normal()

    def init_kaiming_normal(self, mode='fan_in'):
        print('Initializing conv2d weights with Kaiming He normal')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode=mode, nonlinearity=self.act_str)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _get_final_flattened_size(self, bottleneck_conv):
        """
        Calcola la dimensione flatten finale dell'encoder + bottleneck_conv
        usando un input dummy coerente con dtype e device del modello.
        """
        with torch.no_grad():
            # Recupera dtype e device coerenti con il modello
            param = next(self.parameters())
            dtype = param.dtype
            device = param.device

            # Crea input dummy con gli stessi dtype e device
            x = torch.zeros(
                (1, self.in_channels, self.h, self.w),
                dtype=dtype,
                device=device
            )

            # Passa attraverso encoder e bottleneck
            x = self.encoder(x)
            x = bottleneck_conv(x)

            # Ottieni le dimensioni finali
            _, c, h, w = x.size()

        return c * h * w, h, w


    def forward(self, x):
        enc = self.encoder(x)
        enc = self.bottleneck(enc)  # ✅ pass through bottleneck

        return enc

class Decoder(nn.Module):
    def __init__(self, in_channels=1, first_deconv_channels = None, base_filters=32, kernel_size: Union[int, Tuple[int, int]] = 2, num_layers=2,
                 stride: Union[int, Tuple[int, int]] = 2,
                 latent_dim=None, flattened_size=None,
                 img_heigth=16, img_width=16, h_enc=2, w_enc=2, activation=nn.ReLU(),
                 flattened=True, double_deconv=False, conv_padding: Union[int, Tuple[int, int]] = 0, conv_kernel_size=3,
                 conv_stride=1, conv_dilation=1, batch_norm=True):
        super(Decoder, self).__init__()

        self.in_channels = in_channels
        self.first_deconv_channels = first_deconv_channels
        self.kernel_size = kernel_size
        self.base_filters = base_filters
        self.num_layers = num_layers
        self.h = img_heigth
        self.w = img_width
        self.h_enc = h_enc
        self.w_enc = w_enc
        self.act = activation
        self.stride = stride
        self.latent_dim = latent_dim
        self.conv_stride = conv_stride
        self.conv_dilation = conv_dilation
        self.conv_kernel_size = conv_kernel_size
        self.conv_padding = conv_padding
        self.double_deconv = double_deconv
        self.flattened = flattened
        self.flattened_size = flattened_size
        self.batch_norm = batch_norm
        self.act_str = {'relu':'relu', 'lrelu':'leaky_relu', 'elu':'relu'}[{v: k for k, v in activation_dict.items()}.get(activation, 'relu').lower()]

        # Compute output paddings for deconv
        output_paddings = self._compute_output_padding(
            Lb=[self.h_enc, self.w_enc], L_target=[self.h, self.w],
            num_layers=self.num_layers, stride=self.stride)

        # Build decoder layers
        decoder_layers = []
        in_f = self.first_deconv_channels  # start from bottleneck channels

        for i in range(self.num_layers):

            if i == self.num_layers - 1:
                out_f = self.base_filters
            else:
                out_f = in_f // 2

            reshape = (i == 0)

            decoder_layers.append((
                f'dec_lay_{i+1}',
                deconv_block(
                    in_f=in_f,
                    out_f=out_f, # General rule: halve the number of filters at each layer

                    flattened=flattened, latent_dim=latent_dim, flattened_size=flattened_size,
                    first_deconv_channels=first_deconv_channels,
                    h_enc=h_enc, w_enc=w_enc,    # For the reshape layer if reshape=True

                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    activation=self.act,
                    output_padding=output_paddings[i],
                    double_deconv=self.double_deconv,
                    conv_kernel_size=self.conv_kernel_size,
                    conv_padding=self.conv_padding,
                    conv_stride=self.conv_stride,
                    conv_dilation=self.conv_dilation,
                    reshape = reshape
                )
            ))
            in_f = out_f

        self.decoder = nn.Sequential(OrderedDict(decoder_layers))

        # Final reconstruction layer — map base_filters → input channels (1 by default)
        self.decoder_out = nn.Conv2d(self.base_filters, self.in_channels, kernel_size=1)

        self.init_kaiming_normal()

    def _compute_output_padding(self, Lb, L_target, num_layers, stride):
        if isinstance(stride, int):
            stride_h, stride_w = stride, stride
        else:
            stride_h, stride_w = stride

        diff_h = L_target[0] - (Lb[0] * (stride_h ** num_layers))
        diff_w = L_target[1] - (Lb[1] * (stride_w ** num_layers))

        ops = []
        for i in range(num_layers):
            weight_h = stride_h ** (num_layers - 1 - i)
            weight_w = stride_w ** (num_layers - 1 - i)
            op_h = diff_h // weight_h
            op_w = diff_w // weight_w
            ops.append((int(op_h), int(op_w)))
            diff_h = diff_h % weight_h
            diff_w = diff_w % weight_w
        return ops

    def init_kaiming_normal(self, mode='fan_in'):
        print('Initializing conv2d weights with Kaiming He normal')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode=mode, nonlinearity=self.act_str)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.decoder(x)
        out = self.decoder_out(out)
        return out

# define the NN architecture
class CONV_AE2D(nn.Module):
    def __init__(self, cfg):
        super(CONV_AE2D, self).__init__()
        # Features on height, time on width

        self.cfg = cfg
        model_cfg = cfg.model

        self.in_channels = cfg.model.aux_channels  # or cfg.model.in_channel if set there
        self.kernel_size = model_cfg.kernel_size
        self.base_filters = model_cfg.base_filters if not isinstance(model_cfg.base_filters, str) else cfg.dataset.n_features
        self.num_layers = model_cfg.num_layers
        self.act = activation_dict.get(model_cfg.get("activation", None), None)
        self.bottleneck_act = activation_dict.get(model_cfg.get("bottleneck_activation", None), None)
        self.pool = model_cfg.pool
        self.flattened = model_cfg.flattened
        self.compression_factor = model_cfg.compression_factor if model_cfg.get('compression_factor_on_inputs', None) is None else model_cfg.get('compression_factor')
        self.compression_type = 'on_inputs' if model_cfg.get('compression_factor_on_inputs', None) is not None else 'on_features'
        self.increasing = model_cfg.increasing
        self.dilation = model_cfg.dilation
        if self.cfg.opt.get("fine_tuning",0) and self.cfg.opt.get("opt.fine_tuning_mode") == "adaptive_layer":
            # Chage the output size according to the pre-trained model to enable putput adaptation layer
            self.h = len(torch.load(cfg.opt.checkpoint_path)['cfg'].dataset.feats)
            self.w = torch.load(cfg.opt.checkpoint_path)['cfg'].dataset.seq_in_length
        else:
            self.h = cfg.dataset.n_features  # or model_cfg.heigth if set there
            self.w = cfg.dataset.seq_in_length
        self.h = cfg.dataset.n_features  # or model_cfg.heigth if set there
        self.w = cfg.dataset.seq_in_length
        self.halve_time = model_cfg.halve_time  # if True, halve only the time dimension when pooling
        self.halve_features = model_cfg.halve_features  # if True, halve only the feature dimension when pooling
        self.stride = model_cfg.stride if not model_cfg.pool else 1
        self.halve_both = model_cfg.halve_both
        self.double_deconv = model_cfg.double_deconv

        if self.halve_both:
            # halve both height (time) and width (features)
            self.pool_ks = (2, 2)
            self.pool_stride = (2, 2)

        elif self.halve_time and not self.halve_features:
            # halve only the time dimension (height)
            self.pool_ks = (1, 2)
            self.pool_stride = (1, 2)

        elif self.halve_features and not self.halve_time:
            # halve only the feature dimension (width)
            self.pool_ks = (2, 1)
            self.pool_stride = (2, 1)

        elif self.halve_time and self.halve_features:
            # halve both
            self.pool_ks = (2, 2)
            self.pool_stride = (2, 2)

        else:
            # no halving
            self.pool_ks = (1, 1)
            self.pool_stride = (1, 1)


        #self.padding = int((self.dilation * (self.kernel_size - 1) / 2))
        #Lout=[Lin+2⋅P−D⋅(K−1)−1]+1
        # 2P=D⋅(K−1)
        self.padding_h = self._compute_same_padding(
            in_size=self.h,
            kernel_size=self.kernel_size,
            stride=self.stride,
            dilation=self.dilation
        )
        self.padding_w = self._compute_same_padding(
            in_size=self.w,
            kernel_size=self.kernel_size,
            stride=self.stride,
            dilation=self.dilation
        )
        self.padding = (self.padding_h, self.padding_w)

        self.encoder = Encoder(self.in_channels, base_filters= self.base_filters, kernel_size=self.kernel_size,
                               num_layers=self.num_layers,
                               pool_ks = self.pool_ks, pool_stride = self.pool_stride,
                               compression_factor = self.compression_factor,
                               img_heigth=self.h, img_width=self.w, activation=self.act,
                               padding=self.padding, flattened=self.flattened,
                               dilation=self.dilation, bottleneck_act=self.bottleneck_act, compression_type=self.compression_type)
        self.flattened_size = self.encoder.flattened_size
        self.latent_dim = self.encoder.latent_dim
        self.cfg.model.flattened_size = self.flattened_size
        self.decoder = Decoder(in_channels=self.in_channels, first_deconv_channels=self.encoder.last_layers_channels,
                               base_filters= self.base_filters, kernel_size=self.pool_ks, num_layers=self.num_layers,
                               stride=self.pool_stride,
                               latent_dim=self.latent_dim, flattened_size=self.flattened_size,
                               img_heigth=self.h, img_width=self.w, h_enc=self.encoder.h_enc, w_enc=self.encoder.w_enc,
                               activation=self.act, flattened=self.flattened,
                               double_deconv=self.double_deconv, conv_padding=self.padding,
                               conv_kernel_size=self.kernel_size, conv_stride=self.stride, conv_dilation=self.dilation)


    def _compute_same_padding(self, in_size, kernel_size, stride=1, dilation=1):
        """Compute padding (top/bottom or left/right) to keep same output size."""
        pad = ((stride - 1) * in_size - stride + dilation * (kernel_size - 1) + 1) / 2
        return math.floor(pad)

    def forward(self, x):
        if hasattr(self, "input_adapter") and self.input_adapter is not None:
            x = self.input_adapter(x)
        enc = self.encoder(x)
        out = self.decoder(enc)
        if hasattr(self, "output_adapter") and self.output_adapter is not None:
            out = self.output_adapter(out)
        return out
