import torch.nn as nn
import torch
from config import activation_dict
from models.utils.layers import conv_block, deconv_block
from typing import Union, Tuple
import math

torch.manual_seed(0)

class Encoder(nn.Module):
    def __init__(self, in_channel=1, kernel_size=3, padding: Union[int,
                 Tuple[int, int]] = 1,
                 dilation:Union[int, Tuple[int, int]] = 1,
                 pool_ks: Union[int, Tuple[int, int]] = 2,
                 pool_stride: Union[int, Tuple[int, int]] = 2, activation=nn.ReLU(),
                 filter_num_list=None, compression_factor = 2,
                 img_heigth=16, img_width=16, flattened=True):
        super(Encoder, self).__init__()

        self.nn_enc = nn.Sequential()

        if filter_num_list is None:
            self.filter_num_list = [1, 32, 64]

        self.in_channel = in_channel
        self.kernel_size = kernel_size
        self.filter_num_list = filter_num_list
        self.compression_factor = compression_factor
        self.h = img_heigth
        self.w = img_width
        self.act = activation
        self.flattened = flattened
        self.padding = padding
        self.pool_ks = pool_ks
        self.pool_stride = pool_stride

        for i, num in enumerate(self.filter_num_list):
            if i + 2 == len(self.filter_num_list):
                self.nn_enc.add_module('enc_lay_{}'.format(i+1), conv_block(num, self.filter_num_list[i + 1],
                                                                          self.kernel_size, dilation=dilation,
                                                                          pool_ks = self.pool_ks,
                                                                          pool_stride = self.pool_stride,
                                                                          activation=self.act,
                                                                          padding=self.padding))
                break
            self.nn_enc.add_module('enc_lay_{}'.format(i+1), conv_block(num, self.filter_num_list[i+1],
                                                                  self.kernel_size, dilation=dilation,
                                                                      pool_ks=self.pool_ks,
                                                                      pool_stride=self.pool_stride,
                                                                      activation=self.act,
                                                                      padding=self.padding))

        self.flattened_size, self.h_enc, self.w_enc = self._get_final_flattened_size()
        self.latent_dim = int(self.flattened_size // self.compression_factor)
        if self.flattened:
            self.flatten = nn.Flatten()
            self.encoder_layer = nn.Linear(self.flattened_size, self.latent_dim)

        self.init_kaiming_normal()

    def init_kaiming_normal(self, mode='fan_in'):
        print('Initializing conv2d weights with Kaiming He normal')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode=mode)
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(
                (1, self.in_channel, self.h, self.w)
            )
            x = self.nn_enc(x)
            _, c, h, w = x.size()
        return c * w * h, h, w

    def forward(self, x):
        enc = self.nn_enc(x)
        if self.flattened:
            enc = self.flatten(enc)
            #enc = enc.view(-1, self.flattened_size)
            enc = self.encoder_layer(enc)
        return enc


class Decoder(nn.Module):
    def __init__(self, in_channel=1, kernel_size: Union[int, Tuple[int, int]] = 2,
                 stride: Union[int, Tuple[int, int]] = 2,
                 activation=nn.ReLU(),
                 filter_num_list=None, latent_dim=None, flattened_size=None,
                 img_heigth=16, img_width=16, h_enc=2, w_enc=2, flattened=True, double_deconv=False, conv_kernel_size=3,
                 conv_padding = 0,
                 conv_stride=1, conv_dilation=1):
        super(Decoder, self).__init__()

        self.nn_dec = nn.Sequential()

        if filter_num_list is None:
            self.filter_num_list = [32, 64]
        self.in_channel = in_channel
        self.kernel_size = kernel_size
        self.filter_num_list = filter_num_list
        self.flattened_size = flattened_size
        self.h = img_heigth
        self.w = img_width
        self.h_enc = h_enc
        self.w_enc = w_enc
        self.filter_num_list = self.filter_num_list[::-1]
        self.act = activation
        self.flattened = flattened
        self.n_layers = len(self.filter_num_list) - 1
        self.stride = stride  # assuming stride of 2 for upsampling, can
        self.latent_dim = latent_dim if latent_dim is not None else Exception("latent_dim not specified")
        self.conv_stride = conv_stride
        self.conv_dilation = conv_dilation
        self.conv_kernel_size = conv_kernel_size
        self.conv_padding = conv_padding

        if self.flattened:
            self.reshape = nn.Linear(self.latent_dim, self.flattened_size)

        # calcolo output_padding come metodo interno
        output_paddings = self._compute_output_padding(Lb=[self.h_enc, self.w_enc], L_target=[self.h, self.w],
                                                        n_layers=self.n_layers, stride=self.stride)

        for i, num in enumerate(self.filter_num_list):
            if i + 2 == len(self.filter_num_list):
                self.nn_dec.add_module(
                    f'dec_lay_{i + 1}',
                    deconv_block(
                        in_f=num,
                        out_f=self.filter_num_list[i + 1],
                        kernel_size=self.kernel_size,
                        stride=self.stride,
                        activation=self.act,
                        output_padding=output_paddings[i],
                        double_deconv=double_deconv,
                        conv_kernel_size = self.conv_kernel_size,  # for the double deconv option
                        conv_padding = self.conv_padding,
                        conv_stride=self.conv_stride,
                        conv_dilation=self.conv_dilation
                    )
                )
                break
            self.nn_dec.add_module(
                f'dec_lay_{i + 1}',
                deconv_block(
                    in_f=num,
                    out_f=self.filter_num_list[i + 1],
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    activation=self.act,
                    output_padding=output_paddings[i],
                    double_deconv=double_deconv,
                    conv_kernel_size = self.conv_kernel_size,  # for the double deconv option
                    conv_padding=self.conv_padding,
                    conv_stride = self.conv_stride,
                    conv_dilation = self.conv_dilation
                )
            )

        self.decoder_layer = nn.Conv2d(self.filter_num_list[i+1], self.in_channel, kernel_size=1)
        self.init_kaiming_normal()

    def _compute_output_padding(self, Lb, L_target, n_layers, stride):
        """
        Compute output_padding for each ConvTranspose2d layer.
        Supports stride as int or tuple (stride_h, stride_w)

        Lb: input size per dimension [H, W] to first deconv layer
        L_target: target output size [H, W]
        n_layers: number of deconv layers
        stride: int or tuple (stride_h, stride_w)

        Returns: list of tuples [(op_h1, op_w1), (op_h2, op_w2), ...]
        """
        if isinstance(stride, int):
            stride_h, stride_w = stride, stride
        else:
            stride_h, stride_w = stride

        diff_h = L_target[0] - (Lb[0] * (stride_h ** n_layers))
        diff_w = L_target[1] - (Lb[1] * (stride_w ** n_layers))

        ops = []
        for i in range(n_layers):
            weight_h = stride_h ** (n_layers - 1 - i)
            weight_w = stride_w ** (n_layers - 1 - i)

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
                m.weight = nn.init.kaiming_normal_(m.weight, mode=mode)
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        if self.flattened:
            x = self.reshape(x)
            x = x.view((-1, self.filter_num_list[0], self.h_enc, self.w_enc))
        dec = self.nn_dec(x)
        out = self.decoder_layer(dec)
        return out

# define the NN architecture
class CONV_AE2D(nn.Module):
    def __init__(self, cfg):
        super(CONV_AE2D, self).__init__()
        # Features on height, time on width

        self.cfg = cfg
        model_cfg = cfg.model

        self.in_channel = cfg.model.aux_channels  # or cfg.model.in_channel if set there
        self.kernel_size = model_cfg.kernel_size
        self.filter_num = model_cfg.filter_num if not isinstance(model_cfg.filter_num, str) else cfg.dataset.n_features
        self.n_layers = model_cfg.n_layers
        self.act = activation_dict[model_cfg.activation]
        self.pool = model_cfg.pool
        self.flattened = model_cfg.flattened
        self.compression_factor = model_cfg.compression_factor
        self.increasing = model_cfg.increasing
        self.dilation = model_cfg.dilation
        self.h = cfg.dataset.n_features # or model_cfg.heigth if set there
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

        if self.increasing:
            self.filter_num_list = [int(self.filter_num * ((ix + 1) * 2)) for ix in range(self.n_layers)]
        else:
            self.filter_num_list = [int(self.filter_num / ((ix + 1)*2)) for ix in range(self.n_layers)]

        if self.filter_num == cfg.dataset.n_features:
            self.filter_num_list = [self.in_channel] + [self.filter_num*2] + self.filter_num_list[1:]
        else:
            self.filter_num_list = [self.in_channel] + [self.filter_num] + self.filter_num_list

        self.encoder = Encoder(self.in_channel, kernel_size=self.kernel_size,
                               pool_ks = self.pool_ks, pool_stride = self.pool_stride,
                               filter_num_list=self.filter_num_list,
                               compression_factor = self.compression_factor,
                               img_heigth=self.h, img_width=self.w, activation=self.act,
                               padding=self.padding, flattened=self.flattened,
                               dilation=self.dilation)
        self.flattened_size = self.encoder.flattened_size
        self.latent_dim = int(self.encoder.flattened_size // self.compression_factor)
        self.decoder = Decoder(self.in_channel, kernel_size=self.pool_ks, stride=self.pool_stride,
                               filter_num_list=self.filter_num_list,
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
        enc = self.encoder(x)
        out = self.decoder(enc)
        return out

