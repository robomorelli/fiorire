import torch.nn as nn

from config import activation_dict
from models.utils.layers import conv_block, deconv_block
import torch

torch.manual_seed(0)

class Encoder(nn.Module):
    def __init__(self, in_channel=1, kernel_size=3, padding=1, dilation=1, activation=nn.ReLU(),
                 filter_num_list=None, latent_dim=10,
                 img_heigth=16, img_width=16, flattened=True):
        super(Encoder, self).__init__()

        self.nn_enc = nn.Sequential()

        if filter_num_list is None:
            self.filter_num_list = [1, 32, 64]

        self.in_channel = in_channel
        self.kernel_size = kernel_size
        self.filter_num_list = filter_num_list
        self.latent_dim = latent_dim
        self.h = img_heigth
        self.w = img_width
        self.act = activation
        self.flattened = flattened
        self.padding = padding

        for i, num in enumerate(self.filter_num_list):
            if i + 2 == len(self.filter_num_list):
                self.nn_enc.add_module('enc_lay_{}'.format(i), conv_block(num, self.filter_num_list[i + 1],
                                                                          self.kernel_size, dilation=dilation, activation=self.act,
                                                                          padding=self.padding))
                break
            self.nn_enc.add_module('enc_lay_{}'.format(i), conv_block(num, self.filter_num_list[i+1],
                                                                  self.kernel_size, dilation=dilation, activation=self.act,
                                                                      padding=self.padding))

        self.flattened_size, self.h_enc, self.w_enc = self._get_final_flattened_size()

        if self.flattened:
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
            enc = enc.view(-1, self.flattened_size)
            enc = self.encoder_layer(enc)
        return enc


class Decoder(nn.Module):
    def __init__(self, in_channel=1, kernel_size=2, stride=2, activation=nn.ReLU(),
                 filter_num_list=None, latent_dim=10, flattened_size=None,
                 img_heigth=16, img_width=16, h_enc=2, w_enc=2, flattened=True):
        super(Decoder, self).__init__()

        self.nn_dec = nn.Sequential()

        if filter_num_list is None:
            self.filter_num_list = [32, 64]
        self.in_channel = in_channel
        self.kernel_size = kernel_size
        self.filter_num_list = filter_num_list
        self.latent_dim = latent_dim
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

        if self.flattened:
            self.reshape = nn.Linear(self.latent_dim, self.flattened_size)

        # calcolo output_padding come metodo interno
        output_padding_h = self._compute_output_padding(self.h_enc, self.h, self.n_layers)
        output_padding_w = self._compute_output_padding(self.w_enc, self.w, self.n_layers)

        for i, num in enumerate(self.filter_num_list):
            if i + 2 == len(self.filter_num_list):
                self.nn_dec.add_module(
                    f'dec_lay_{i}',
                    deconv_block(
                        num,
                        self.filter_num_list[i + 1],
                        kernel_size=self.kernel_size,
                        stride=self.stride,
                        activation=self.act,
                        output_padding=(output_padding_h[i], output_padding_w[i])
                    )
                )
                break
            self.nn_dec.add_module(
                f'dec_lay_{i}',
                deconv_block(
                    num,
                    self.filter_num_list[i + 1],
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    activation=self.act,
                    output_padding=(output_padding_h[i], output_padding_w[i])
                )
            )

        self.decoder_layer = nn.Conv2d(self.filter_num_list[i+1], self.in_channel, kernel_size=1)
        self.init_kaiming_normal()

    def _compute_output_padding(self, Lb, L_target, n_layers):
        """
        Calcola output_padding per ogni layer di ConvTranspose2d.
        """
        diff = L_target - (Lb * (self.stride ** n_layers))
        ops = []
        for i in range(n_layers):
            weight = self.stride ** (n_layers - 1 - i)
            op_i = diff // weight
            ops.append(int(op_i))
            diff = diff % weight
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

        self.cfg = cfg
        model_cfg = cfg.model

        self.in_channel = cfg.model.aux_channels  # or cfg.model.in_channel if set there
        self.kernel_size = model_cfg.kernel_size
        self.filter_num = model_cfg.filter_num
        self.n_layers = model_cfg.n_layers
        self.act = activation_dict[model_cfg.activation]
        self.pool = model_cfg.pool
        self.flattened = model_cfg.flattened
        self.increasing = model_cfg.increasing
        self.dilation = model_cfg.dilation
        self.latent_dim = model_cfg.latent_dim
        self.h = cfg.dataset.n_features  # or model_cfg.heigth if set there
        self.w = cfg.dataset.seq_in_length
        self.stride = model_cfg.stride if not model_cfg.pool else 1

        #self.padding = int((self.dilation * (self.kernel_size - 1) / 2))
        #Lout=[Lin+2⋅P−D⋅(K−1)−1]+1
        # 2P=D⋅(K−1)
        self.padding = int((model_cfg.dilation * (model_cfg.kernel_size - 1)) / 2)

        if self.increasing:
            self.filter_num_list = [int(self.filter_num * ((ix + 1) * 2)) for ix in range(self.n_layers)]
        else:
            self.filter_num_list = [int(self.filter_num / ((ix + 1)*2)) for ix in range(self.n_layers)]

        self.filter_num_list = [self.in_channel] + [self.filter_num] + self.filter_num_list

        self.encoder = Encoder(self.in_channel, kernel_size=self.kernel_size, filter_num_list=self.filter_num_list,
                               latent_dim=self.latent_dim,
                               img_heigth=self.h, img_width=self.w, activation=self.act, padding=self.padding, flattened=self.flattened,
                               dilation=self.dilation)
        self.flattened_size = self.encoder.flattened_size
        self.decoder = Decoder(self.in_channel, kernel_size=2, filter_num_list=self.filter_num_list,
                               latent_dim=self.latent_dim, flattened_size=self.flattened_size,
                               img_heigth=self.h, img_width=self.w, h_enc=self.encoder.h_enc, w_enc=self.encoder.w_enc,
                               activation=self.act, flattened=self.flattened)

    def forward(self, x):
        enc = self.encoder(x)
        out = self.decoder(enc)
        return out