import os
import torch.nn as nn
from models.utils.layers import conv_block, deconv_block
import torch
from tqdm import tqdm

class Encoder(nn.Module):
    def __init__(self, in_channel=1, kernel_size=3, filter_num_list=None, latent_dim=10,
                 img_heigth=16, img_width=16, activation=nn.ReLU(), padding=1, flattened=True):
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
        self.padding = padding
        self.flattened = flattened

        for i, num in enumerate(self.filter_num_list):
            if i + 2 == len(self.filter_num_list):
                self.nn_enc.add_module('enc_lay_{}'.format(i), conv_block(num, self.filter_num_list[i + 1],
                                                                          self.kernel_size, activation = self.act,
                                                                          padding=self.padding))
                break
            self.nn_enc.add_module('enc_lay_{}'.format(i), conv_block(num, self.filter_num_list[i+1],
                                                                  self.kernel_size, activation = self.act,
                                                                      padding=self.padding))
        self.flattened_size, self.h_enc, self.w_enc = self._get_final_flattened_size()
        if self.flattened:

            #self.nn_enc.add_module('flat_lay'.format(i), nn.Linear(self.flattened_size, self.latent_dim))
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
            _, c, w, h = x.size()
        return c * w * h, w, h

    def forward(self, x):
        enc = self.nn_enc(x)
        if self.flattened:
            enc = enc.view(-1, self.flattened_size)
            enc = self.encoder_layer(enc)
        #enc = self.act(enc)
        return enc


class Decoder(nn.Module):
    def __init__(self, in_channel=1, kernel_size=3, filter_num_list=None, latent_dim=10, flattened_size=None,
                 img_heigth=16, img_width=16, h_enc=2, w_enc=2, activation=nn.ReLU(), flattened=True):
        super(Decoder, self).__init__()

        self.nn_dec = nn.Sequential()

        if filter_num_list is None:
            self.filter_num_list = [32, 64]
        self.in_channel = in_channel
        self.kernel_size = kernel_size
        self.filter_num_list = filter_num_list
        self.latent_dim = latent_dim
        self.flattened=flattened
        self.flattened_size = flattened_size
        self.h = img_heigth
        self.w = img_width
        self.h_enc = h_enc
        self.w_enc = w_enc
        self.act = activation
        self.filter_num_list = self.filter_num_list[::-1]

        print('FLATTENEDDDD', self.flattened)

        if self.flattened:
            self.reshape = nn.Linear(self.latent_dim, self.flattened_size)

        for i, num in enumerate(self.filter_num_list):
            if i + 2 == len(self.filter_num_list):
                self.nn_dec.add_module('dec_lay_{}'.format(i), deconv_block(num, self.filter_num_list[i+1],
                                                                            activation=self.act))
                break
            self.nn_dec.add_module('dec_lay_{}'.format(i), deconv_block(num, self.filter_num_list[i + 1],
                                                                        activation=self.act))

        self.decoder_layer = nn.Conv2d(self.filter_num_list[i + 1], self.in_channel, kernel_size=1)
        self.init_kaiming_normal()

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
            x = x.view((-1, self.filter_num_list[0], self.h_enc, self.w_enc)) # (bsie, num_list, n_features, -1)
        out = self.nn_dec(x)
        out = self.decoder_layer(out)
        return out

# define the NN architecture
class CONV_AE(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(CONV_AE, self).__init__()

        self.in_channel = kwargs['in_channel']
        self.seq_in_length = kwargs['width']
        self.n_features = kwargs['heigth']
        self.kernel_size = kwargs['kernel_size']
        self.filter_num = kwargs['filter_num']
        self.latent_dim = kwargs['latent_dim']
        self.n_layers = kwargs['n_layers']
        self.act = kwargs['activation']

        if 'padding' in list(kwargs.keys()):
            self.padding = kwargs['padding']
        else:
            self.padding=1
        if 'flattened' in list(kwargs.keys()):
            self.flattened = kwargs['flattened']
        else:
            self.flattened=True
        if 'increasing' in list(kwargs.keys()):
            self.increasing = kwargs['increasing']
        else:
            self.increasing=False #default choice

        self.h = self.n_features # number of features (to transpose respect to the df format)
        self.w = self.seq_in_length  # sequence length (to transpose respect to the df format)

        if self.increasing:
            self.filter_num_list = [int(self.filter_num * ((ix + 1)*2)) for ix in range(self.n_layers)]
        else:
            self.filter_num_list = [int(self.filter_num / ((ix + 1) * 2)) for ix in range(self.n_layers)]

        self.filter_num_list = [self.in_channel] + [self.filter_num] + self.filter_num_list

        self.encoder = Encoder(self.in_channel, kernel_size=self.kernel_size, filter_num_list=self.filter_num_list,
                               latent_dim=self.latent_dim,
                               img_heigth=self.h, img_width=self.w, activation = self.act, padding=self.padding
                               ,flattened=self.flattened)
        self.flattened_size = self.encoder.flattened_size
        self.decoder = Decoder(self.in_channel, kernel_size=self.kernel_size, filter_num_list=self.filter_num_list,
                               latent_dim=self.latent_dim, flattened_size=self.flattened_size,
                               img_heigth=self.h, img_width=self.w, h_enc=self.encoder.h_enc, w_enc=self.encoder.w_enc,
                               activation = self.act, flattened=self.flattened)

    def forward(self, x):
        enc = self.encoder(x)
        out = self.decoder(enc)
        return out
