import torch.nn as nn
from config import activation_dict
from models.utils.layers import conv_block, deconv_block, conv_block1D, deconv_block1D
import torch

torch.manual_seed(0)

class Encoder(nn.Module):
    def __init__(self, in_channel=1, length = 16, kernel_size=3, latent_dim = 40, filter_num_list=None,
                  activation=nn.ReLU(),  stride=1, pool=True, padding = 1, flattened=True):
        super(Encoder, self).__init__()

        self.nn_enc = nn.Sequential()
        if filter_num_list is None:
            self.filter_num_list = [1, 32, 64]

        self.in_channel = in_channel
        self.kernel_size = kernel_size
        self.filter_num_list = filter_num_list
        self.l = length
        self.act = activation
        self.pool = pool
        self.stride = stride
        self.padding = padding
        self.flattened = flattened
        self.latent_dim = latent_dim

        for i, num in enumerate(self.filter_num_list):
            if i + 2 == len(self.filter_num_list):
                self.nn_enc.add_module('enc_lay_{}'.format(i), conv_block1D(num, self.filter_num_list[i + 1],
                                                                          self.kernel_size, activation = self.act,
                                                                          pool=self.pool, stride=self.stride, padding=self.padding))
                break
            self.nn_enc.add_module('enc_lay_{}'.format(i), conv_block1D(num, self.filter_num_list[i+1],
                                                                  self.kernel_size, activation = self.act,
                                                                      pool=self.pool, stride=self.stride, padding=self.padding))

        self.flattened_size, self.l_enc = self._get_final_flattened_size()
        if self.flattened:
            self.encoder_layer = nn.Linear(self.flattened_size, self.latent_dim)

        self.init_kaiming_normal()

    def init_kaiming_normal(self, mode='fan_in'):
        print('Initializing conv2d weights with Kaiming He normal')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode=mode)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(
                (1, self.in_channel, self.l)
            )
            x = self.nn_enc(x)
            _, c, w = x.size()
        return c * w, w

    def forward(self, x):
        enc = self.nn_enc(x)
        if self.flattened:
            enc = enc.view(-1, self.flattened_size)
            enc = self.encoder_layer(enc)
        return enc


class Decoder(nn.Module):
    def __init__(self, in_channel=1, kernel_size=3, latent_dim = 60, filter_num_list=None, flattened_size=None,
                 length=16, activation=nn.ReLU(), flattened=True):
        super(Decoder, self).__init__()

        self.nn_dec = nn.Sequential()
        if filter_num_list is None:
            self.filter_num_list = [32, 64]
        self.in_channel = in_channel
        self.kernel_size = kernel_size
        self.filter_num_list = filter_num_list
        self.act = activation
        self.filter_num_list = self.filter_num_list[::-1]
        self.flattened = flattened
        self.flattened_size = flattened_size
        self.length = length
        self.latent_dim = latent_dim

        if self.flattened:
            self.reshape=nn.Linear(self.latent_dim, self.flattened_size)

        for i, num in enumerate(self.filter_num_list):
            if i + 2 == len(self.filter_num_list):
                self.nn_dec.add_module('dec_lay_{}'.format(i), deconv_block1D(num, self.filter_num_list[i+1],
                                                                            activation=self.act))
                break
            self.nn_dec.add_module('dec_lay_{}'.format(i), deconv_block1D(num, self.filter_num_list[i + 1],
                                                                        activation=self.act))
        self.decoder_layer = nn.Conv1d(self.filter_num_list[i + 1], self.in_channel, kernel_size=1)
        self.init_kaiming_normal()

    def init_kaiming_normal(self, mode='fan_in'):
        print('Initializing conv2d weights with Kaiming He normal')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode=mode)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        if self.flattened:
            x = self.reshape(x)
            x = x.view((-1, self.filter_num_list[0], self.length))
        dec = self.nn_dec(x)
        dec = self.decoder_layer(dec)
        return dec

# define the NN architecture
class CONV_AE1D(nn.Module):
    def __init__(self, cfg):
        super(CONV_AE1D, self).__init__()
        self.cfg = cfg
        model_cfg = cfg.model

        self.in_channel = cfg.dataset.n_features  # or cfg.model.in_channel if set there
        self.kernel_size = model_cfg.kernel_size
        self.filter_num = model_cfg.filter_num
        self.n_layers = model_cfg.n_layers
        self.act = activation_dict[model_cfg.activation]
        self.pool = model_cfg.pool
        self.stride = model_cfg.stride if not model_cfg.pool else 1
        self.padding = int((model_cfg.dilation * (model_cfg.kernel_size - 1)) / 2)

        self.increasing = getattr(model_cfg, 'increasing', False)
        self.flattened = getattr(model_cfg, 'flattened', True)
        self.latent_dim = getattr(model_cfg, 'latent_dim', 60)
        self.output_size = model_cfg.output_size if hasattr(model_cfg, 'output_size') else self.n_features

        self.length = cfg.dataset.seq_in_length

        if self.increasing:
            self.filter_num_list = [int(self.filter_num * ((ix + 1) * 2)) for ix in range(self.n_layers)]
        else:
            self.filter_num_list = [int(self.filter_num / ((ix + 1) * 2)) for ix in range(self.n_layers)]

        self.filter_num_list = [self.in_channel] + [self.filter_num] + self.filter_num_list

        self.encoder = Encoder(
            self.in_channel,
            kernel_size=self.kernel_size,
            latent_dim=self.latent_dim,
            filter_num_list=self.filter_num_list,
            length=self.length,
            activation=self.act,
            pool=self.pool,
            stride=self.stride,
            padding=self.padding,
            flattened=self.flattened
        )

        self.flattened_size = self.encoder.flattened_size

        self.decoder = Decoder(
            self.in_channel,
            kernel_size=self.kernel_size,
            latent_dim=self.latent_dim,
            filter_num_list=self.filter_num_list,
            flattened_size=self.flattened_size,
            length=self.encoder.l_enc,
            activation=self.act,
            flattened=self.flattened
        )

    def forward(self, x):
        enc = self.encoder(x)
        out = self.decoder(enc)
        return out