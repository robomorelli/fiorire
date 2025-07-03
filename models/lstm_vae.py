import os
import torch.nn as nn
import torch
from tqdm import tqdm
import sys

from models.utils.layers import *
from models.utils.losses import *
import torch.nn.functional as F


torch.manual_seed(0)

class Encoder_vae(nn.Module):
    def __init__(self, seq_in, no_features, embedding_size, latent_dim, n_layers_1=1,
                 n_layers_2=1):
        super().__init__()

        # self.seq_len = seq_in
        self.no_features = no_features  # The number of expected features(= dimension size) in the input x
        self.embedding_size = embedding_size  # the number of features in the embedded points of the inputs' number of features
        self.n_layers_1 = n_layers_1
        self.n_layers_2 = n_layers_2
        self.hidden_size = (2 * embedding_size)  # The number of features in the hidden state h
        self.latent_dim = latent_dim

        self.LSTMenc = nn.LSTM(
            input_size=no_features,
            hidden_size=self.hidden_size,
            num_layers=n_layers_1,
            batch_first=True
        )
        self.LSTM1 = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=embedding_size,
            num_layers=n_layers_2,
            batch_first=True
        )

        self.mu = nn.Linear(embedding_size, latent_dim)
        self.log_var = nn.Linear(embedding_size, latent_dim)

        #self.act2 = InverseSquareRootLinearUnit()

        def _init_weights(self, module):
            if isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std =1.0)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.Linear):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)


    def forward(self, x):
        x, (hidden_state, cell_state) = self.LSTMenc(x)
        x, (hidden_state, cell_state) = self.LSTM1(x)
        last_lstm_layer_hidden_state = hidden_state[-1, :, :]

        mu = self.mu(last_lstm_layer_hidden_state)
        log_var = self.log_var(last_lstm_layer_hidden_state)
        return mu, log_var



class Decoder_vae(nn.Module):
    def __init__(self, seq_out, embedding_size, output_size, latent_dim, n_layers_1=1,
                 n_layers_2=1, recon_loss_type='custom'):
        super().__init__()

        self.seq_len = seq_out
        self.embedding_size = embedding_size
        self.hidden_size = (2 * embedding_size)
        self.n_layers_1 = n_layers_1
        self.n_layers_2 = n_layers_2
        self.output_size = output_size
        self.latent_dim = latent_dim
        self.Nf_lognorm = output_size
        self.recon_loss_type = recon_loss_type

        self.act2 = InverseSquareRootLinearUnit()
        self.act3 = ClippedTanh()

        self.LSTMdec = nn.LSTM(
            input_size=latent_dim,
            hidden_size=embedding_size,
            num_layers=n_layers_1,
            batch_first=True
        )
        self.LSTM1 = nn.LSTM(
            input_size=embedding_size,
            hidden_size=self.hidden_size,
            num_layers=n_layers_2,
            batch_first=True
        )

        self.par1 = nn.Linear(self.hidden_size, output_size)
        if self.recon_loss_type == 'custom':
            self.par2 = nn.Linear(self.hidden_size, self.Nf_lognorm)
            self.par3 = nn.Linear(self.hidden_size, self.Nf_lognorm)


    def forward(self, z):

        z = z.unsqueeze(1).repeat(1, self.seq_len, 1) #x[0,0,:]==x[0,1,:] ## we nedd to repeat to have an output of secquences (how is our target)
        x, (hidden_state, cell_state) = self.LSTMdec(z)
        x, (_, _) = self.LSTM1(x)
        x = x.reshape((-1, self.seq_len, self.hidden_size))    #fc layer input (input_size, output_size) but if you have (batch, seq_len, input_size) it takes the same operation batch by bath and for
        # each sequence
        # we use the output to target a regression o a label
        # out = self.fc(x) #it needs ([32, n, 64]) because in the next operation needs to output a sequence of n
                        #if you don't reshape with sequence lenght in dimension 1 we don'n have out.size = [batch,n , n_features)
                        #also for this we need: x = x.unsqueeze(1).repeat(1, self.seq_len, 1) >>> lstm output >>> reshape

        if self.recon_loss_type == 'custom':
            par2 = self.par2(x)
            par3 = self.par3(x)
            return self.par1(x), self.act2(par2), self.act3(par3)
        else:
            return self.par1(x)

class LSTM_VAE(nn.Module):
    def __init__(self, cfg, **kwargs):
        super().__init__()

        self.seq_in = kwargs['sequence_length']
        self.seq_out = kwargs['sequence_length']
        self.no_features = kwargs['no_features']
        self.embedding_dim = kwargs['embedding_dim']
        self.hidden_size = 2*kwargs['embedding_dim']
        self.n_layers_1 = kwargs['n_layers_1']
        self.n_layers_2 = kwargs['n_layers_2']
        self.output_size = kwargs['no_features']
        self.latent_dim = kwargs['latent_dim']
        self.recon_loss_type = kwargs['recon_loss_type']

        self.encoder = Encoder_vae(self.seq_in, self.no_features
                                   , self.embedding_dim, self.latent_dim, self.n_layers_1,
                                   self.n_layers_2)
        self.decoder = Decoder_vae(self.seq_out, self.embedding_dim
                                   , self.output_size, self.latent_dim, self.n_layers_1,
                                   self.n_layers_2, recon_loss_type=self.recon_loss_type)

        self.apply(self.weight_init)

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def sample(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    #def forward(self, x):
        ##torch.manual_seed(0)
        #mu, log_var = self.encoder(x)
        ##sigma = torch.exp(0.5*log_var)
        #z = self.sample(mu, log_var)
        #out = self.decoder(z)

        #return x, mu, log_var, out

    def forward(self, x):
        torch.manual_seed(0)
        mu, log_var = self.encoder(x)
        z = self.sample(mu, log_var)
        pars = self.decoder(z)

        return x, mu, log_var, pars

    def encode(self, x):
        self.eval()
        mu, log_var = self.encoder(x)
        return mu, log_var

    def decode(self, x):
        self.eval()
        decoded = self.decoder(x)
        squeezed_decoded = decoded.squeeze()
        return squeezed_decoded

    def load(self, PATH):
        """
        Loads the model's parameters from the path mentioned
        :param PATH: Should contain pickle file
        :return: None
        """
        self.is_fitted = True
        self.load_state_dict(torch.load(PATH))
