
import os
import torch
import torch.nn as nn
# from .early_stopping import *
from json import dump
import torch
from tqdm import tqdm
torch.manual_seed(0)

class Encoder(nn.Module):
    def __init__(self, seq_in, n_features, embedding_size, latent_dim, n_layers_1=2,
                 n_layers_2=1, activation=None):
        super().__init__()

        # self.seq_len = seq_in
        self.n_features = n_features  # The number of expected features(= dimension size) in the input x
        self.embedding_size = embedding_size  # the number of features in the embedded points of the inputs' number of features
        self.n_layers_1 = n_layers_1
        self.n_layers_2 = n_layers_2
        self.hidden_size = (2 * embedding_size)  # The number of features in the hidden state h
        self.latent_dim = latent_dim
        self.activation = activation

        self.LSTMenc = nn.LSTM(
            input_size=n_features,
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
        self.enc = nn.Linear(embedding_size, self.latent_dim)

        if self.activation is not None:
            self.activation = activation

    def forward(self, x):
        # Inputs: input, (h_0, c_0). -> If (h_0, c_0) is not provided, both h_0 and c_0 default to zero.
        # x is the output and so the size is > (batch, seq_len, hidden_size)
        # x, (_,_) = self.LSTM1(x)
        x, (hidden_state, cell_state) = self.LSTMenc(x)
        x, (hidden_state, cell_state) = self.LSTM1(x) ### to switch to x because it needs repeated sequence
        # x[0,-1,:]==hidden_state[-1,0,:] with x ([32, 10, 64]) and h_state 1, 32, 64])
        last_lstm_layer_hidden_state = hidden_state[-1, :, :] #take only the last layer
        #we need hidden state only here because is like our encoding of the time series
        enc = self.enc(last_lstm_layer_hidden_state)
        if self.activation is not None:
            self.activation(x)
        return enc

# (2) Decoder
class Decoder(nn.Module):
    def __init__(self, seq_out, embedding_size, output_size, latent_dim, n_layers_1=2,  n_layers_2=1):
        super().__init__()

        self.seq_len = seq_out
        self.embedding_size = embedding_size
        self.hidden_size = (2 * embedding_size)
        self.n_layers_1 = n_layers_1
        self.n_layers_2 = n_layers_2
        self.output_size = output_size
        self.latent_dim = latent_dim

        self.LSTMdec = nn.LSTM(
            input_size=latent_dim,
            hidden_size=embedding_size,
            num_layers=n_layers_2,
            batch_first=True
        )

        self.LSTM1 = nn.LSTM(
            input_size=embedding_size,
            hidden_size=self.hidden_size,
            num_layers=n_layers_1,
            batch_first=True
        )
        self.fc = nn.Linear(self.hidden_size, output_size)

    def forward(self, x):
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1) # x[0,0,:]==x[0,1,:] ## we nedd to repeat to have an output of secquences (how is our target)
        x, (hidden_state, cell_state) = self.LSTMdec(x)
        x, (_, _) = self.LSTM1(x)
        x = x.reshape((-1, self.seq_len, self.hidden_size))    #fc layer input (input_size, output_size) but if you have (batch, seq_len, input_size) it takes the same operation batch by bath and for
        # each sequence
        # we use the output to target a regression o a label
        out = self.fc(x) #it needs ([32, n, 64]) because in the next operation needs to output a sequence of n
                        #if you don't reshape with sequence lenght in dimension 1 we don'n have out.size = [batch,n , n_features)
                        #also for this we need: x = x.unsqueeze(1).repeat(1, self.seq_len, 1) >>> lstm output >>> reshape
        return out

class LSTM_AE(nn.Module):
    def __init__(self, cfg, **kwargs):
        super().__init__()

        self.seq_in_length = kwargs['seq_in_length']
        self.n_features = kwargs['n_features']
        self.embedding_dim = kwargs['embedding_dim']
        self.latent_dim = kwargs['latent_dim']
        self.n_layers_1 = kwargs['n_layers_1']
        self.n_layers_2 = kwargs['n_layers_2']
        self.seq_out_length = kwargs['seq_in_length']
        self.output_size = kwargs['n_features']
        self.activation = kwargs['activation']

        self.encoder = Encoder(self.seq_in_length, self.n_features,
                               self.embedding_dim, self.latent_dim, self.n_layers_1, self.n_layers_2, self.activation)
        self.decoder = Decoder(self.seq_out_length, self.embedding_dim,
                               self.output_size, self.latent_dim, self.n_layers_1, self.n_layers_2)

    def forward(self, x):
        #torch.manual_seed(0)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return x, encoded, decoded

    def encode(self, x):
        self.eval()
        encoded = self.encoder(x)
        return encoded

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
