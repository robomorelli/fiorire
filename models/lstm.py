import torch.nn as nn
import torch

torch.manual_seed(0)

class Encoder(nn.Module):
    def __init__(self, seq_in, seq_out, n_features, output_size,
                 embedding_size, n_layers_1=2, n_layers_2=1):
        super().__init__()

        self.n_features = n_features  # The number of expected features(= dimension size) in the input x
        self.embedding_size = embedding_size  # the number of features in the embedded points of the inputs' number of features
        self.n_layers_1 = n_layers_1
        self.n_layers_2 = n_layers_2
        self.hidden_size = (2 * embedding_size)  # The number of features in the hidden state h
        # self.seq_out = seq_out
        # self.seq_in = seq_in
        self.output_size = output_size

        self.LSTMenc = nn.LSTM(
            input_size=self.n_features,
            hidden_size=self.hidden_size,
            num_layers=self.n_layers_1,
            batch_first=True
        )
        self.LSTM1 = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.embedding_size,
            num_layers=self.n_layers_2,
            batch_first=True
        )

        self.out = nn.Linear(self.embedding_size, self.output_size)
        self.apply(self.weight_init)

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        x, (hidden_state, cell_state) = self.LSTMenc(x)
        x, (hidden_state, cell_state) = self.LSTM1(x)  ### to switch to x because it needs repeated sequence
        out = self.out(x)
        return out


# (2) Decoder
class LSTM(nn.Module):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.cfg = cfg
        model_cfg = cfg.model

        self.embedding_dim = model_cfg.embedding_dim
        self.n_layers_1 = model_cfg.n_layers_cell_1
        self.n_layers_2 = model_cfg.n_layers_cell_2
        self.output_size = model_cfg.output_size if hasattr(model_cfg, 'output_size') else self.n_features

        self.seq_in_length = cfg.dataset.seq_in_length
        self.seq_out_length = cfg.dataset.seq_out_length  # Add in trainer
        self.n_features = cfg.dataset.n_features

        self.encoder = Encoder(self.seq_in_length, self.seq_out_length, self.n_features,
                               self.output_size,
                               self.embedding_dim, self.n_layers_1, self.n_layers_2)

    def forward(self, x):
        # torch.manual_seed(0)
        out = self.encoder(x)
        return out

    def load(self, PATH):
        """
        Loads the model's parameters from the path mentioned
        :param PATH: Should contain pickle file
        :return: None
        """
        self.is_fitted = True
        self.load_state_dict(torch.load(PATH))
