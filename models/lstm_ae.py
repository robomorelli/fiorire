import torch
import torch.nn as nn

torch.manual_seed(0)


class Encoder(nn.Module):
    def __init__(self, n_features, hidden_dim, latent_dim, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])   # (B, latent_dim)


class Decoder(nn.Module):
    def __init__(self, seq_len, n_features, latent_dim, num_layers):
        super().__init__()
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.n_features = n_features

        # hidden_size = latent_dim: z IS the initial hidden state
        self.lstm = nn.LSTM(n_features, latent_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(latent_dim, n_features)

    def forward(self, z, x=None):
        # z: (B, latent_dim) — used directly as h_0
        # x: (B, T, F)       — ground truth for teacher forcing, None at inference
        h = z.unsqueeze(0).repeat(self.num_layers, 1, 1).contiguous()  # (num_layers, B, latent_dim)
        c = torch.zeros_like(h)

        preds = []

        # first prediction directly from h_0, no LSTM step (Malhotra et al.)
        pred = self.fc(h[-1])           # (B, F)
        preds.append(pred.unsqueeze(1))

        for t in range(1, self.seq_len):
            # teacher forcing: ground truth; inference: own prediction
            inp = x[:, self.seq_len - t, :] if x is not None else pred   # (B, F)
            _, (h, c) = self.lstm(inp.unsqueeze(1), (h, c))
            pred = self.fc(h[-1])       # (B, F)
            preds.append(pred.unsqueeze(1))

        # preds in reverse order [x'[T-1], ..., x'[0]] — flip to original temporal order
        return torch.cat(preds, dim=1).flip(1)      # (B, T, F)


class LSTM_AE(nn.Module):  # noqa: N801
    def __init__(self, cfg):
        super().__init__()
        seq_len    = cfg.dataset.seq_in_length
        n_features = cfg.dataset.n_features
        hidden_dim = cfg.model.hidden_dim
        latent_dim = cfg.model.latent_dim
        num_layers = cfg.model.num_layers

        self.is_fitted = False
        self.encoder = Encoder(n_features, hidden_dim, latent_dim, num_layers)
        self.decoder = Decoder(seq_len, n_features, latent_dim, num_layers)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded, x if self.training else None)
        return decoded

    def encode(self, x):
        self.eval()
        return self.encoder(x)

    def decode(self, z):
        self.eval()
        return self.decoder(z).squeeze()

    def load(self, path):
        self.is_fitted = True
        self.load_state_dict(torch.load(path))