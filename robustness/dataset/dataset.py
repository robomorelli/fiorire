import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        csv_path,
        seq_len,
        stride=1,
        mean=None,
        std=None
    ):
        """
        CSV shape: [T, N]
        Output: [1, N, seq_len]
        """
        self.seq_len = seq_len
        self.stride = stride

        # Load CSV
        data = pd.read_csv(csv_path).values.astype(np.float32)
        # data: [T, N]

        if mean is None:
            self.mean = data.mean(axis=0, keepdims=True)
            self.std = data.std(axis=0, keepdims=True) + 1e-8
        else:
            self.mean = mean
            self.std = std

        data = (data - self.mean) / self.std
        self.data = data

        self.indices = list(
            range(0, len(data) - seq_len + 1, stride)
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]

        # [L, N]
        window = self.data[i:i + self.seq_len]

        # → [N, L]
        window = window.T

        # → [1, N, L]
        window = np.expand_dims(window, axis=0)

        return torch.from_numpy(window)
