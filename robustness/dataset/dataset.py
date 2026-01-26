import torch
from numpy.typing import NDArray
import numpy as np
from torch.utils.data import Dataset
from typing import Optional
from sklearn.preprocessing import StandardScaler
from robustness.dataset.wombats import apply_random_wombats_anomaly


class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        data: np.ndarray,                  # [T, F]
        seq_len: int,
        stride: int,
        scaler: Optional[StandardScaler] = None,
        delta_range: Optional[tuple[float, float]] = None,
        Xok_ref: Optional[NDArray] = None,  # [N, W]
    ):
        self.data = data
        self.seq_len = seq_len
        self.stride = stride
        self.scaler = scaler
        self.delta_range = delta_range
        self.Xok_ref = Xok_ref

        self.indices = list(range(0, len(data) - self.seq_len + 1, self.stride))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> torch.Tensor:
        i = self.indices[idx]
        window = self.data[i : i + self.seq_len]  # [W, F]

        if self.scaler is not None:
            window = self.scaler.transform(window)

        if self.delta_range is not None:
            window = self._inject_anomaly(window)

        # [W, F] → [1, F, W]
        window = torch.from_numpy(window.T).unsqueeze(0)
        return window.float()

    def _inject_anomaly(self, window: NDArray) -> NDArray:
        """
        Applica una anomalia WOMBATS canale-wise
        """
        # guard obbligatori (per pylance)
        if self.delta_range is None or self.Xok_ref is None:
            return window

        W, F = window.shape
        channel = np.random.randint(F)

        window[:, channel] = apply_random_wombats_anomaly(
            signal=window[:, channel],
            Xok_ref=self.Xok_ref[:, :, channel],
            delta_range=self.delta_range,
        )

        return window

