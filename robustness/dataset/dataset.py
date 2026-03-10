import torch
from torch import Tensor
from numpy.typing import NDArray
import numpy as np
from torch.utils.data import Dataset
from typing import Optional
from sklearn.preprocessing import StandardScaler
from robustness.dataset.wombats import apply_random_wombats_anomaly


class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        data: NDArray,  # [T, F]
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

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        i = self.indices[idx]
        window = self.data[i : i + self.seq_len]  # [W, F]

        if self.scaler is not None:
            window = self.scaler.transform(window)

        window = np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0)

        if self.delta_range is not None:
            window, mask = self._inject_anomaly(window)
            label = mask  # [W]
        else:
            label = np.zeros(self.seq_len, dtype=np.int64)  # [W]

        # [W, F] → [1, F, W]
        window = torch.from_numpy(window.T).unsqueeze(0).float()
        label = torch.from_numpy(label)  # [W]
        return window, label

    def _inject_anomaly(self, window: NDArray) -> tuple[NDArray, NDArray]:
        """
        Applica una anomalia WOMBATS canale-wise.
        Restituisce (window, mask) dove mask[t]=1 se il timestep t è stato modificato.
        """
        # guard obbligatori (per pylance)
        if self.delta_range is None or self.Xok_ref is None:
            return window, np.zeros(window.shape[0], dtype=np.int64)

        W, F = window.shape
        feature = np.random.randint(F)
        original_col = window[:, feature].copy()

        window[:, feature] = apply_random_wombats_anomaly(
            signal=window[:, feature],
            Xok_ref=self.Xok_ref[:, :, feature],
            delta_range=self.delta_range,
        )

        mask = (~np.isclose(window[:, feature], original_col)).astype(np.int64)  # [W]
        return window, mask
