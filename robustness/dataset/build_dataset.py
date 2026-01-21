import torch
from robustness.dataset.dataset import TimeSeriesDataset

def build_datasets(cfg):
    csv = cfg.dataset.csv_path
    seq_len = cfg.dataset.seq_in_length

    full_ds = TimeSeriesDataset(
        csv_path=csv,
        seq_len=seq_len,
        stride=cfg.dataset.stride
    )

    val_len = int(len(full_ds) * cfg.dataset.val_ratio)
    train_len = len(full_ds) - val_len

    train_ds = torch.utils.data.Subset(
        full_ds, range(0, train_len)
    )
    val_ds = torch.utils.data.Subset(
        full_ds, range(train_len, train_len + val_len)
    )

    return train_ds, val_ds
