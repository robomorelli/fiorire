import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from robustness.dataset.dataset import TimeSeriesDataset


class DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage=None):
        full_ds = TimeSeriesDataset(
            csv_path=self.cfg.dataset.csv_path,
            seq_len=self.cfg.dataset.seq_in_length,
            stride=self.cfg.dataset.stride
        )

        val_len = int(len(full_ds) * self.cfg.dataset.val_ratio)
        train_len = len(full_ds) - val_len

        self.train_ds = torch.utils.data.Subset(
            full_ds, range(0, train_len)
        )
        self.val_ds = torch.utils.data.Subset(
            full_ds, range(train_len, train_len + val_len)
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.dataset.batch_size,
            shuffle=True,   # shuffle delle finestre è OK
            num_workers=self.cfg.dataset.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.cfg.dataset.batch_size,
            shuffle=False,
            num_workers=self.cfg.dataset.num_workers,
            pin_memory=True
        )
