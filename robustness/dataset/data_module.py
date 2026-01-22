import numpy as np
import pandas as pd
from omegaconf import DictConfig, ListConfig
import pytorch_lightning as pl

from torch.utils.data import DataLoader, ConcatDataset, RandomSampler
from sklearn.preprocessing import StandardScaler

from robustness.dataset.dataset import TimeSeriesDataset
from robustness.dataset.data_types import Config
from robustness.dataset.wombats import AnomalyConfig


class DataModule(pl.LightningDataModule):
    def __init__(self, cfg: Config, mode: str = "train"):
        super().__init__()
        self.cfg = cfg
        self.mode = mode

    def setup(self, stage: str | None = None) -> None:
        data = pd.read_csv(self.cfg.dataset.csv_path).values.astype(np.float32)

        # chunk temporali
        chunks = np.array_split(data, self.cfg.dataset.n_chunks)

        n_test = int(len(chunks) * self.cfg.dataset.test_chunk_ratio)
        test_chunks = chunks[-n_test:]
        trainval_chunks = chunks[:-n_test]

        np.random.shuffle(trainval_chunks)

        n_val = int(len(trainval_chunks) * self.cfg.dataset.val_ratio)
        val_chunks = trainval_chunks[:n_val]
        train_chunks = trainval_chunks[n_val:]

        train_data = np.concatenate(train_chunks)
        val_data = np.concatenate(val_chunks)
        test_data = np.concatenate(test_chunks)

        # scaler SOLO su train
        self.scaler = StandardScaler().fit(train_data)

        W = self.cfg.dataset.seq_in_length

        # riferimento WOMBATS
        idx = np.random.choice(len(train_data) - W, size=256, replace=False)
        Xok_ref = np.stack([train_data[i : i + W, 0] for i in idx])

        # ===== TRAIN =====
        if self.mode == "train":
            self.train_ds = TimeSeriesDataset(
                train_data,
                W,
                self.cfg.dataset.stride,
                scaler=self.scaler,
            )

            val_clean = TimeSeriesDataset(
                val_data,
                W,
                self.cfg.dataset.stride,
                scaler=self.scaler,
            )

            anomaly_cfg: AnomalyConfig = {
                "ratio": self.cfg.dataset.val_anomaly_ratio,
                "delta_range": (
                    self.cfg.dataset.delta_min,
                    self.cfg.dataset.delta_max,
                ),
            }

            val_anom = TimeSeriesDataset(
                val_data,
                W,
                self.cfg.dataset.stride,
                scaler=self.scaler,
                anomaly_cfg=anomaly_cfg,
                Xok_ref=Xok_ref,
            )

            self.val_ds = ConcatDataset([val_clean, val_anom])

            self.val_sampler = (
                RandomSampler(self.val_ds)
                if self.cfg.dataset.val_shuffle_augmented
                else None
            )

        # ===== TEST =====
        elif self.mode == "test":
            self.test_ds = TimeSeriesDataset(
                test_data,
                W,
                self.cfg.dataset.stride,
                scaler=self.scaler,
            )

    def train_dataloader(self) -> DataLoader:
        if self.mode != "train":
            raise RuntimeError("train_dataloader chiamato in mode != train")

        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.dataset.batch_size,
            shuffle=True,
            num_workers=self.cfg.dataset.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        if self.mode != "train":
            raise RuntimeError("val_dataloader chiamato in mode != train")

        return DataLoader(
            self.val_ds,
            batch_size=self.cfg.dataset.batch_size,
            sampler=self.val_sampler,
            shuffle=self.val_sampler is None,
            num_workers=self.cfg.dataset.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        if self.mode != "test":
            raise RuntimeError("test_dataloader chiamato in mode != test")

        return DataLoader(
            self.test_ds,
            batch_size=self.cfg.dataset.batch_size,
            shuffle=False,
            num_workers=self.cfg.dataset.num_workers,
            pin_memory=True,
        )
