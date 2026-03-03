import numpy as np
from omegaconf import DictConfig
import pandas as pd
import pytorch_lightning as pl
import torch

from torch.utils.data import DataLoader, ConcatDataset, RandomSampler
from sklearn.preprocessing import StandardScaler

from robustness.dataset.dataset import TimeSeriesDataset


class DataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig, mode: str):
        super().__init__()
        self.cfg = cfg
        self.mode = mode

    def setup(self, stage: str | None = None) -> None:
        rng = np.random.default_rng(42)
        df = pd.read_csv(self.cfg["dataset"]["csv_path"])
        df = df.dropna()
        data = df.values.astype(np.float32)
        T, F = data.shape
        # overwrite - not used here, only in the model
        self.cfg["dataset"]["n_features"] = F

        W = self.cfg["dataset"]["seq_in_length"]
        n_seq_chunk = self.cfg["dataset"]["n_seq_chunk"]

        # stride = 1 per definire le sequenze "canoniche"
        sequences = np.stack(
            [data[i : i + W] for i in range(0, T - W + 1)]
        )  # [N_seq, W, F]

        n_total_seq = len(sequences)
        n_chunks = n_total_seq // n_seq_chunk

        # tronchiamo per avere chunk completi
        sequences = sequences[: n_chunks * n_seq_chunk]

        chunks = np.split(sequences, n_chunks)
        # print("Before shuffle:", [c[0,0,0] for c in chunks[:3]])
        rng.shuffle(chunks)
        # print("After shuffle:", [c[0,0,0] for c in chunks[:3]])
        # list[NDArray] con shape [n_seq_chunk, W, F]

        n_test = int(n_chunks * self.cfg["dataset"]["test_chunk_ratio"])

        test_chunks = chunks[-n_test:]
        trainval_chunks = chunks[:-n_test]
        n_val = int(len(trainval_chunks) * self.cfg["dataset"]["val_ratio"])
        train_chunks = trainval_chunks[:-n_val]
        val_chunks = trainval_chunks[-n_val:]

        # PRIMA costruisci train_data (serve per il fit dello scaler e di Xok_ref)
        train_data = np.concatenate(train_chunks, axis=0).reshape(-1, F)
        val_data = np.concatenate(val_chunks, axis=0).reshape(-1, F)
        test_data = np.concatenate(test_chunks, axis=0).reshape(-1, F)

        # print("Train mean/std:", train_data.mean(), train_data.std())
        # print("Test mean/std:", test_data.mean(), test_data.std())

        self.scaler = StandardScaler().fit(train_data)
        assert self.scaler.scale_ is not None  # per Pylance / mypy
        self.scaler.scale_[self.scaler.scale_ == 0] = 1.0

        train_data_scaled = self.scaler.transform(train_data)
        test_data_scaled = self.scaler.transform(test_data)

        # print("Train scaled mean/std:", train_data_scaled.mean(), train_data_scaled.std())
        # print("Test scaled mean/std:", test_data_scaled.mean(), test_data_scaled.std())

        # Xok_ref costruito SOLO su train_data, scaled
        W = self.cfg["dataset"]["seq_in_length"]
        train_data_scaled = self.scaler.transform(train_data)
        # print(f"Min: {train_data_scaled.min():.3f}, Max: {train_data_scaled.max():.3f}")
        # print(f"1%: {np.percentile(train_data_scaled, 1):.3f}, 99%: {np.percentile(train_data_scaled, 99):.3f}")
        self.Xok_ref_train = np.stack(
            [train_data_scaled[i : i + W] for i in range(len(train_data_scaled) - W + 1)]
        )  # [N_train, W, F]

        if self.mode == "train":
            self.train_ds = TimeSeriesDataset(
                train_data,
                W,
                self.cfg["dataset"]["seq_stride_train"],
                scaler=self.scaler,
            )

            self.val_ds = self._build_dataset_with_anomalies(
                base_data=val_data,
                stride=self.cfg["dataset"]["seq_stride_val"],
                anomaly_ratio=self.cfg["dataset"]["val_anomaly_ratio"],
                delta_range=(
                    self.cfg["dataset"]["delta_min"],
                    self.cfg["dataset"]["delta_max"],
                ),
            )

            self.train_sampler = None
            if self.cfg["dataset"]["shuffle_train"]:
                self.train_sampler = RandomSampler(
                    self.train_ds,
                    replacement=False,
                    generator=torch.Generator().manual_seed(42),
                )

            self.val_sampler = None
            if self.cfg["dataset"]["shuffle_val"]:
                self.val_sampler = RandomSampler(
                    self.val_ds,
                    replacement=False,
                    generator=torch.Generator().manual_seed(42),
                )

        elif self.mode == "test":
            ratio = self.cfg["dataset"]["test_anomaly_ratio"]
            delta_range = (
                self.cfg["dataset"]["delta_min"],
                self.cfg["dataset"]["delta_max"],
            )

            self.test_ds = self._build_dataset_with_anomalies(
                base_data=test_data,
                stride=self.cfg["dataset"]["seq_stride_test"],
                anomaly_ratio=ratio,
                delta_range=delta_range,
            )

            self.test_sampler = None
            if self.cfg["dataset"]["shuffle_test"]:
                self.test_sampler = RandomSampler(
                    self.test_ds,
                    replacement=False,
                    generator=torch.Generator().manual_seed(42),
                )

    def train_dataloader(self) -> DataLoader:
        if self.mode != "train":
            raise RuntimeError("train_dataloader chiamato in mode != train")

        return DataLoader(
            self.train_ds,
            batch_size=self.cfg["opt"]["batch_size"],
            sampler=self.train_sampler,
            shuffle=False,
            num_workers=self.cfg["dataset"]["num_workers"],
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        if self.mode != "train":
            raise RuntimeError("val_dataloader chiamato in mode != train")

        return DataLoader(
            self.val_ds,
            batch_size=self.cfg["opt"]["batch_size"],
            sampler=self.val_sampler,
            shuffle=False,
            num_workers=self.cfg["dataset"]["num_workers"],
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        if self.mode != "test":
            raise RuntimeError("test_dataloader chiamato in mode != test")

        return DataLoader(
            self.test_ds,
            batch_size=self.cfg["opt"]["batch_size"],
            sampler=self.test_sampler,
            shuffle=False,
            num_workers=self.cfg["dataset"]["num_workers"],
            pin_memory=True,
        )

    def _build_dataset_with_anomalies(self, base_data, stride, anomaly_ratio, delta_range=None):
        clean_ds = TimeSeriesDataset(base_data, self.cfg["dataset"]["seq_in_length"], stride, scaler=self.scaler)

        if delta_range is not None and anomaly_ratio > 0:
            anom_ds = TimeSeriesDataset(
                base_data,
                self.cfg["dataset"]["seq_in_length"],
                stride,
                scaler=self.scaler,
                delta_range=delta_range,
                Xok_ref=self.Xok_ref_train,  # ← sempre dal train
            )
            n_anom = int(anomaly_ratio * len(clean_ds))
            idx = np.random.choice(len(anom_ds), size=n_anom, replace=False).tolist()
            return ConcatDataset([clean_ds, torch.utils.data.Subset(anom_ds, idx)])

        return clean_ds

