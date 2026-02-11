from typing import Optional
import numpy as np
from omegaconf import DictConfig
import pandas as pd
import pytorch_lightning as pl
import torch

from torch.utils.data import DataLoader, ConcatDataset, RandomSampler
from sklearn.preprocessing import StandardScaler

from robustness.dataset.dataset import TimeSeriesDataset


class DataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig, mode: str, test_mode: Optional[str]):
        super().__init__()
        self.cfg = cfg
        self.mode = mode
        self.test_mode = test_mode

    def setup(self, stage: str | None = None) -> None:

        df = pd.read_csv(self.cfg["dataset"]["csv_path"])
        df = df.dropna()
        data = df.values.astype(np.float32)
        T, F = data.shape
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
        # list[NDArray] con shape [n_seq_chunk, W, F]

        n_test = int(n_chunks * self.cfg["dataset"]["test_chunk_ratio"])

        test_chunks = chunks[-n_test:]
        trainval_chunks = chunks[:-n_test]

        n_val = int(len(trainval_chunks) * self.cfg["dataset"]["val_ratio"])

        val_chunks = trainval_chunks[:n_val]
        train_chunks = trainval_chunks[n_val:]

        # shuffle solo il validation
        np.random.shuffle(val_chunks)

        # torniamo a una shape (W, F)
        train_data = np.concatenate(train_chunks, axis=0).reshape(-1, F)
        val_data = np.concatenate(val_chunks, axis=0).reshape(-1, F)
        test_data = np.concatenate(test_chunks, axis=0).reshape(-1, F)

        self.scaler = StandardScaler().fit(train_data)
        assert self.scaler.scale_ is not None  # per Pylance / mypy
        # evita divisioni per zero
        self.scaler.scale_[self.scaler.scale_ == 0] = 1.0

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

            self.val_sampler = RandomSampler(
                self.val_ds,
                replacement=False,
                generator=torch.Generator().manual_seed(42),
            )

        elif self.mode == "test":
            if self.test_mode == "anom":
                ratio = self.cfg["dataset"]["test_anomaly_ratio"]
                delta_range = (
                    self.cfg["dataset"]["delta_min"],
                    self.cfg["dataset"]["delta_max"],
                )
            else:
                ratio = 0
                delta_range = None

            self.test_ds = self._build_dataset_with_anomalies(
                base_data=test_data,
                stride=self.cfg["dataset"]["seq_stride_test"],
                anomaly_ratio=ratio,
                delta_range=delta_range,
            )

    def train_dataloader(self) -> DataLoader:
        if self.mode != "train":
            raise RuntimeError("train_dataloader chiamato in mode != train")

        return DataLoader(
            self.train_ds,
            batch_size=self.cfg["opt"]["batch_size"],
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
            shuffle=False,
            num_workers=self.cfg["dataset"]["num_workers"],
            pin_memory=True,
        )

    def _build_dataset_with_anomalies(
        self,
        base_data: np.ndarray,
        stride: int,
        anomaly_ratio: float,
        delta_range: Optional[tuple[float, float]] = None,
    ):
        """
        Costruisce un dataset concatenando i dati 'clean' con un subset di anomalie.
        - base_data: dati originali
        - stride: stride per TimeSeriesDataset
        - anomaly_ratio: percentuale di anomalie da aggiungere
        - Xok_ref: riferimento per perturbazioni
        - delta_range: range di perturbazioni
        """

        # dataset clean
        clean_ds = TimeSeriesDataset(
            base_data,
            self.cfg["dataset"]["seq_in_length"],
            stride,
            scaler=self.scaler,
        )

        # se vogliamo aggiungere anomalie
        if delta_range is not None and anomaly_ratio > 0:
            W = self.cfg["dataset"]["seq_in_length"]
            Xok_ref = np.stack(
                [base_data[i : i + W] for i in range(len(base_data) - W + 1)]
            )
            anom_ds = TimeSeriesDataset(
                base_data,
                self.cfg["dataset"]["seq_in_length"],
                stride,
                scaler=self.scaler,
                delta_range=delta_range,
                Xok_ref=Xok_ref,
            )
            n_anom = int(anomaly_ratio * len(clean_ds))
            idx = np.random.choice(len(anom_ds), size=n_anom, replace=False).tolist()
            anom_ds = torch.utils.data.Subset(anom_ds, idx)

            combined_ds = ConcatDataset([clean_ds, anom_ds])
            return combined_ds

        return clean_ds
