import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

from torch.utils.data import DataLoader, ConcatDataset, RandomSampler
from sklearn.preprocessing import StandardScaler

from robustness.dataset.dataset import TimeSeriesDataset
from robustness.dataset.data_types import Config


class DataModule(pl.LightningDataModule):
    def __init__(self, cfg: Config, mode: str = "train"):
        super().__init__()
        self.cfg = cfg
        self.mode = mode

    def setup(self, stage: str | None = None) -> None:
        data = pd.read_csv(self.cfg.dataset.csv_path).values.astype(np.float32)
        T, F = data.shape

        W = self.cfg.dataset.seq_in_length
        n_seq_chunk = self.cfg.dataset.n_seq_chunk

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

        n_test = int(n_chunks * self.cfg.dataset.test_chunk_ratio)

        test_chunks = chunks[-n_test:]
        trainval_chunks = chunks[:-n_test]

        n_val = int(len(trainval_chunks) * self.cfg.dataset.val_ratio)

        val_chunks = trainval_chunks[:n_val]
        train_chunks = trainval_chunks[n_val:]

        # shuffle solo il validation
        np.random.shuffle(val_chunks)

        # torniamo a una shape (W, F)
        train_data = np.concatenate(train_chunks, axis=0).reshape(-1, F)
        val_data = np.concatenate(val_chunks, axis=0).reshape(-1, F)
        test_data = np.concatenate(test_chunks, axis=0).reshape(-1, F)

        self.scaler = StandardScaler().fit(train_data)

        W = self.cfg.dataset.seq_in_length
        n_ref = min(self.cfg.dataset.n_wombats_ref, len(val_data) - W)
        idx = np.random.choice(len(val_data) - W, size=n_ref, replace=False)
        Xok_ref = np.stack([val_data[i : i + W] for i in idx])  # shape: [n_ref, W, F]

        if self.mode == "train":
            self.train_ds = TimeSeriesDataset(
                train_data,
                W,
                self.cfg.dataset.seq_stride_train,
                scaler=self.scaler,
            )

            val_clean = TimeSeriesDataset(
                val_data,
                W,
                self.cfg.dataset.seq_stride_val,
                scaler=self.scaler,
            )

            val_anom = TimeSeriesDataset(
                val_data,
                W,
                self.cfg.dataset.seq_stride_val,
                scaler=self.scaler,
                delta_range=(self.cfg.dataset.delta_min, self.cfg.dataset.delta_max),
                Xok_ref=Xok_ref,
            )

            # 3. prendi una percentuale CASUALE
            ratio = self.cfg.dataset.val_anomaly_ratio  # 0.3
            n_anom = int(ratio * len(val_clean))
            idx = np.random.choice(len(val_anom), size=n_anom, replace=False).tolist()
            val_anom = torch.utils.data.Subset(val_anom, idx)

            self.val_ds = ConcatDataset([val_clean, val_anom])

            # RandomSampler riproducibile
            self.val_sampler = RandomSampler(
                self.val_ds,
                replacement=False,
                generator=torch.Generator().manual_seed(42),
            )

        elif self.mode == "test":
            if self.cfg.metrics.anomalous_test:
                # generiamo anomalie su tutto il test set
                n_total = len(test_data) - W
                idx = np.arange(n_total)  # tutte le sequenze
                Xok_ref = None  # opzionale, se vuoi usarlo come riferimento
                test_anom_ds = TimeSeriesDataset(
                    test_data,
                    W,
                    self.cfg.dataset.seq_stride_test,
                    scaler=self.scaler,
                    delta_range=(self.cfg.dataset.delta_min, self.cfg.dataset.delta_max),
                    Xok_ref=Xok_ref,
                )
                self.test_ds = test_anom_ds
            else:
                self.test_ds = TimeSeriesDataset(
                    test_data,
                    W,
                    self.cfg.dataset.seq_stride_test,
                    scaler=self.scaler,
                )

    def train_dataloader(self) -> DataLoader:
        if self.mode != "train":
            raise RuntimeError("train_dataloader chiamato in mode != train")

        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.dataset.batch_size,
            shuffle=False,
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
            shuffle=False,
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
