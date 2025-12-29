"""
Optimized Dataset for time series sequences.
Supports both new (tensor-based) and legacy (DataFrame-based) initialization.
Internally ALWAYS uses tensors for speed.
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from typing import Optional, Union, List
from omegaconf import ListConfig


class Dataset_seq(Dataset):
    """
    Universal time series dataset - accepts both tensors AND DataFrames.

    **Mode 1 - New (Fast): Pre-extracted tensors**
        >>> sequences = np.random.randn(1000, 16, 8)
        >>> dataset = Dataset_seq(sequences=sequences)

    **Mode 2 - Legacy (Compatible): DataFrame + indices**
        >>> dataset = Dataset_seq(df=df, indices=train_idx, sequence_length=16)
        >>> # Extracts sequences ONCE on init, then uses tensors internally

    Both modes result in same fast tensor-based __getitem__!
    """

    def __init__(
        self,
        # New mode parameters
        sequences: Optional[Union[np.ndarray, torch.Tensor]] = None,
        targets: Optional[Union[np.ndarray, torch.Tensor]] = None,
        anomaly_labels: Optional[Union[np.ndarray, torch.Tensor]] = None,
        transform=None,
        # Legacy mode parameters
        df: Optional[pd.DataFrame] = None,
        indices: Optional[np.ndarray] = None,
        sequence_length: Optional[int] = None,
        target: Optional[Union[str, List[str]]] = None,
        out_window: Optional[int] = None,
        forecast: bool = False,
        remove_target: bool = False,
        is_anomaly_column: Optional[str] = None,
        sampler=None,
        **kwargs
    ):
        """
        Initialize dataset - auto-detects mode based on parameters.

        NEW MODE (sequences provided):
            Directly uses pre-extracted sequences (fastest).

        LEGACY MODE (df + indices provided):
            Extracts sequences from DataFrame ONCE during init,
            then stores as tensors internally (fast __getitem__).
        """
        self.transform = transform
        self.forecast = forecast
        self.remove_target = remove_target

        # MODE DETECTION
        if sequences is not None:
            # NEW MODE
            self._init_from_sequences(sequences, targets, anomaly_labels)

        elif df is not None and indices is not None and sequence_length is not None:
            # LEGACY MODE
            self._init_from_dataframe(
                df, indices, sequence_length, target, out_window,
                forecast, remove_target, is_anomaly_column, sampler
            )

        else:
            raise ValueError(
                "Must provide either:\n"
                "  - sequences [N, L, F] (new mode), or\n"
                "  - df + indices + sequence_length (legacy mode)"
            )

    def _init_from_sequences(self, sequences, targets, anomaly_labels):
        """Initialize from pre-extracted sequences (NEW MODE - FASTEST)."""
        # Convert to numpy
        if isinstance(sequences, torch.Tensor):
            sequences = sequences.numpy()

        self.sequences = sequences.astype(np.float32)

        # Targets default to sequences (reconstruction)
        if targets is None:
            self.targets = self.sequences.copy()
        else:
            if isinstance(targets, torch.Tensor):
                targets = targets.numpy()
            self.targets = targets.astype(np.float32)

        # Anomaly labels
        if anomaly_labels is not None:
            if isinstance(anomaly_labels, torch.Tensor):
                anomaly_labels = anomaly_labels.numpy()
            # Ensure shape [N, L, 1]
            if anomaly_labels.ndim == 2:
                anomaly_labels = anomaly_labels[:, :, np.newaxis]
            self.anomaly_labels = anomaly_labels.astype(np.float32)
        else:
            # NaN labels
            N, L = self.sequences.shape[0], self.sequences.shape[1]
            self.anomaly_labels = np.full((N, L, 1), np.nan, dtype=np.float32)

        # Metadata
        self.N = len(self.sequences)
        self.sequence_length = self.sequences.shape[1]
        self.n_features = self.sequences.shape[2]
        self.out_window = self.targets.shape[1]
        self.n_target_features = self.targets.shape[2]

        # For compatibility
        self.df = None
        self.indices = np.arange(self.N)
        self.feature_columns = [f'feature_{i}' for i in range(self.n_features)]
        self.target_columns = [f'feature_{i}' for i in range(self.n_target_features)]

    def _init_from_dataframe(
        self,
        df,
        indices,
        sequence_length,
        target,
        out_window,
        forecast,
        remove_target,
        is_anomaly_column,
        sampler
    ):
        """
        Initialize from DataFrame (LEGACY MODE - COMPATIBLE).
        Extracts sequences ONCE and stores as tensors internally.
        """
        self.df = df
        self.sequence_length = sequence_length
        self.out_window = out_window if out_window is not None else sequence_length

        # Handle indices
        if indices is not None:
            self.indices = indices
        elif sampler is not None and hasattr(sampler, 'indices'):
            self.indices = sampler.indices
        else:
            max_idx = len(df) - sequence_length
            self.indices = np.arange(0, max_idx, sequence_length)

        # Determine feature columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['is_anomaly', 'anomaly_type', 'affected_channels']

        # Handle is_anomaly_column
        is_anomaly_column_list = None
        if is_anomaly_column:
            if isinstance(is_anomaly_column, (list, ListConfig)):
                is_anomaly_column_list = list(is_anomaly_column)
            elif isinstance(is_anomaly_column, str):
                is_anomaly_column_list = [is_anomaly_column]

            if is_anomaly_column_list:
                exclude_cols.extend(is_anomaly_column_list)

        feature_cols = [c for c in numeric_cols if c not in exclude_cols]

        # Handle remove_target
        if remove_target and target:
            target_list = target if isinstance(target, (list, ListConfig)) else [target]
            feature_cols = [c for c in feature_cols if c not in target_list]

        self.feature_columns = feature_cols
        self.n_features = len(feature_cols)

        # Determine target columns
        if target is None:
            self.target_columns = feature_cols
        elif isinstance(target, str):
            self.target_columns = [target] if target in df.columns else feature_cols
        elif isinstance(target, (list, ListConfig)):
            self.target_columns = [c for c in target if c in df.columns]
            if not self.target_columns:
                self.target_columns = feature_cols
        else:
            self.target_columns = feature_cols

        self.n_target_features = len(self.target_columns)

        # EXTRACT ALL SEQUENCES AS TENSORS (ONE-TIME COST)
        print(f"   📦 Extracting {len(self.indices)} sequences from DataFrame...")

        N = len(self.indices)
        L = sequence_length
        F = self.n_features
        F_target = self.n_target_features

        # Pre-allocate
        self.sequences = np.zeros((N, L, F), dtype=np.float32)
        self.targets = np.zeros((N, self.out_window, F_target), dtype=np.float32)

        # Extract labels if available
        has_anomaly_labels = (
            is_anomaly_column_list and
            all(col in df.columns for col in is_anomaly_column_list)
        )

        if has_anomaly_labels:
            self.anomaly_labels = np.zeros((N, L, 1), dtype=np.float32)
        else:
            self.anomaly_labels = np.full((N, L, 1), np.nan, dtype=np.float32)

        # Extract
        max_idx = len(df) - sequence_length
        valid_count = 0

        for i, start_idx in enumerate(self.indices):
            if start_idx > max_idx:
                continue

            end_idx = start_idx + sequence_length

            # Extract input
            self.sequences[valid_count] = df.iloc[start_idx:end_idx][self.feature_columns].values

            # Extract target
            if forecast:
                target_start = end_idx
                target_end = target_start + self.out_window
                if target_end <= len(df):
                    self.targets[valid_count] = df.iloc[target_start:target_end][self.target_columns].values
                else:
                    continue
            else:
                target_seq = df.iloc[start_idx:end_idx][self.target_columns].values
                self.targets[valid_count, :len(target_seq), :] = target_seq

            # Extract labels
            if has_anomaly_labels:
                anomaly_col = is_anomaly_column_list[0]
                self.anomaly_labels[valid_count, :, 0] = df.iloc[start_idx:end_idx][anomaly_col].values

            valid_count += 1

        # Trim
        if valid_count < N:
            self.sequences = self.sequences[:valid_count]
            self.targets = self.targets[:valid_count]
            self.anomaly_labels = self.anomaly_labels[:valid_count]
            self.indices = self.indices[:valid_count]

        self.N = valid_count

        print(f"   ✓ Extracted {self.N} sequences ({self.get_memory_size_mb():.2f} MB)")
        print(f"      - Sequences: {self.sequences.shape}")
        print(f"      - Targets: {self.targets.shape}")

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        """
        Get single sequence (FAST - always uses pre-extracted tensors).

        Returns:
            data: [L, F] input sequence tensor
            target: [L_out, F_out] target sequence tensor
            anomaly_labels: [L, 1] binary labels tensor
        """
        data = self.sequences[idx]
        target = self.targets[idx]
        labels = self.anomaly_labels[idx]

        if self.transform is not None:
            data = self.transform(data)
            target = self.transform(target)
            labels = self.transform(labels)
        else:
            data = torch.FloatTensor(data)
            target = torch.FloatTensor(target)
            labels = torch.FloatTensor(labels)

        return data, target, labels

    def get_memory_size(self):
        """Estimate memory usage in bytes."""
        size = self.sequences.nbytes
        size += self.targets.nbytes
        size += self.anomaly_labels.nbytes
        return size

    def get_memory_size_mb(self):
        """Get memory size in megabytes."""
        return self.get_memory_size() / (1024 ** 2)

    def __repr__(self):
        has_labels = not np.isnan(self.anomaly_labels).all()
        mode = "tensor" if self.df is None else "dataframe"
        return (f"Dataset_seq(mode={mode}, N={self.N}, L={self.sequence_length}, "
                f"F={self.n_features}, has_labels={has_labels}, "
                f"memory={self.get_memory_size_mb():.2f}MB)")


def concatenate_datasets(*datasets):
    """
    Concatenate multiple Dataset_seq objects.

    Args:
        *datasets: Variable number of Dataset_seq instances

    Returns:
        New Dataset_seq with concatenated sequences
    """
    all_sequences = np.concatenate([d.sequences for d in datasets], axis=0)
    all_targets = np.concatenate([d.targets for d in datasets], axis=0)

    all_labels = []
    for d in datasets:
        if not np.isnan(d.anomaly_labels).all():
            all_labels.append(d.anomaly_labels)
        else:
            N, L = d.sequences.shape[0], d.sequences.shape[1]
            all_labels.append(np.full((N, L, 1), np.nan, dtype=np.float32))

    all_labels = np.concatenate(all_labels, axis=0)

    transform = datasets[0].transform if datasets else None

    return Dataset_seq(
        sequences=all_sequences,
        targets=all_targets,
        anomaly_labels=all_labels,
        transform=transform
    )