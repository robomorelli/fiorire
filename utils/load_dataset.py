"""
Dataset loading and index generation utilities.
Minimal, focused on the refactored tensor-based system.
"""

import pandas as pd
import numpy as np
import os
from scipy import interpolate
from omegaconf import ListConfig
from xyzservices.providers import data_path

from utils.metric_dataset_generator import generate_and_save_metric_dataset
from preprocessing.scaling import *


# =========================================================
# CORE FUNCTIONS - USATE NEL NUOVO SISTEMA
# =========================================================

def load_and_preprocess_dataframe(cfg, data_path):
    """
    Load DataFrame and apply preprocessing:
    - Feature selection
    - Target handling
    - Column removal
    - NaN removal
    - Optional upsampling

    Args:
        cfg: Configuration
        data_path: Path to data file

    Returns:
        df: Preprocessed DataFrame
    """
    # 1. Load raw data
    ext = os.path.splitext(data_path)[1].lower()

    print(f"   📂 Loading from: {data_path}")

    if ext == '.pkl' or ext == '.pickle':
        df = pd.read_pickle(data_path)
    elif ext == '.csv':
        df = pd.read_csv(data_path)
    elif ext == '.parquet':
        df = pd.read_parquet(data_path)
    elif ext in ['.xlsx', '.xls']:
        df = pd.read_excel(data_path)
    else:
        df = pd.read_csv(data_path)

    print(f"   ✓ Loaded: {df.shape}")

    # 2. Normalize target to list
    cfg.dataset.target = (
        cfg.dataset.target
        if isinstance(cfg.dataset.target, (list, ListConfig))
        else [cfg.dataset.target] if cfg.dataset.target
        else None
    )

    # 3. Handle features
    # If feats is None, use all columns
    if cfg.dataset.feats is None or cfg.dataset.feats == [None]:
        cfg.dataset.feats = df.columns.tolist()

    # Remove unwanted columns
    remove_columns = cfg.dataset.get("remove_columns", [])
    if remove_columns:
        cfg.dataset.feats = [x for x in cfg.dataset.feats if x not in remove_columns]
        print(f"   🔧 Removed columns: {remove_columns}")

    # 4. Build column list
    # Include target columns not already in feats
    if cfg.dataset.target:
        columns = cfg.dataset.feats + [
            x for x in cfg.dataset.target if x not in cfg.dataset.feats
        ]
    else:
        columns = cfg.dataset.feats

    # If target is None, use all feature columns as target (autoencoder)
    if cfg.dataset.target is None:
        cfg.dataset.target = columns

    # Add anomaly column if present (for filtering later)
    ano_col = cfg.dataset.get('is_anomaly_column')
    if ano_col and ano_col in df.columns and ano_col not in columns:
        columns = columns + [ano_col]

    # 5. Select columns and drop NaN
    print(f"   📊 Shape before dropna: {df.shape}")
    df = df[columns].dropna()
    print(f"   ✓ Shape after dropna: {df.shape}")

    # 6. Subset dataset if requested
    dataset_subset = cfg.dataset.get('dataset_subset', None)
    if dataset_subset is not None:
        print(f"   🔧 Using subset: first {dataset_subset} samples")
        df = df.iloc[:dataset_subset, :]

    # 7. Optional upsampling
    if cfg.dataset.get('upsample_factor', 0) > 1:
        print(f"   🔧 Upsampling by factor {cfg.dataset.upsample_factor}")
        up_factor = cfg.dataset.get('upsample_factor')
        method = cfg.dataset.get('upsample_method', 'cubic')
        augmentation = cfg.dataset.get('upsample_augmentation', False)

        df = upsample_and_augment(
            df_low=df,
            factor=up_factor,
            method=method,
            do_augmentation=augmentation,
            ano_col=ano_col
        )

    print(f"   ✅ Final preprocessed shape: {df.shape}")
    print(f"   ✅ Features: {cfg.dataset.feats}")

    return df


def create_train_val_df_indexes(
        cfg,
        df,
        return_anomalies=False,
        ano_col="is_anomaly",
        seed=42,
        min_chunk_size=None,
        max_chunks=200,
        tolerance=0.01,

):
    """
    Split dataframe into train/val by chunk sampling with optional anomaly filtering.

    Returns:
        train_indexes: Training indices
        val_indexes: Validation indices
        df_train_values_for_scaling: DataFrame subset for scaler fitting
        anomalous_indexes: Anomalous indices (if return_anomalies=True)
    """

    def valid_chunk_splits(
            total_len,
            val_ratio=0.2,
            max_chunks=100,
            min_chunk_size=None,
            tolerance=0.01,
    ):
        """Find optimal chunking configuration."""
        best_solution = None

        for num_chunks in range(1, max_chunks + 1):
            chunk_size = total_len // num_chunks
            if min_chunk_size and chunk_size < min_chunk_size:
                continue

            val_chunks = int(np.ceil(num_chunks * val_ratio))
            actual_ratio = val_chunks / num_chunks

            if abs(actual_ratio - val_ratio) <= tolerance:
                candidate = {
                    "num_chunks": num_chunks,
                    "val_chunks": val_chunks,
                    "chunk_size": chunk_size,
                    "actual_ratio": actual_ratio,
                }

                if (best_solution is None or
                        candidate["num_chunks"] > best_solution["num_chunks"]):
                    best_solution = candidate

        if best_solution is None:
            raise ValueError(
                "No valid chunking found — increase max_chunks or reduce min_chunk_size/tolerance."
            )

        return best_solution

    # Configuration
    seq_len = cfg.dataset.seq_in_length
    val_ratio = 1 - cfg.dataset.train_val_split
    np.random.seed(seed)
    df = df.reset_index(drop=True)
    total_len = len(df)

    min_chunk_size = (cfg.dataset.seq_in_length * cfg.dataset.seq_in_length_into_chunk
                      if min_chunk_size is None else min_chunk_size)

    # Find chunking configuration
    solution = valid_chunk_splits(
        total_len,
        val_ratio=val_ratio,
        max_chunks=max_chunks,
        min_chunk_size=min_chunk_size,
        tolerance=tolerance,
    )

    num_chunks = solution["num_chunks"]
    val_chunk_num = solution["val_chunks"]
    chunk_size = solution["chunk_size"]

    print(f"      - Chunks: {num_chunks} total, {val_chunk_num} validation")
    print(f"      - Chunk size: {chunk_size} timesteps")

    # Select validation chunks
    chunks = np.arange(num_chunks)
    np.random.shuffle(chunks)
    val_chunk_idxs = chunks[:val_chunk_num]

    val_indexes = []
    for i in val_chunk_idxs:
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < num_chunks - 1 else total_len
        val_indexes.extend(range(start, end))

    val_indexes = np.array(val_indexes)
    all_indexes = np.arange(total_len)
    train_indexes = np.setdiff1d(all_indexes, val_indexes, assume_unique=True)

    # =========================================================
    # FALLBACK: Check if anomaly filtering is requested but column missing
    # =========================================================
    if return_anomalies:
        if not ano_col or ano_col not in df.columns:
            print("\n" + "!" * 80)
            print("⚠️  WARNING: Anomaly filtering requested but anomaly column not found!")
            print(f"    - Requested column: '{ano_col}'")
            print(f"    - Available columns: {list(df.columns)}")
            print("    - FALLBACK: Proceeding WITHOUT anomaly filtering")
            print("    - Config updated: filter_anomalies = False")
            print("!" * 80 + "\n")

            # Update config to reflect reality
            if hasattr(cfg, 'opt'):
                cfg.opt.filter_anomalies = False

            # Return as if return_anomalies=False
            df_train_values_for_scaling = df.iloc[train_indexes].reset_index(drop=True)
            return train_indexes, val_indexes, df_train_values_for_scaling, None

    # If anomalies not needed, return here
    if not return_anomalies:
        df_train_values_for_scaling = df.iloc[train_indexes].reset_index(drop=True)
        return train_indexes, val_indexes, df_train_values_for_scaling, None

    # =========================================================
    # ANOMALY FILTERING (only if column exists)
    # =========================================================
    full_anomalous_idx = df[df[ano_col] == 1].index.to_numpy()

    if len(full_anomalous_idx) == 0:
        print("\n" + "!" * 80)
        print(f"⚠️  WARNING: Anomaly column '{ano_col}' exists but contains NO anomalies!")
        print("    - All values are 0 or non-1")
        print("    - FALLBACK: Proceeding WITHOUT anomaly filtering")
        print("!" * 80 + "\n")

        df_train_values_for_scaling = df.iloc[train_indexes].reset_index(drop=True)
        return train_indexes, val_indexes, df_train_values_for_scaling, None

    def get_anomaly_window_indexes(anomalous_idx, total_len):
        window_indexes = set()
        for i in anomalous_idx:
            start = max(i - seq_len, 0)
            end = min(start + seq_len, total_len)
            start = max(end - seq_len, 0)
            window_indexes.update(range(start, end))
        return window_indexes

    train_anomalous_indexes = np.intersect1d(full_anomalous_idx, train_indexes)
    val_anomalous_indexes = np.intersect1d(full_anomalous_idx, val_indexes)

    anomalous_window_indexes = list(get_anomaly_window_indexes(full_anomalous_idx, total_len))
    train_anomalous_windows = get_anomaly_window_indexes(train_anomalous_indexes, total_len)
    val_anomalous_windows = get_anomaly_window_indexes(val_anomalous_indexes, total_len)


    train_normal_indexes = np.setdiff1d(train_indexes, list(train_anomalous_windows), assume_unique=True)
    val_normal_indexes = np.setdiff1d(val_indexes, list(val_anomalous_windows), assume_unique=True)

    # For scaler fitting: only features (exclude anomaly column)
    scaling_cols = df.columns.difference([ano_col]) if ano_col in df.columns else df.columns
    scaling_cols = [x for x in df.columns if x in scaling_cols]
    df_train_values_for_scaling = df[scaling_cols].iloc[train_normal_indexes].reset_index(drop=True)

    print(f"      - Train normal: {len(train_normal_indexes)}")
    print(f"      - Val normal: {len(val_normal_indexes)}")
    if anomalous_window_indexes:
        print(f"      - Anomalous windows: {len(anomalous_window_indexes)}")

    return train_normal_indexes, val_normal_indexes, df_train_values_for_scaling, anomalous_window_indexes


def extract_sequences_with_labels(
    df,
    indices,
    seq_len,
    feature_columns,
    anomaly_column
):
    """
    Extract sequences AND labels from DataFrame.

    Args:
        df: DataFrame (scaled)
        indices: Starting indices
        seq_len: Sequence length
        feature_columns: Feature column names
        anomaly_column: Anomaly label column

    Returns:
        sequences: [N, L, F] numpy array
        labels: [N, L, 1] numpy array (1=anomaly, 0=normal)
    """
    N = len(indices)
    F = len(feature_columns)

    sequences = np.zeros((N, seq_len, F), dtype=np.float32)
    labels = np.zeros((N, seq_len, 1), dtype=np.float32)

    valid_count = 0
    max_idx = len(df) - seq_len

    for i, start_idx in enumerate(indices):
        if start_idx > max_idx:
            continue

        end_idx = start_idx + seq_len

        # Extract sequence
        sequences[valid_count] = df.iloc[start_idx:end_idx][feature_columns].values

        # Extract labels
        if anomaly_column and anomaly_column in df.columns:
            labels[valid_count, :, 0] = df.iloc[start_idx:end_idx][anomaly_column].values
        else:
            labels[valid_count, :, 0] = np.nan

        valid_count += 1

    # Trim to valid count
    if valid_count < N:
        sequences = sequences[:valid_count]
        labels = labels[:valid_count]

    return sequences, labels


def prepare_shared_configuration(cfg):
    """
    Prepare shared configuration using Ray Object Store.
    Large objects (sequences, scaler) are stored in Ray's shared memory.

    Returns:
        shared_config: Dict with Ray ObjectRefs and small metadata
    """
    import ray
    from pathlib import Path

    print("\n" + "=" * 80)
    print("📦 PREPARING SHARED CONFIGURATION")
    print("=" * 80)

    # Extract config parameters
    seed = cfg.opt.get('seed', 42)
    filter_anomalies = cfg.opt.get('filter_anomalies', False)
    is_anomaly_column = cfg.dataset.get('is_anomaly_column', None)

    # 1. Load data
    print("\n1️⃣ Loading dataset...")
    df = load_and_preprocess_dataframe(cfg, data_path=cfg.dataset.data_path)
    feature_columns = cfg.dataset.feats
    print(f"   ✓ Loaded: {df.shape}")

    # 2. Split indices
    print("\n2️⃣ Splitting train/val indices...")
    train_indexes, val_indexes, train_df_for_scaling, anomalous_indexes = (
        create_train_val_df_indexes(
            cfg=cfg,
            df=df,
            return_anomalies=filter_anomalies,
            ano_col=is_anomaly_column,
            seed=seed
        )
    )

    print(f"   ✓ Train indices: {len(train_indexes)}")
    print(f"   ✓ Val indices: {len(val_indexes)}")
    if anomalous_indexes is not None:
        print(f"   ✓ Anomalous indices: {len(anomalous_indexes)}")

    # 3. Fit scaler on CLEAN train data
    print("\n3️⃣ Fitting scaler on CLEAN train data...")
    scaler, df_scaled, scaler_params = get_scaler(
        cfg=cfg,
        df_fit=train_df_for_scaling,  # Already filtered
        df_transform=df
    )
    print(f"   ✓ Scaler fitted: {scaler.__class__.__name__}")

    # 4. Extract sequences (no overlap - base sequences)
    print("\n4️⃣ Extracting base sequences (no overlap)...")
    seq_len = cfg.dataset.seq_in_length

    train_sequences = extract_sequences_from_indices(
        df=df_scaled,
        indices=train_indexes,
        seq_len=seq_len,
        feature_columns=feature_columns,
        perc_overlap=0  # No overlap - base sequences
    )

    val_sequences = extract_sequences_from_indices(
        df=df_scaled,
        indices=val_indexes,
        seq_len=seq_len,
        feature_columns=feature_columns,
        perc_overlap=0  # No overlap - base sequences
    )

    print(f"   ✓ Train sequences: {train_sequences.shape}")
    print(f"   ✓ Val sequences: {val_sequences.shape}")
    print(f"   ✓ Memory: ~{(train_sequences.nbytes + val_sequences.nbytes) / 1024 ** 3:.2f} GB")

    # 5. Generate metric dataset
    print("\n5️⃣ Generating metric dataset...")
    metric_dataset_path = None
    if cfg.opt.get('anomaly_strategy', 'none') != 'none':
        metric_dataset_path = generate_and_save_metric_dataset(
            cfg=cfg,
            df_scaled=df_scaled,
            val_indices=val_indexes,
            feature_columns=feature_columns,
            scaler=scaler,
            output_dir='./metric_datasets/',
            force_regenerate=False,
            plot_samples=cfg.opt.get('plot_metric_samples', False),
            plot_percentage=cfg.opt.get('plot_metric_percentage', 0.05)
        )
        print(f"   ✓ Metric dataset: {metric_dataset_path}")

    # ✅ 6. Put large objects in Ray Object Store
    print("\n6️⃣ Storing objects in Ray Object Store...")

    train_sequences_ref = ray.put(train_sequences)
    val_sequences_ref = ray.put(val_sequences)
    scaler_ref = ray.put(scaler)

    print(f"   ✓ train_sequences → Ray ObjectRef")
    print(f"   ✓ val_sequences → Ray ObjectRef")
    print(f"   ✓ scaler → Ray ObjectRef")

    # Clean up local copies to free memory
    del train_sequences
    del val_sequences
    del df
    del df_scaled
    import gc
    gc.collect()
    print(f"   ✓ Local memory freed")

    # ✅ 7. Build lightweight shared config
    shared_config = {
        # Ray ObjectRefs (tiny - just pointers!)
        'train_sequences': train_sequences_ref,
        'val_sequences': val_sequences_ref,
        'scaler': scaler_ref,

        # Small metadata
        'scaler_params': scaler_params,
        'feature_columns': list(feature_columns),
        'seq_len': int(seq_len),
        'metric_loader_path': metric_dataset_path,

        # Statistics
        'train_size': int(len(train_indexes)),
        'val_size': int(len(val_indexes)),
    }

    print("\n" + "=" * 80)
    print("✅ SHARED CONFIGURATION READY")
    print("=" * 80)
    print(f"   - Train indices: {shared_config['train_size']}")
    print(f"   - Val indices: {shared_config['val_size']}")
    print(f"   - Scaler: {scaler.__class__.__name__}")
    if metric_dataset_path:
        print(f"   - Metric dataset: {metric_dataset_path}")
    print(f"   - Using Ray Object Store: YES ✓")
    print("=" * 80)

    return shared_config


# =========================================================
# UTILITY FUNCTIONS
# =========================================================

def upsample_and_augment(
    df_low: pd.DataFrame,
    factor: int = 4,
    method: str = "cubic",
    do_augmentation: bool = True,
    jitter_std: float = 0.001,
    scale_range=(0.995, 1.005),
    ano_col=None,
    propagate_mode: str = "forward",
) -> pd.DataFrame:
    """
    Upsample DataFrame with optional augmentation.
    """
    if factor <= 1:
        return df_low.copy()

    df_low = df_low.sort_index()
    n = len(df_low)

    if ano_col is None:
        ano_cols = []
    elif isinstance(ano_col, str):
        ano_cols = [ano_col]
    else:
        ano_cols = list(ano_col)

    if isinstance(df_low.index, pd.DatetimeIndex):
        idx = pd.DatetimeIndex(df_low.index)
        if idx.tz is not None:
            idx = idx.tz_convert(None)
        x_low = idx.view(np.int64) / 1e9
        idx_type = "datetime"
    else:
        x_low = df_low.index.to_numpy(dtype=float)
        idx_type = "numeric"

    x_min, x_max = x_low[0], x_low[-1]
    n_new = (n - 1) * factor + 1
    x_high = np.linspace(x_min, x_max, n_new)

    upsampled_cols = {}

    for col in df_low.columns:
        y_low = df_low[col].to_numpy()

        if col in ano_cols:
            if propagate_mode == "backward":
                y_high = np.repeat(y_low, factor)
                y_high = np.append(y_low[0], y_high[:-1])
            else:
                y_high = np.repeat(y_low, factor)
                y_high = np.append(y_high, y_low[-1])
            y_high = y_high[:n_new]
            upsampled_cols[col] = y_high.astype(int)
            continue

        col_method = "linear" if (method == "cubic" and len(df_low) < 4) else method
        f = interpolate.interp1d(
            x_low, y_low, kind=col_method, bounds_error=False, fill_value="extrapolate"
        )
        y_high = f(x_high)

        if do_augmentation:
            std = np.nanstd(y_low)
            if std == 0 or np.isnan(std):
                std = 1.0
            noise = np.random.normal(0, jitter_std * std, len(y_high))
            scale = np.random.uniform(scale_range[0], scale_range[1])
            y_high = y_high * scale + noise

        upsampled_cols[col] = y_high

    if idx_type == "datetime":
        new_index = pd.to_datetime(x_high, unit="s")
    else:
        new_index = x_high

    df_high = pd.DataFrame(upsampled_cols, index=new_index)
    df_high.index.name = df_low.index.name
    return df_high


# =========================================================
# TENSOR-BASED LOADING FUNCTIONS (per i trial)
# =========================================================

def load_sequences_for_trial(cfg, shared_config, overlap):
    """
    Load train/val sequences for a single trial with specified overlap.

    Uses shared base indices and applies trial-specific overlap.

    Args:
        cfg: Trial configuration
        shared_config: Shared configuration from prepare_shared_configuration
        overlap: Overlap percentage for this trial

    Returns:
        train_sequences: [N_train, L, F] numpy array
        val_sequences: [N_val, L, F] numpy array
    """
    from preprocessing.scaling import apply_scaler

    # Extract shared data
    train_base_indices = np.array(shared_config['train_indices'])
    val_base_indices = np.array(shared_config['val_indices'])
    scaler = shared_config['scaler']
    dataset_path = shared_config['dataset_path']
    feature_columns = shared_config['feature_columns']
    seq_len = shared_config['seq_len']

    # Load DataFrame
    df = load_and_preprocess_dataframe(cfg, dataset_path)

    # Scale
    df_scaled = apply_scaler(df, scaler, feature_columns)

    # Apply overlap to base indices
    train_indices_with_overlap = apply_overlap_to_indices(
        train_base_indices, seq_len, overlap
    )
    val_indices_with_overlap = apply_overlap_to_indices(
        val_base_indices, seq_len, overlap
    )

    print(f"   📊 Trial overlap: {overlap}")
    print(f"      - Train: {len(train_base_indices)} → {len(train_indices_with_overlap)} sequences")
    print(f"      - Val: {len(val_base_indices)} → {len(val_indices_with_overlap)} sequences")

    # Extract sequences
    train_sequences = extract_sequences_from_indices(
        df_scaled, train_indices_with_overlap, seq_len, feature_columns
    )
    val_sequences = extract_sequences_from_indices(
        df_scaled, val_indices_with_overlap, seq_len, feature_columns
    )

    return train_sequences, val_sequences


def apply_overlap_to_indices(base_indices, seq_len, overlap):
    """
    Apply overlap to base indices, respecting gaps.

    Gap-aware: only creates intermediate sequences between consecutive indices.

    Args:
        base_indices: Base indices (overlap=0)
        seq_len: Sequence length
        overlap: Overlap percentage (0.0 to 1.0)

    Returns:
        new_indices: Indices with overlap applied
    """
    if overlap == 0.0:
        return base_indices

    step = seq_len - int(seq_len * overlap)

    new_indices = [base_indices[0]]

    for i in range(len(base_indices) - 1):
        current = base_indices[i]
        next_idx = base_indices[i + 1]

        # Gap-aware: only if consecutive (next - current == seq_len)
        if next_idx - current == seq_len:
            # Create intermediate sequences
            intermediates = range(current + step, next_idx, step)
            new_indices.extend(intermediates)

        new_indices.append(next_idx)

    return np.array(new_indices)


def extract_sequences_from_indices(df, indices, seq_len, feature_columns, perc_overlap=0):
    """
    Extract sequences from DataFrame at given indices (OPTIMIZED).
    """
    # Convert to NumPy ONCE
    data = df[feature_columns].values  # [T, F]

    max_idx = len(data) - seq_len
    valid_indices = indices[indices <= max_idx]
    step = seq_len - int(seq_len * perc_overlap)
    step = max(1, step)
    valid_indices = valid_indices[::step]

    # Vectorized extraction usando broadcasting
    idx_array = valid_indices[:, None] + np.arange(seq_len)[None, :]  # [N, seq_len]
    sequences = data[idx_array]  # [N, seq_len, F] - allocazione diretta

    return sequences.astype(np.float32)  # Cast finale se necessario


def get_transform(cfg):
    """
    Get data transform (augmentation) if any.

    Args:
        cfg: Configuration

    Returns:
        transform function or None
    """
    # Currently no transforms, but you can add augmentation here
    # Example:
    # if cfg.dataset.get('augmentation', False):
    #     return some_transform_function
    return None


def create_dataloaders(train_dataset, val_dataset, cfg):
    """
    Create train and validation dataloaders from datasets.

    Args:
        train_dataset: Training Dataset_seq
        val_dataset: Validation Dataset_seq
        cfg: Configuration

    Returns:
        trainloader: Training DataLoader
        valloader: Validation DataLoader
    """
    from torch.utils.data import DataLoader

    trainloader = DataLoader(
        train_dataset,
        batch_size=cfg.opt.batch_size,
        shuffle=cfg.dataset.get('shuffle_train', True),
        num_workers=cfg.opt.get('num_workers', 0),
        pin_memory=True if cfg.resources.gpu_trial else False
    )

    valloader = DataLoader(
        val_dataset,
        batch_size=cfg.opt.batch_size,
        shuffle=False,
        num_workers=cfg.opt.get('num_workers', 0),
        pin_memory=True if cfg.resources.gpu_trial else False
    )

    return trainloader, valloader


def load_metric_loader_with_metadata(filepath, verbose=True):
    """
    Load metric loader and extract metadata.

    Args:
        filepath: Path to saved metric loader
        verbose: Print info

    Returns:
        metric_loader: DataLoader
        metadata: Dictionary with metadata
    """
    import torch

    if verbose:
        print(f"\n📂 Loading metric loader: {filepath}")

    # Load saved dict
    saved_dict = torch.load(filepath, map_location='cpu')

    # Check if it's the new format (with metadata)
    if isinstance(saved_dict, dict) and 'metadata' in saved_dict:
        metric_loader = saved_dict['loader']
        metadata = saved_dict['metadata']

        if verbose:
            print(f"   ✓ Loaded: {metadata.get('num_sequences', 'N/A')} sequences")
            print(f"   ✓ Strategy: {metadata.get('strategy', 'N/A')}")
            print(f"   ✓ Data scale: {'STANDARDIZED' if metadata.get('is_standardized') else 'ORIGINAL'}")

        return metric_loader, metadata

    else:
        # Old format (backward compatibility)
        if verbose:
            print(f"   ⚠️  WARNING: Old format detected (no metadata)")
            print(f"   ℹ️  Assuming data is STANDARDIZED (legacy behavior)")

        metric_loader = saved_dict
        metadata = {
            'is_standardized': True,  # Assume standardized for old files
            'legacy': True
        }

        return metric_loader, metadata


def apply_scaler_to_batch(batch, scaler, device):
    """
    Apply sklearn scaler to a PyTorch batch.

    Args:
        batch: Dictionary with 'input' tensor [B, L, F]
        scaler: Fitted sklearn scaler
        device: torch device

    Returns:
        batch: Dictionary with scaled 'input' tensor
    """
    import torch
    import numpy as np

    x = batch['input']  # [B, L, F]
    B, L, F = x.shape

    # Convert to numpy
    x_np = x.cpu().numpy()  # [B, L, F]

    # Reshape to [B*L, F] for scaler
    x_flat = x_np.reshape(-1, F)  # [B*L, F]

    # Apply scaler
    x_scaled = scaler.transform(x_flat)  # [B*L, F]

    # Reshape back to [B, L, F]
    x_scaled = x_scaled.reshape(B, L, F)

    # Convert back to tensor
    x_tensor = torch.from_numpy(x_scaled).float().to(device)

    # Update batch
    batch['input'] = x_tensor

    return batch


def load_metric_loader_with_metadata(filepath, verbose=True):
    """
    Load metric loader and extract metadata.

    Args:
        filepath: Path to saved metric loader
        verbose: Print info

    Returns:
        metric_loader: DataLoader
        metadata: Dictionary with metadata (always contains 'is_standardized')
    """
    import torch

    if verbose:
        print(f"\n📂 Loading metric loader: {filepath}")

    # Load saved dict
    saved_dict = torch.load(filepath, map_location='cpu')

    # Check if it's the new format (with metadata)
    if isinstance(saved_dict, dict) and 'metadata' in saved_dict:
        metric_loader = saved_dict['loader']
        metadata = saved_dict['metadata']

        if verbose:
            print(f"   ✓ Loaded: {metadata.get('num_sequences', 'N/A')} sequences")
            print(f"   ✓ Strategy: {metadata.get('strategy', 'N/A')}")

            is_standardized = metadata.get('is_standardized', True)
            if is_standardized:
                print(f"   ✓ Data scale: STANDARDIZED ✓")
            else:
                print(f"   ⚠️  Data scale: ORIGINAL (NOT standardized!) ⚠️")

        return metric_loader, metadata

    else:
        # Old format (backward compatibility)
        if verbose:
            print(f"   ⚠️  WARNING: Old format detected (no metadata)")
            print(f"   ℹ️  Assuming data is STANDARDIZED (legacy behavior)")

        metric_loader = saved_dict
        metadata = {
            'is_standardized': True,  # Assume standardized for old files
            'legacy': True
        }

        return metric_loader, metadata