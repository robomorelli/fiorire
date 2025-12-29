"""
Generate and save metric loaders based on anomaly strategy.
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from typing import Optional

from dataset.sentinel import Dataset_seq, concatenate_datasets
from utils.load_dataset import extract_sequences_with_labels


def generate_and_save_metric_loader(
        cfg: DictConfig,
        df_scaled,
        val_indices,
        feature_columns,
        scaler,
        output_dir='./metric_loaders/',
        force_regenerate=False
):
    """
    Generate metric loader based on anomaly strategy and save to disk.

    DEFAULT: Data is STANDARDIZED (is_standardized=True)
    OPTIONAL: Set force_destandardization=True to save in ORIGINAL SCALE
    """
    strategy = cfg.opt.get('anomaly_strategy', 'none')

    if strategy == 'none':
        print("\n   ℹ️  Strategy='none': No metric loader")
        return None

    # Create filename base
    dataset_name = cfg.dataset.get('name', 'dataset')
    exp_name = cfg.opt.get('exp_name', 'experiment')
    seed = cfg.opt.get('seed', 42)

    # Generate based on strategy
    seq_len = cfg.dataset.seq_in_length
    anomaly_column = cfg.dataset.get('is_anomaly_column')

    if strategy == 'use_original':
        metric_dataset, is_standardized = _create_original_anomaly_dataset(
            df_scaled, seq_len, feature_columns, anomaly_column
        )

    elif strategy == 'corrupt_validation':
        metric_dataset, is_standardized = _create_corrupted_validation_dataset(
            cfg, df_scaled, val_indices, seq_len, feature_columns, scaler
        )

    elif strategy == 'both':
        original_dataset, _ = _create_original_anomaly_dataset(
            df_scaled, seq_len, feature_columns, anomaly_column
        )
        corrupted_dataset, is_standardized = _create_corrupted_validation_dataset(
            cfg, df_scaled, val_indices, seq_len, feature_columns, scaler
        )

        if original_dataset is not None and corrupted_dataset is not None:
            metric_dataset = concatenate_datasets(original_dataset, corrupted_dataset)
            print(f"   ✓ Combined: {len(metric_dataset)} sequences")
        elif original_dataset is not None:
            metric_dataset = original_dataset
            is_standardized = True
        elif corrupted_dataset is not None:
            metric_dataset = corrupted_dataset
        else:
            print(f"   ⚠️  No datasets to combine!")
            return None

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    if metric_dataset is None:
        print(f"   ⚠️  No metric dataset created")
        return None

    # ✅ Add suffix based on scale
    if is_standardized:
        scale_suffix = ""  # Default - no suffix for standardized
        scale_info = "STANDARDIZED (ready for model)"
    else:
        scale_suffix = "_original"  # Mark non-standardized data
        scale_info = "ORIGINAL SCALE (needs standardization)"

    filename = f"metric_{exp_name}_{dataset_name}_{strategy}_seed{seed}{scale_suffix}.pkl"
    filepath = os.path.join(output_dir, filename)

    # Check if already exists
    if os.path.exists(filepath) and not force_regenerate:
        print(f"\n   ✓ Metric loader already exists: {filepath}")
        print(f"     (Use force_regenerate=True to regenerate)")
        return filepath

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n📊 Generating metric loader:")
    print(f"   - Strategy: {strategy}")
    print(f"   - Output: {filepath}")
    print(f"   - Data scale: {scale_info}")

    if not is_standardized:
        print(f"\n   ⚠️  WARNING: Data saved in ORIGINAL SCALE")
        print(f"      This metric loader will require standardization before use in training")

    # Create metadata dictionary
    metadata = {
        'is_standardized': is_standardized,
        'strategy': strategy,
        'dataset_name': dataset_name,
        'exp_name': exp_name,
        'seed': seed,
        'seq_len': seq_len,
        'feature_columns': list(feature_columns),
        'num_sequences': len(metric_dataset),
        'anomaly_column': anomaly_column,
    }

    # Add corruption config if applicable
    if strategy in ['corrupt_validation', 'both']:
        corruption_cfg = cfg.opt.get('corruption_config', {})
        metadata['corruption_config'] = {
            'anomalies_type': list(corruption_cfg.get('anomalies_type', [])),
            'delta_mean': float(corruption_cfg.get('delta_mean', 0.8)),
            'corruption_ratio': float(corruption_cfg.get('corruption_ratio', 1.0)),
        }

    # Create DataLoader
    metric_loader = DataLoader(
        metric_dataset,
        batch_size=cfg.opt.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    # Save loader + metadata together
    save_dict = {
        'loader': metric_loader,
        'metadata': metadata,
    }

    torch.save(save_dict, filepath)

    print(f"\n   ✅ Metric loader saved:")
    print(f"      - Path: {filepath}")
    print(f"      - Sequences: {len(metric_dataset)}")
    print(f"      - File size: {os.path.getsize(filepath) / 1024 ** 2:.2f} MB")
    print(f"      - is_standardized: {is_standardized}")

    return filepath


def _create_original_anomaly_dataset(df_scaled, seq_len, feature_columns, anomaly_column):
    """
    Extract original anomalies from dataset.

    Returns:
        dataset: Dataset object
        is_standardized: True (data is in scaled form)
    """

    if not anomaly_column or anomaly_column not in df_scaled.columns:
        print(f"   ⚠️  No anomaly column '{anomaly_column}' found")
        return None, False

    # Find anomalous sequences
    anomalous_indices = []
    for i in range(len(df_scaled) - seq_len):
        end_idx = i + seq_len
        if df_scaled.iloc[i:end_idx][anomaly_column].sum() > 0:
            anomalous_indices.append(i)

    if not anomalous_indices:
        print(f"   ⚠️  No original anomalies found")
        return None, False

    anomalous_indices = np.array(anomalous_indices)

    print(f"   📌 Extracting original anomalies:")
    print(f"      - Found: {len(anomalous_indices)} sequences")

    # Extract sequences with labels
    sequences, labels = extract_sequences_with_labels(
        df_scaled,
        anomalous_indices,
        seq_len,
        feature_columns,
        anomaly_column
    )

    dataset = Dataset_seq(
        sequences=sequences,
        anomaly_labels=labels
    )

    print(f"      ✓ Dataset created: {len(dataset)} sequences")
    print(f"      ℹ️  Data is in STANDARDIZED scale (from df_scaled)")

    return dataset, True  # True = data is standardized


def _create_corrupted_validation_dataset(
        cfg,
        df_scaled,
        val_indices,
        seq_len,
        feature_columns,
        scaler
):
    """
    Corrupt validation sequences using WOMBAT pipeline.
    Uses scaler already fitted on training data (no data leakage).

    Returns:
        dataset: Dataset object
        is_standardized: True/False based on force_destandardization
    """
    from utils.anomaly_corruption import corrupt_sequences_wombat
    from utils.load_dataset import extract_sequences_from_indices

    print(f"   🔧 Creating WOMBAT-corrupted validation dataset...")

    # Extract clean validation sequences (ALREADY SCALED by training scaler)
    clean_sequences = extract_sequences_from_indices(
        df_scaled,
        val_indices,
        seq_len,
        feature_columns,
        perc_overlap=cfg.opt.get("metric_seq_overlap", 0)
    )  # [N, L, F] - ALREADY SCALED

    print(f"      - Clean sequences: {clean_sequences.shape}")
    print(f"      - Data already scaled (using training scaler)")

    # Get corruption config
    random_seed = cfg.opt.get('corruption_config', {}).get('random_seed', 123)
    force_destd = cfg.opt.get('force_destandardization', False)

    # ✅ WOMBAT corruption with mode="skip" (sequences already scaled)
    corrupted_sequences, labels, anomaly_types, affected_channels, is_standardized = corrupt_sequences_wombat(
        sequences=clean_sequences,
        feature_columns=feature_columns,
        cfg=cfg,
        standardization_mode="skip",  # ← EXPLICITLY SKIP (already scaled)
        scaler=scaler,  # ← Provided for de-std if forced
        force_destandardization=force_destd,
        random_seed=random_seed,
        verbose=True
    )

    # Create dataset
    dataset = Dataset_seq(
        sequences=corrupted_sequences,
        anomaly_labels=labels
    )

    # Store metadata
    dataset.anomaly_types = anomaly_types
    dataset.affected_channels = affected_channels

    print(f"\n      ✅ Dataset created: {len(dataset)} sequences")
    print(f"         - Anomaly ratio: {labels.mean():.2%}")
    print(f"         - Metadata: anomaly_types, affected_channels")
    if is_standardized:
        print(f"         - ⚠️  Data is in STANDARDIZED scale")
    else:
        print(f"         - ✓ Data is in ORIGINAL scale")

    # Summary
    print(f"\n      📋 Anomaly distribution:")
    unique_types, counts = np.unique(anomaly_types[anomaly_types != 'normal'], return_counts=True)
    for atype, count in zip(unique_types, counts):
        print(f"         * {atype}: {count}")

    return dataset, is_standardized