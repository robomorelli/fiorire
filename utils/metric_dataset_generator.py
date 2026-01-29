"""
Generate and save metric loaders based on anomaly strategy.
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from typing import Optional, List, Tuple
import sys
import shutil
sys.path.append("./")

from dataset.sentinel import Dataset_seq, concatenate_datasets
from wombats.anomalies.increasing import *
from wombats.anomalies.invariant import *
from wombats.anomalies.decreasing import *

ANOMALIES_REGISTRY = {
    'GWN':GWN,
    'Constant':Constant,
    'Step':Step,
    'Impulse':Impulse,
    'GNN':GNN,
    'PrincipalSubspaceAlteration': PrincipalSubspaceAlteration,
    }


def generate_and_save_metric_dataset(
        cfg,
        val_sequences_raw,
        feature_columns,
        scaler_raw,
        scaler_smoothed,
        anomaly_strategy,
        original_anomaly_sequences=None,
        original_anomaly_labels=None,
        output_dir='./metric_datasets/',
        force_regenerate=False,
        plot_samples=False,
        plot_percentage=0.05
):
    """
    Generate and save metric dataset with two-scaler pipeline.

    Args:
        cfg: Configuration
        val_sequences_raw: [N, L, F] RAW validation sequences (not smoothed, not standardized)
        feature_columns: List of feature column names
        scaler_raw: Scaler fitted on RAW training data
        scaler_smoothed: Scaler fitted on SMOOTHED training data
        anomaly_strategy: 'corrupt_validation', 'use_original', or 'both'
        original_anomaly_sequences: Optional pre-processed anomaly sequences
        original_anomaly_labels: Optional anomaly labels
        output_dir: Output directory
        force_regenerate: Force regeneration even if file exists
        plot_samples: Whether to plot sample sequences
        plot_percentage: Percentage of samples to plot

    Returns:
        str: Path to saved dataset file
    """
    import torch
    import numpy as np
    from pathlib import Path
    from dataset.sentinel import Dataset_seq

    print(f"\n   🔄 TWO-SCALER ANOMALY PIPELINE")
    print(f"      Simulates: Raw anomalies → Production filter → Production scaler")
    print(f"      Input: {val_sequences_raw.shape} (RAW, not smoothed, not standardized)")
    print(f"      Strategy: {anomaly_strategy}")

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    N, L, F = val_sequences_raw.shape

    # ============================================================
    # BUILD FILENAME (like old version - detailed)
    # ============================================================
    model_name = cfg.model.get('name', 'unknown_model')
    dataset_name = cfg.dataset.get('name', 'dataset')
    exp_name = cfg.opt.get('exp_name', 'experiment')
    seed = cfg.opt.get('seed', 42)
    seq_len = cfg.dataset.seq_in_length

    # Get corruption config
    corruption_cfg = cfg.opt.get('corruption_config', {})
    anomalies_type = corruption_cfg.get('anomalies_type', ['GWN'])
    delta_mean = corruption_cfg.get('delta_mean', 0.8)

    # Predict standardization status
    force_destd = cfg.opt.get('force_destandardization', False)
    if anomaly_strategy == 'use_original':
        predicted_is_standardized = True
    elif anomaly_strategy in ['corrupt_validation', 'both']:
        predicted_is_standardized = not force_destd
    else:
        predicted_is_standardized = True

    # Build detailed filename like old version
    scale_suffix = "" if predicted_is_standardized else "_original"
    anomalies_str = '_'.join(anomalies_type)
    smooth_suffix = "" if cfg.dataset.get('smooth', None) is None else '_smoothed'

    filename = f"metric_{model_name}_{exp_name}_{dataset_name}_{seq_len}_{anomaly_strategy}_{anomalies_str}_seed{seed}{scale_suffix}_{delta_mean}{smooth_suffix}.pt"
    save_path = output_dir / filename

    # Check if already exists
    if save_path.exists() and not force_regenerate:
        print(f"\n      ℹ️  Metric dataset already exists: {save_path}")
        print(f"      ℹ️  Skipping generation (use force_regenerate=True to recreate)")
        print(f"      ℹ️  Loading existing dataset for verification...")

        try:
            saved_data = torch.load(save_path, map_location='cpu')
            print(f"      ✓ Loaded existing dataset: {len(saved_data['dataset'])} sequences")
            return str(save_path)
        except Exception as e:
            print(f"      ⚠️  Failed to load existing dataset: {e}")
            print(f"      → Will regenerate...")


    if force_regenerate and save_path.exists():
        print(f"\n      🔄 force_regenerate=True: Regenerating dataset...")
        os.remove(save_path)

    # ============================================================
    # PREPARE CLEAN SEQUENCES (baseline - production pipeline)
    # ============================================================
    print(f"\n      Preparing clean sequences (Production pipeline: Raw → Smooth → Std)...")

    val_sequences_smoothed = apply_smoothing_to_sequences(
        sequences=val_sequences_raw,
        cfg=cfg,
        feature_columns=feature_columns
    )

    clean_sequences = standardize_sequences(
        sequences=val_sequences_smoothed,
        scaler=scaler_smoothed,
        feature_columns=feature_columns
    )

    print(f"      ✓ Clean sequences: {clean_sequences.shape} (smoothed + std with scaler_smoothed)")

    # ============================================================
    # STRATEGY: CORRUPT VALIDATION
    # ============================================================
    if anomaly_strategy == 'corrupt_validation':
        print(f"\n      📊 Strategy: Corrupt Validation (Simulate raw anomalies)")

        # STEP 1: Standardize with scaler_raw
        print(f"\n      Step 1/5: Standardizing RAW sequences (with scaler_raw)...")
        val_sequences_std_from_raw = standardize_sequences(
            sequences=val_sequences_raw,
            scaler=scaler_raw,
            feature_columns=feature_columns
        )
        print(f"         ✓ Standardized (raw scale): {val_sequences_std_from_raw.shape}")

        # STEP 2: Inject anomalies
        print(f"\n      Step 2/5: Injecting anomalies...")
        corrupted_sequences, labels, anomaly_types, affected_channels, is_standardized, indices_to_corrupt = corrupt_sequences_wombat(
            sequences=val_sequences_std_from_raw,
            feature_columns=feature_columns,
            anomalies_type=anomalies_type,
            delta_mean=delta_mean,
            corruption_ratio=corruption_cfg.get('corruption_ratio', 1.0),
            random_seed=corruption_cfg.get('random_seed', 123),
            scaler=None,
            force_destandardization=False,
            target_channels=corruption_cfg.get('target_channels', None)
        )
        print(f"         ✓ Anomalies injected: {corrupted_sequences.shape}")

        # Save intermediate for plotting (before smoothing)
        corrupted_sequences_before_smooth = corrupted_sequences.copy()

        # STEP 3: De-standardize
        print(f"\n      Step 3/5: De-standardizing (with scaler_raw)...")
        corrupted_sequences_raw = destandardize_sequences(
            sequences=corrupted_sequences,
            scaler=scaler_raw,
            feature_columns=feature_columns
        )
        print(f"         ✓ De-standardized to raw scale: {corrupted_sequences_raw.shape}")

        # STEP 4: Smooth
        print(f"\n      Step 4/5: Applying smoothing to anomalous sequences...")
        corrupted_sequences_smoothed = apply_smoothing_to_sequences(
            sequences=corrupted_sequences_raw,
            cfg=cfg,
            feature_columns=feature_columns
        )
        print(f"         ✓ Smoothed: {corrupted_sequences_smoothed.shape}")

        # STEP 5: Re-standardize
        print(f"\n      Step 5/5: Re-standardizing (with scaler_smoothed)...")
        corrupted_sequences_final = standardize_sequences(
            sequences=corrupted_sequences_smoothed,
            scaler=scaler_smoothed,
            feature_columns=feature_columns
        )
        print(f"         ✓ Re-standardized (smoothed scale): {corrupted_sequences_final.shape}")
        print(f"\n      ✅ Pipeline complete: Raw anomalies → Production processing → Model input")

        # Verification
        if cfg.opt.get('verify_anomaly_smoothness', False):
            print(f"\n      🔍 Verifying anomaly smoothness...")
            verify_anomaly_smoothness_comparison(
                sequences_clean=clean_sequences,
                sequences_anomalous_before_smooth=corrupted_sequences_before_smooth,
                sequences_anomalous_after_smooth=corrupted_sequences_final,
                labels=labels,
                feature_columns=feature_columns
            )

        # Build dataset
        dataset, is_standardized_final = _create_corrupted_sequences_dataset(
            cfg=cfg,
            clean_sequences=clean_sequences,
            corrupted_sequences=corrupted_sequences_final,
            labels=labels,
            anomaly_types=anomaly_types,
            affected_channels=affected_channels,
            indices_to_corrupt=indices_to_corrupt,
            feature_columns=feature_columns,
            include_clean=True
        )

        print(f"\n      ✓ Final dataset: {len(dataset)} sequences")
        print(f"         - Clean: {N} sequences")
        print(f"         - Anomalous: {(labels.sum(axis=(1, 2)) > 0).sum()} sequences")

        # Plot samples
        if plot_samples:
            print(f"\n      📊 Plotting sample sequences for verification...")

            # Extract only the corrupted sequences for plotting
            was_corrupted = (labels.sum(axis=(1, 2)) > 0)
            actually_corrupted_indices = np.array([i for i in indices_to_corrupt if was_corrupted[i]])

            # Get original clean and final corrupted
            original_for_plot = clean_sequences[actually_corrupted_indices]  # Clean versions
            corrupted_for_plot = corrupted_sequences_final[actually_corrupted_indices]  # After full pipeline
            labels_for_plot = labels[actually_corrupted_indices]
            anomaly_types_for_plot = [anomaly_types[i] for i in actually_corrupted_indices]
            affected_channels_for_plot = [affected_channels[i] for i in actually_corrupted_indices]

            plot_corrupted_sequences_samples(
                cfg=cfg,
                original_sequences=original_for_plot,
                corrupted_sequences=corrupted_for_plot,
                labels=labels_for_plot,
                anomaly_types=anomaly_types_for_plot,
                affected_channels=affected_channels_for_plot,
                feature_columns=feature_columns,
                dataset_filepath=str(save_path),
                sample_percentage=plot_percentage,
                max_samples=20,
                random_seed=seed
            )

    # ============================================================
    # STRATEGY: USE ORIGINAL
    # ============================================================
    elif anomaly_strategy == 'use_original':
        print(f"\n      📊 Strategy: Use Original Anomalies")

        if original_anomaly_sequences is None or original_anomaly_labels is None:
            raise ValueError("Original anomaly sequences/labels required for 'use_original' strategy")

        all_sequences = np.concatenate([clean_sequences, original_anomaly_sequences], axis=0)
        clean_labels = np.zeros((N, 1, L), dtype=np.float32)
        all_labels = np.concatenate([clean_labels, original_anomaly_labels], axis=0)

        all_anomaly_types = ['normal'] * N + ['original'] * len(original_anomaly_sequences)
        all_affected_channels = ['none'] * N + ['multiple'] * len(original_anomaly_sequences)

        dataset = Dataset_seq(
            sequences=all_sequences,
            targets=all_sequences,
            anomaly_labels=all_labels,
            transform=None
        )

        dataset.anomaly_types = all_anomaly_types
        dataset.affected_channels = all_affected_channels
        is_standardized_final = True

        print(f"      ✓ Final dataset: {len(dataset)} sequences")

    # ============================================================
    # STRATEGY: BOTH
    # ============================================================
    elif anomaly_strategy == 'both':
        print(f"\n      📊 Strategy: Both (Corrupted + Original)")

        if original_anomaly_sequences is None or original_anomaly_labels is None:
            raise ValueError("Original anomaly sequences/labels required for 'both' strategy")

        # Generate corrupted sequences (same as corrupt_validation)
        print(f"\n      Part 1: Generating corrupted sequences...")

        val_sequences_std_from_raw = standardize_sequences(
            sequences=val_sequences_raw,
            scaler=scaler_raw,
            feature_columns=feature_columns
        )

        corrupted_sequences, labels, anomaly_types, affected_channels, is_standardized, indices_to_corrupt = corrupt_sequences_wombat(
            sequences=val_sequences_std_from_raw,
            feature_columns=feature_columns,
            anomalies_type=anomalies_type,
            delta_mean=delta_mean,
            corruption_ratio=corruption_cfg.get('corruption_ratio', 1.0),
            random_seed=corruption_cfg.get('random_seed', 123),
            scaler=None,
            force_destandardization=False,
            target_channels=corruption_cfg.get('target_channels', None)
        )

        corrupted_sequences_raw = destandardize_sequences(corrupted_sequences, scaler_raw, feature_columns)
        corrupted_sequences_smoothed = apply_smoothing_to_sequences(corrupted_sequences_raw, cfg, feature_columns)
        corrupted_sequences_final = standardize_sequences(corrupted_sequences_smoothed, scaler_smoothed,
                                                          feature_columns)

        print(f"         ✓ Corrupted sequences: {corrupted_sequences_final.shape}")

        # Combine
        print(f"\n      Part 2: Combining all sequences...")

        all_sequences = np.concatenate([
            clean_sequences,
            corrupted_sequences_final,
            original_anomaly_sequences
        ], axis=0)

        clean_labels = np.zeros((N, 1, L), dtype=np.float32)
        all_labels = np.concatenate([clean_labels, labels, original_anomaly_labels], axis=0)

        all_anomaly_types = (
                ['normal'] * N +
                anomaly_types +
                ['original'] * len(original_anomaly_sequences)
        )

        all_affected_channels = (
                ['none'] * N +
                affected_channels +
                ['multiple'] * len(original_anomaly_sequences)
        )

        dataset = Dataset_seq(
            sequences=all_sequences,
            targets=all_sequences,
            anomaly_labels=all_labels,
            transform=None
        )

        dataset.anomaly_types = all_anomaly_types
        dataset.affected_channels = all_affected_channels
        is_standardized_final = True

        print(f"      ✓ Final dataset: {len(dataset)} sequences")

        # Plot only corrupted part
        if plot_samples:
            print(f"\n      📊 Plotting corrupted samples (original anomalies not plotted)...")

            was_corrupted = (labels.sum(axis=(1, 2)) > 0)
            actually_corrupted_indices = np.array([i for i in indices_to_corrupt if was_corrupted[i]])

            original_for_plot = clean_sequences[actually_corrupted_indices]
            corrupted_for_plot = corrupted_sequences_final[actually_corrupted_indices]
            labels_for_plot = labels[actually_corrupted_indices]
            anomaly_types_for_plot = [anomaly_types[i] for i in actually_corrupted_indices]
            affected_channels_for_plot = [affected_channels[i] for i in actually_corrupted_indices]

            plot_corrupted_sequences_samples(
                cfg=cfg,
                original_sequences=original_for_plot,
                corrupted_sequences=corrupted_for_plot,
                labels=labels_for_plot,
                anomaly_types=anomaly_types_for_plot,
                affected_channels=affected_channels_for_plot,
                feature_columns=feature_columns,
                dataset_filepath=str(save_path),
                sample_percentage=plot_percentage,
                max_samples=20,
                random_seed=seed
            )

    else:
        raise ValueError(f"Unknown anomaly strategy: {anomaly_strategy}")

    # ============================================================
    # SAVE DATASET
    # ============================================================
    print(f"\n      💾 Saving metric dataset...")

    torch.save({
        'dataset': dataset,
        'metadata': {
            'strategy': anomaly_strategy,
            'n_sequences': len(dataset),
            'is_standardized': is_standardized_final,
            'is_smoothed': True,
            'uses_two_scalers': True,
            'feature_columns': feature_columns,
            'scaler_smoothed_params': serialize_scaler(scaler_smoothed),
            'scaler_raw_params': serialize_scaler(scaler_raw),
            'smoothing_config': cfg.dataset.get('smooth'),
            'corruption_config': {
                'anomalies_type': list(anomalies_type),
                'delta_mean': float(delta_mean),
                'corruption_ratio': float(corruption_cfg.get('corruption_ratio', 1.0)),
                'random_seed': corruption_cfg.get('random_seed', 123),
            },
            'seed': seed,
            'model_name': model_name,
            'dataset_name': dataset_name,
            'exp_name': exp_name,
            'seq_len': seq_len,
        }
    }, save_path)

    print(f"      ✓ Saved: {save_path}")
    print(f"      ✓ File size: {save_path.stat().st_size / (1024 * 1024):.2f} MB")
    print(f"      ✓ Filename: {filename}")

    return str(save_path)

def _create_original_anomaly_sequences_dataset(
        clean_sequences,
        original_anomaly_sequences,
        original_anomaly_labels,
        force_destandardization,
        scaler,
        feature_columns,
        include_clean=True
):
    """
    Create dataset with original anomaly sequences.

    Args:
        clean_sequences: [N, L, F] clean validation sequences
        original_anomaly_sequences: [M, L, F] original anomalous sequences
        original_anomaly_labels: [M, 1, L] binary labels for anomalies
        force_destandardization: Whether to destandardize
        scaler: Fitted scaler
        feature_columns: Feature names
        include_clean: If True, return [clean + original]. If False, return ONLY original.

    Returns:
        dataset: Dataset_seq
        is_standardized: bool
    """
    import numpy as np
    from dataset.sentinel import Dataset_seq

    print(f"\n   🔧 Creating original anomaly dataset...")
    print(f"      - Include clean: {include_clean}")

    if original_anomaly_sequences is None or len(original_anomaly_sequences) == 0:
        print(f"      ⚠️  No original anomaly sequences provided!")
        return None, True

    print(f"      - Clean sequences: {clean_sequences.shape}")
    print(f"      - Original anomaly sequences: {original_anomaly_sequences.shape}")
    print(f"      - Original anomaly labels: {original_anomaly_labels.shape}")

    # Check destandardization
    if force_destandardization:
        print(f"      ⚠️  Destandardizing sequences")
        from preprocessing.scaling import inverse_transform_array

        # Destandardize both clean and original
        clean_sequences_destd = inverse_transform_array(clean_sequences, scaler, feature_columns)
        original_anomaly_sequences_destd = inverse_transform_array(original_anomaly_sequences, scaler, feature_columns)

        is_standardized = False
    else:
        clean_sequences_destd = clean_sequences
        original_anomaly_sequences_destd = original_anomaly_sequences
        is_standardized = True

    N_clean = len(clean_sequences)
    M_original = len(original_anomaly_sequences)
    L = clean_sequences.shape[1]

    if include_clean:
        # Standard: clean + original anomalies
        all_sequences = np.concatenate([clean_sequences_destd, original_anomaly_sequences_destd], axis=0)

        clean_labels = np.zeros((N_clean, 1, L), dtype=np.float32)
        all_labels = np.concatenate([clean_labels, original_anomaly_labels], axis=0)

        # Metadata (assume original anomalies don't have type/channel info)
        all_anomaly_types = ['normal'] * N_clean + ['original'] * M_original
        all_affected_channels = ['none'] * N_clean + ['unknown'] * M_original

        print(f"      ✓ Combined dataset: {all_sequences.shape}")
        print(f"         - Clean: {N_clean} sequences")
        print(f"         - Original anomalies: {M_original} sequences")
    else:
        # Only original anomalies (for "both" strategy)
        all_sequences = original_anomaly_sequences_destd
        all_labels = original_anomaly_labels

        # Metadata
        all_anomaly_types = ['original'] * M_original
        all_affected_channels = ['unknown'] * M_original

        print(f"      ✓ Original anomalies only: {all_sequences.shape}")
        print(f"         - Original anomalies: {M_original} sequences")

    # Verify label statistics
    n_anomalous_timesteps = original_anomaly_labels.sum()
    total_timesteps = original_anomaly_labels.size
    anomaly_density = n_anomalous_timesteps / total_timesteps
    print(f"      ✓ Anomaly density in original sequences: {anomaly_density:.2%}")

    # Create Dataset_seq
    dataset = Dataset_seq(
        sequences=all_sequences,
        targets=all_sequences,
        anomaly_labels=all_labels,
        transform=None
    )

    # Metadata
    dataset.anomaly_types = all_anomaly_types
    dataset.affected_channels = all_affected_channels

    print(f"      ✓ Dataset created: {len(dataset)} sequences")

    return dataset, is_standardized


def corrupt_sequences_wombat(
        sequences: np.ndarray,
        feature_columns: List[str],
        anomalies_type: List[str],
        delta_mean: float,
        corruption_ratio: float,
        random_seed: int,
        scaler: Optional[object] = None,
        force_destandardization: bool = False,
        target_channels: Optional[List[str]] = None
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], bool, np.ndarray]:
    """
    Inject WOMBAT anomalies with full standardization handling.

    Returns:
        corrupted_sequences: [N, L, F]
        labels: [N, 1, L]
        anomaly_types_list: List[str]
        affected_channels_list: List[str]
        is_standardized: bool
        indices_corrupted: np.ndarray - Indices that were selected for corruption
    """

    from tqdm import tqdm
    from omegaconf import ListConfig

    # ✅ Convert OmegaConf ListConfig to Python lists
    if isinstance(feature_columns, ListConfig):
        feature_columns = list(feature_columns)
    if isinstance(anomalies_type, ListConfig):
        anomalies_type = list(anomalies_type)
    if target_channels is not None and isinstance(target_channels, ListConfig):
        target_channels = list(target_channels)

    # Set seed
    np.random.seed(random_seed)

    N, L, F = sequences.shape
    sequences_std = sequences.copy()  # Work on standardized data

    print(f"\n   🧪 WOMBAT Injection Configuration:")
    print(f"      - Input: {sequences.shape} (ASSUMED STANDARDIZED)")
    print(f"      - Anomaly types: {anomalies_type}")
    print(f"      - Delta: {delta_mean} (FIXED)")
    print(f"      - Corruption ratio: {corruption_ratio:.1%}")
    print(f"      - Force destandardization: {force_destandardization}")
    print(f"      - Random seed: {random_seed}")

    # Available channels
    if target_channels is not None:
        available_channels = [
            feature_columns.index(ch) for ch in target_channels
            if ch in feature_columns
        ]
    else:
        available_channels = list(range(F))

    # Initialize
    corrupted_std = sequences_std.copy()
    labels = np.zeros((N, 1, L), dtype=np.float32)
    anomaly_types_arr = np.array(['normal'] * N, dtype=object)
    affected_channels_idx = np.full(N, -1, dtype=np.int32)

    # Cache
    fitted_cache = {}

    # Select to corrupt
    n_to_corrupt = int(N * corruption_ratio)
    indices_to_corrupt = np.random.choice(N, size=n_to_corrupt, replace=False)

    print(f"\n      💉 Step 1: Injecting anomalies ({n_to_corrupt}/{N} sequences)...")

    # =========================================================
    # STEP 1: INJECT (on standardized)
    # =========================================================
    for seq_idx in tqdm(indices_to_corrupt, desc="      Injecting", unit="seq", ncols=100, leave=False):
        anomaly_type = np.random.choice(anomalies_type)
        channel_idx = np.random.choice(available_channels)

        cache_key = (channel_idx, anomaly_type)

        # ✅ LAZY FIT
        if cache_key not in fitted_cache:
            if anomaly_type not in ANOMALIES_REGISTRY:
                tqdm.write(f"         ⚠️  Unknown: {anomaly_type}")
                continue

            anomaly_cls = ANOMALIES_REGISTRY[anomaly_type]
            anomaly_obj = anomaly_cls(delta=delta_mean)

            # Fit on ALL sequences for this channel
            channel_data = sequences_std[:, :, channel_idx]  # [N, L]

            try:
                anomaly_obj.fit(channel_data)
                tqdm.write(
                    f"         🔧 Fitted {anomaly_type} on channel {channel_idx} ({feature_columns[channel_idx]})")
            except Exception as e:
                tqdm.write(f"         ⚠️  Fit failed: {e}")
                anomaly_obj.fit(channel_data[:1])

            fitted_cache[cache_key] = anomaly_obj
        else:
            anomaly_obj = fitted_cache[cache_key]

        # Extract channel
        original_seq = sequences_std[seq_idx, :, channel_idx].copy()  # [L]

        # ✅ DISTORT
        try:
            distorted_seq = anomaly_obj.distort(original_seq.reshape(1, -1))[0]  # [L]
        except Exception as e:
            tqdm.write(f"         ⚠️  Distort failed: {e}")
            distorted_seq = original_seq.copy()

        # ✅ CRITICAL: Compute mask ONLY where actually changed
        mask = (np.abs(original_seq - distorted_seq) > 1e-10)

        if mask.any():
            # Update corrupted ONLY where changed
            corrupted_std[seq_idx, mask, channel_idx] = distorted_seq[mask]

            # Labels = 1 where changed
            labels[seq_idx, 0, mask] = 1.0

            # Metadata
            anomaly_types_arr[seq_idx] = anomaly_type
            affected_channels_idx[seq_idx] = channel_idx

    total_anomalous = labels.sum()
    print(f"\n      ✅ Injection complete:")
    print(f"         - Corrupted: {n_to_corrupt}/{N} sequences")
    print(f"         - Fitted injectors: {len(fitted_cache)}")
    print(f"         - Anomalous timesteps: {total_anomalous:.0f}/{N * L} ({total_anomalous / (N * L) * 100:.1f}%)")

    # =========================================================
    # STEP 2: DESTANDARDIZATION (if requested)
    # =========================================================
    if force_destandardization:
        if scaler is None:
            raise ValueError("force_destandardization=True requires scaler")

        print(f"\n      📈 Step 2: Destandardizing...")

        from preprocessing.scaling import inverse_transform_array

        # Destandardize corrupted
        corrupted_destd = inverse_transform_array(corrupted_std, scaler, feature_columns)

        # Destandardize original
        sequences_destd = inverse_transform_array(sequences_std, scaler, feature_columns)

        # ✅ PRESERVE: Replace normal timesteps with original values
        print(f"         ↳ Preserving original values for normal timesteps...")
        normal_mask = (labels[:, 0, :] == 0)  # [N, L]

        for f_idx in range(F):
            corrupted_destd[:, :, f_idx][normal_mask] = sequences_destd[:, :, f_idx][normal_mask]

        corrupted_sequences = corrupted_destd
        is_standardized = False

        print(f"         ✓ Destandardized (back to original scale)")

    else:
        print(f"\n      📈 Step 2: Keeping standardized")

        # ✅ PRESERVE even in standardized (safety - handles numerical errors)
        print(f"         ↳ Preserving original values for normal timesteps...")
        normal_mask = (labels[:, 0, :] == 0)  # [N, L]

        #for f_idx in range(F):
        #    corrupted_std[:, :, f_idx][normal_mask] = sequences_std[:, :, f_idx][normal_mask]

        corrupted_sequences = corrupted_std
        is_standardized = True

        print(f"         ✓ Output is STANDARDIZED")

    # Convert metadata
    anomaly_types_list = anomaly_types_arr.tolist()
    affected_channels_list = [
        feature_columns[int(idx)] if idx >= 0 else 'none'
        for idx in affected_channels_idx
    ]

    # ✅ NEW: Return also the indices that were corrupted
    return (
        corrupted_sequences,
        labels,
        anomaly_types_list,
        affected_channels_list,
        is_standardized,
        indices_to_corrupt  # ← CRITICAL: Gli indici selezionati per corruzione
    )


def _create_corrupted_sequences_dataset(
        cfg,
        clean_sequences,
        corrupted_sequences,
        labels,
        anomaly_types,
        affected_channels,
        indices_to_corrupt,
        feature_columns,
        include_clean=True
):
    """
    Create dataset with corrupted sequences (for two-scaler pipeline).

    Args:
        clean_sequences: (N, L, F) - clean sequences (already smoothed+std)
        corrupted_sequences: (N, L, F) - corrupted sequences (already smoothed+std)
        labels: (N, 1, L) - anomaly labels
        anomaly_types: list of anomaly type names
        affected_channels: list of affected channel names
        indices_to_corrupt: indices that were selected for corruption
        include_clean: if True, include clean sequences in final dataset
    """
    from dataset.sentinel import Dataset_seq
    import numpy as np

    print(f"\n   🔧 Creating corrupted sequences dataset...")
    print(f"      - Clean sequences: {clean_sequences.shape}")
    print(f"      - Corrupted sequences: {corrupted_sequences.shape}")
    print(f"      - Include clean: {include_clean}")

    # Find which sequences were actually corrupted
    was_corrupted = (labels.sum(axis=(1, 2)) > 0)  # [N] boolean

    # Filter to get only actually corrupted sequences
    actually_corrupted_indices = []
    for idx in indices_to_corrupt:
        if was_corrupted[idx]:
            actually_corrupted_indices.append(idx)

    actually_corrupted_indices = np.array(actually_corrupted_indices)

    # Extract corrupted sequences
    corrupted_only = corrupted_sequences[actually_corrupted_indices]
    labels_corrupted = labels[actually_corrupted_indices]
    anomaly_types_corrupted = [anomaly_types[i] for i in actually_corrupted_indices]
    affected_channels_corrupted = [affected_channels[i] for i in actually_corrupted_indices]

    print(f"      ✓ Actually corrupted: {len(actually_corrupted_indices)}/{len(indices_to_corrupt)}")

    N_clean = len(clean_sequences)
    L = clean_sequences.shape[1]

    if include_clean:
        # Combine clean + corrupted
        all_sequences = np.concatenate([clean_sequences, corrupted_only], axis=0)

        clean_labels = np.zeros((N_clean, 1, L), dtype=np.float32)
        all_labels = np.concatenate([clean_labels, labels_corrupted], axis=0)

        all_anomaly_types = ['normal'] * N_clean + anomaly_types_corrupted
        all_affected_channels = ['none'] * N_clean + affected_channels_corrupted

        print(f"      ✓ Combined dataset: {all_sequences.shape}")
        print(f"         - Clean: {N_clean} sequences")
        print(f"         - Corrupted: {len(corrupted_only)} sequences")
    else:
        # Only corrupted
        all_sequences = corrupted_only
        all_labels = labels_corrupted

        all_anomaly_types = anomaly_types_corrupted
        all_affected_channels = affected_channels_corrupted

        print(f"      ✓ Corrupted-only dataset: {all_sequences.shape}")

    # Create dataset
    dataset = Dataset_seq(
        sequences=all_sequences,
        targets=all_sequences,
        anomaly_labels=all_labels,
        transform=None
    )

    dataset.anomaly_types = all_anomaly_types
    dataset.affected_channels = all_affected_channels

    # Store corruption mapping
    dataset.corruption_mapping = {
        'original_indices': actually_corrupted_indices,
        'seed': cfg.opt.get('corruption_config', {}).get('random_seed', 123),
        'n_clean': N_clean if include_clean else 0,
        'n_corrupted': len(corrupted_only)
    }

    return dataset, True  # is_standardized = True


def plot_corrupted_sequences_samples(
        cfg,
        original_sequences,
        corrupted_sequences,
        labels,
        anomaly_types,
        affected_channels,
        feature_columns,
        dataset_filepath,
        sample_percentage=0.05,
        max_samples=10,
        random_seed=42
):
    """
    Plot comparison between original and corrupted sequences.
    Shows affected channel + neighboring channels for verification.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.ioff()
    import numpy as np
    from pathlib import Path

    print(f"\n📊 Generating comparison plots...")

    # Convert to numpy if needed
    if torch.is_tensor(original_sequences):
        original_sequences = original_sequences.numpy()
    if torch.is_tensor(corrupted_sequences):
        corrupted_sequences = corrupted_sequences.numpy()
    if torch.is_tensor(labels):
        labels = labels.numpy()

    N, L, F = original_sequences.shape

    # Determine number of samples
    n_samples = min(int(N * sample_percentage), max_samples)

    if n_samples == 0:
        print(f"   ⚠️  No samples to plot (N={N}, percentage={sample_percentage})")
        return

    print(f"   - Total sequences: {N}")
    print(f"   - Sample percentage: {sample_percentage:.1%}")
    print(f"   - Sequences to plot: {n_samples}")

    # Create output directory
    dataset_path = Path(dataset_filepath)
    dataset_name = dataset_path.stem
    smooth_suffix = "" if cfg.dataset.get('smooth', None) is None else '_smoothed'
    output_dir = dataset_path.parent / f"{dataset_name}{smooth_suffix}_plots"

    if os.path.exists(output_dir):
         shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"   - Output directory: {output_dir}")

    # Sample sequences (only those with anomalies)
    np.random.seed(random_seed)

    # Find sequences with anomalies
    has_anomaly = labels.sum(axis=(1, 2)) > 0
    anomalous_indices = np.where(has_anomaly)[0]

    if len(anomalous_indices) == 0:
        print(f"   ⚠️  No anomalous sequences found to plot")
        return

    # Sample from anomalous sequences
    n_samples_actual = min(n_samples, len(anomalous_indices))
    sampled_indices = np.random.choice(anomalous_indices, size=n_samples_actual, replace=False)

    print(f"   - Anomalous sequences available: {len(anomalous_indices)}")
    print(f"   - Plotting: {n_samples_actual} sequences")

    # Convert feature_columns to list
    if isinstance(feature_columns, (list, tuple)):
        feature_columns_list = list(feature_columns)
    else:
        feature_columns_list = list(feature_columns)

    # Convert metadata to numpy arrays
    if not isinstance(anomaly_types, np.ndarray):
        anomaly_types = np.array(anomaly_types)
    if not isinstance(affected_channels, np.ndarray):
        affected_channels = np.array(affected_channels)

    # Plot each sampled sequence
    for plot_idx, seq_idx in enumerate(sampled_indices):
        # Extract data for this sequence
        orig_seq = original_sequences[seq_idx]  # [L, F]
        corr_seq = corrupted_sequences[seq_idx]  # [L, F]
        label_seq = labels[seq_idx, 0, :]  # [L]
        anom_type = anomaly_types[seq_idx]
        aff_channel = affected_channels[seq_idx]

        # Find affected channel index
        try:
            affected_idx = feature_columns_list.index(aff_channel)
        except ValueError:
            print(f"   ⚠️  Warning: Channel '{aff_channel}' not found. Skipping seq {seq_idx}")
            continue

        # ✅ Get neighboring channels
        prev_idx = affected_idx - 1 if affected_idx > 0 else None
        next_idx = affected_idx + 1 if affected_idx < F - 1 else None

        # Create figure with 5 subplots
        fig, axes = plt.subplots(5, 1, figsize=(16, 14), sharex=True)
        fig.suptitle(
            f'Sequence {seq_idx} | Anomaly: {anom_type} | Affected: {aff_channel}',
            fontsize=16, fontweight='bold'
        )

        timesteps = np.arange(L)

        # Highlight anomaly region (for all plots)
        if label_seq.any():
            anomaly_timesteps = np.where(label_seq > 0)[0]
            if len(anomaly_timesteps) > 0:
                anom_start = anomaly_timesteps[0]
                anom_end = anomaly_timesteps[-1]

        # ============================================================
        # Plot 1: AFFECTED CHANNEL - Original vs Corrupted
        # ============================================================
        ax1 = axes[0]

        orig_feature = orig_seq[:, affected_idx]
        corr_feature = corr_seq[:, affected_idx]

        ax1.plot(timesteps, orig_feature, label=f'{aff_channel} (original)',
                 alpha=0.8, linewidth=2, color='blue')
        ax1.plot(timesteps, corr_feature, label=f'{aff_channel} (corrupted)',
                 alpha=0.8, linewidth=2, color='red', linestyle='--')

        ax1.set_ylabel('Value', fontsize=11)
        ax1.set_title(f'🎯 AFFECTED: {aff_channel}', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3)

        if label_seq.any() and len(anomaly_timesteps) > 0:
            ax1.axvspan(anom_start, anom_end, alpha=0.2, color='red')

        # ============================================================
        # Plot 2: PREVIOUS CHANNEL - Should be UNCHANGED
        # ============================================================
        ax2 = axes[1]

        if prev_idx is not None:
            prev_channel = feature_columns_list[prev_idx]
            prev_orig = orig_seq[:, prev_idx]
            prev_corr = corr_seq[:, prev_idx]

            ax2.plot(timesteps, prev_orig, label=f'{prev_channel} (original)',
                     alpha=0.8, linewidth=2, color='blue')
            ax2.plot(timesteps, prev_corr, label=f'{prev_channel} (corrupted)',
                     alpha=0.8, linewidth=2, color='green', linestyle='--')

            # Check if unchanged
            max_diff = np.abs(prev_orig - prev_corr).max()
            status = "✓ UNCHANGED" if max_diff < 1e-8 else f"⚠️ CHANGED! (max diff: {max_diff:.2e})"
            ax2.set_title(f'Previous Channel: {prev_channel} | {status}',
                          fontsize=11, fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'No previous channel', ha='center', va='center',
                     transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Previous Channel: N/A', fontsize=11, fontweight='bold')

        ax2.set_ylabel('Value', fontsize=11)
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(True, alpha=0.3)

        if label_seq.any() and len(anomaly_timesteps) > 0:
            ax2.axvspan(anom_start, anom_end, alpha=0.1, color='gray')

        # ============================================================
        # Plot 3: NEXT CHANNEL - Should be UNCHANGED
        # ============================================================
        ax3 = axes[2]

        if next_idx is not None:
            next_channel = feature_columns_list[next_idx]
            next_orig = orig_seq[:, next_idx]
            next_corr = corr_seq[:, next_idx]

            ax3.plot(timesteps, next_orig, label=f'{next_channel} (original)',
                     alpha=0.8, linewidth=2, color='blue')
            ax3.plot(timesteps, next_corr, label=f'{next_channel} (corrupted)',
                     alpha=0.8, linewidth=2, color='green', linestyle='--')

            # Check if unchanged
            max_diff = np.abs(next_orig - next_corr).max()
            status = "✓ UNCHANGED" if max_diff < 1e-8 else f"⚠️ CHANGED! (max diff: {max_diff:.2e})"
            ax3.set_title(f'Next Channel: {next_channel} | {status}',
                          fontsize=11, fontweight='bold')
        else:
            ax3.text(0.5, 0.5, 'No next channel', ha='center', va='center',
                     transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Next Channel: N/A', fontsize=11, fontweight='bold')

        ax3.set_ylabel('Value', fontsize=11)
        ax3.legend(loc='upper right', fontsize=10)
        ax3.grid(True, alpha=0.3)

        if label_seq.any() and len(anomaly_timesteps) > 0:
            ax3.axvspan(anom_start, anom_end, alpha=0.1, color='gray')

        # ============================================================
        # Plot 4: DIFFERENCE (Affected channel only)
        # ============================================================
        ax4 = axes[3]

        diff_feature = corr_feature - orig_feature

        ax4.plot(timesteps, diff_feature, label=f'{aff_channel}',
                 alpha=0.8, linewidth=2, color='purple')
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
        ax4.set_ylabel('Difference', fontsize=11)
        ax4.set_title(f'Difference (Corrupted - Original) - {aff_channel}',
                      fontsize=11, fontweight='bold')
        ax4.legend(loc='upper right', fontsize=10)
        ax4.grid(True, alpha=0.3)

        if label_seq.any() and len(anomaly_timesteps) > 0:
            ax4.axvspan(anom_start, anom_end, alpha=0.2, color='red')

        # ============================================================
        # Plot 5: ANOMALY LABELS
        # ============================================================
        ax5 = axes[4]

        ax5.fill_between(timesteps, 0, label_seq,
                         where=(label_seq > 0),
                         color='red', alpha=0.6, label='Anomaly')
        ax5.fill_between(timesteps, 0, 1,
                         where=(label_seq == 0),
                         color='green', alpha=0.3, label='Normal')

        ax5.set_xlabel('Timestep', fontsize=12)
        ax5.set_ylabel('Label', fontsize=11)
        ax5.set_title('Anomaly Labels', fontsize=11, fontweight='bold')
        ax5.set_ylim(-0.1, 1.1)
        ax5.set_yticks([0, 1])
        ax5.set_yticklabels(['Normal', 'Anomaly'])
        ax5.legend(loc='upper right', fontsize=10)
        ax5.grid(True, alpha=0.3, axis='x')

        # ============================================================
        # Save figure
        # ============================================================
        plt.tight_layout()

        output_filename = f"seq_{seq_idx:06d}_{anom_type}_{aff_channel}.png"
        output_path = output_dir / output_filename

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        # Progress
        if (plot_idx + 1) % 10 == 0 or (plot_idx + 1) == n_samples_actual:
            print(f"   ✓ Generated {plot_idx + 1}/{n_samples_actual} plots")

    print(f"\n   ✅ All plots saved to: {output_dir}")
    print(f"      - Total plots: {n_samples_actual}")

    # Create summary statistics
    summary_file = output_dir / "summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"PLOT SUMMARY\n")
        f.write(f"=" * 80 + "\n\n")
        f.write(f"Dataset: {dataset_filepath}\n")
        f.write(f"Total sequences: {N}\n")
        f.write(f"Anomalous sequences: {len(anomalous_indices)}\n")
        f.write(f"Plotted sequences: {n_samples_actual}\n")
        f.write(f"Sample percentage: {sample_percentage:.1%}\n\n")

        f.write(f"Anomaly type distribution in plots:\n")
        unique_types, counts = np.unique(
            anomaly_types[sampled_indices],
            return_counts=True
        )
        for atype, count in zip(unique_types, counts):
            f.write(f"  - {atype}: {count}\n")

        f.write(f"\nAffected channels distribution:\n")
        unique_channels, counts = np.unique(
            affected_channels[sampled_indices],
            return_counts=True
        )
        for channel, count in zip(unique_channels, counts):
            f.write(f"  - {channel}: {count}\n")

    print(f"   ✓ Summary saved to: {summary_file}")


def apply_smoothing_to_dataframe(df, cfg, feature_columns):
    """
    Apply smoothing to a DataFrame.

    Args:
        df: DataFrame with time series data
        cfg: configuration
        feature_columns: list of columns to smooth

    Returns:
        df_smoothed: DataFrame with smoothed data
    """
    from scipy.signal import savgol_filter
    import numpy as np

    smoothing_method = cfg.dataset.get('smoothing_method', None)

    if smoothing_method is None:
        print(f"      ℹ️  No smoothing configured - returning original data")
        return df.copy()

    df_smoothed = df.copy()

    if smoothing_method == 'savgol':
        window_length = cfg.dataset.get('smoothing_window', 5)
        polyorder = cfg.dataset.get('smoothing_polyorder', 2)

        # Ensure odd window
        if window_length % 2 == 0:
            window_length += 1
        window_length = max(window_length, polyorder + 2)

        print(f"      - Method: Savitzky-Golay (window={window_length}, poly={polyorder})")

        for col in feature_columns:
            if col in df_smoothed.columns:
                df_smoothed[col] = savgol_filter(
                    df_smoothed[col].values,
                    window_length,
                    polyorder,
                    mode='interp'
                )

    elif smoothing_method == 'moving_average':
        window_length = cfg.dataset.get('smoothing_window', 5)

        print(f"      - Method: Moving Average (window={window_length})")

        for col in feature_columns:
            if col in df_smoothed.columns:
                df_smoothed[col] = df_smoothed[col].rolling(
                    window=window_length,
                    center=True,
                    min_periods=1
                ).mean()

    else:
        raise ValueError(f"Unknown smoothing method: {smoothing_method}")

    return df_smoothed




def verify_anomaly_smoothness_comparison(
        sequences_clean,
        sequences_anomalous_before_smooth,
        sequences_anomalous_after_smooth,
        labels,
        feature_columns,
        n_samples=3
):
    """
    Verify that anomaly smoothing pipeline works correctly.

    Args:
        sequences_clean: (N, L, F) - clean sequences (smoothed+std with scaler_smoothed)
        sequences_anomalous_before_smooth: (N, L, F) - after injection (std with scaler_raw)
        sequences_anomalous_after_smooth: (N, L, F) - final (smoothed+std with scaler_smoothed)
        labels: (N, 1, L) - anomaly labels
        feature_columns: list of feature names
        n_samples: number of samples to plot
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Find anomalous sequences
    anomalous_mask = (labels.sum(axis=(1, 2)) > 0)
    anomalous_indices = np.where(anomalous_mask)[0]

    if len(anomalous_indices) == 0:
        print(f"         ⚠️  No anomalous sequences found!")
        return

    # Sample
    n_samples = min(n_samples, len(anomalous_indices))
    sample_indices = np.random.choice(anomalous_indices, size=n_samples, replace=False)

    for idx in sample_indices:
        # Find affected channel
        affected_timesteps = np.where(labels[idx, 0, :] > 0)[0]
        if len(affected_timesteps) == 0:
            continue

        # Find which channel was affected
        ch_idx = 0
        for c in range(len(feature_columns)):
            if np.any(sequences_anomalous_before_smooth[idx, affected_timesteps[0], c] !=
                      sequences_clean[idx, affected_timesteps[0], c]):
                ch_idx = c
                break

        L = sequences_clean.shape[1]

        fig, axes = plt.subplots(4, 1, figsize=(14, 12))

        # 1. Clean (smoothed baseline)
        axes[0].plot(sequences_clean[idx, :, ch_idx], label='Clean (smoothed+std)', color='green', linewidth=2)
        axes[0].fill_between(range(L), -3, 3,
                             where=labels[idx, 0, :] > 0, alpha=0.2, color='red', label='Anomaly region')
        axes[0].set_title(f'Sequence {idx} - Channel: {feature_columns[ch_idx]} - CLEAN (Smoothed Baseline)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylabel('Standardized value')

        # 2. Anomaly BEFORE smoothing (std with scaler_raw)
        axes[1].plot(sequences_anomalous_before_smooth[idx, :, ch_idx],
                     label='Anomaly (after injection, NOT smoothed)', color='red', linewidth=2)
        axes[1].fill_between(range(L), -3, 3,
                             where=labels[idx, 0, :] > 0, alpha=0.2, color='red')
        axes[1].set_title('BEFORE Smoothing (Standardized with scaler_raw)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylabel('Standardized value')

        # 3. Anomaly AFTER smoothing (std with scaler_smoothed)
        axes[2].plot(sequences_anomalous_after_smooth[idx, :, ch_idx],
                     label='Anomaly (smoothed+restd)', color='orange', linewidth=2)
        axes[2].fill_between(range(L), -3, 3,
                             where=labels[idx, 0, :] > 0, alpha=0.2, color='red')
        axes[2].set_title('AFTER Smoothing (Standardized with scaler_smoothed)')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylabel('Standardized value')

        # 4. Differences (smoothness metric)
        diff_clean = np.abs(np.diff(sequences_clean[idx, :, ch_idx]))
        diff_before = np.abs(np.diff(sequences_anomalous_before_smooth[idx, :, ch_idx]))
        diff_after = np.abs(np.diff(sequences_anomalous_after_smooth[idx, :, ch_idx]))

        axes[3].plot(diff_clean, label='|Δ Clean|', color='green', alpha=0.7)
        axes[3].plot(diff_before, label='|Δ Before Smooth|', color='red', alpha=0.7)
        axes[3].plot(diff_after, label='|Δ After Smooth|', color='orange', alpha=0.7)
        axes[3].set_title('First Differences (Smoothness Check)')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        axes[3].set_ylabel('|Δx|')
        axes[3].set_xlabel('Timestep')

        # Stats in anomaly region
        anomaly_mask_diff = labels[idx, 0, :-1] > 0
        if anomaly_mask_diff.any():
            ratio_before = diff_before[anomaly_mask_diff].mean() / diff_clean.mean() if diff_clean.mean() > 0 else 0
            ratio_after = diff_after[anomaly_mask_diff].mean() / diff_clean.mean() if diff_clean.mean() > 0 else 0

            axes[3].text(0.02, 0.98,
                         f'Smoothness Ratio (in anomaly region):\n'
                         f'Before smoothing: {ratio_before:.2f}x\n'
                         f'After smoothing: {ratio_after:.2f}x',
                         transform=axes[3].transAxes,
                         verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(f'./anomaly_smoothing_verification_seq{idx}.png', dpi=150)
        plt.close()

        print(f"         ✓ Saved: ./anomaly_smoothing_verification_seq{idx}.png")


def apply_smoothing_to_sequences(sequences, cfg, feature_columns):
    """
    Apply smoothing to 3D sequences array.

    Args:
        sequences: (N, L, F) array
        cfg: configuration
        feature_columns: list of feature names

    Returns:
        sequences_smoothed: (N, L, F) array
    """
    from scipy.signal import savgol_filter
    import numpy as np

    smoothing_method = cfg.dataset.get('smoothing_method', None)

    if smoothing_method is None:
        return sequences.copy()

    N, L, F = sequences.shape
    sequences_smoothed = sequences.copy()

    if smoothing_method == 'savgol':
        window_length = cfg.dataset.get('smoothing_window', 5)
        polyorder = cfg.dataset.get('smoothing_polyorder', 2)

        if window_length % 2 == 0:
            window_length += 1
        window_length = max(window_length, polyorder + 2)

        print(f"         - Savitzky-Golay: window={window_length}, poly={polyorder}")

        for i in range(N):
            for f in range(F):
                signal = sequences[i, :, f]
                if len(signal) >= window_length:
                    try:
                        sequences_smoothed[i, :, f] = savgol_filter(
                            signal,
                            window_length,
                            polyorder,
                            mode='interp'
                        )
                    except Exception as e:
                        sequences_smoothed[i, :, f] = signal

    elif smoothing_method == 'moving_average':
        window_length = cfg.dataset.get('smoothing_window', 5)

        print(f"         - Moving Average: window={window_length}")

        kernel = np.ones(window_length) / window_length

        for i in range(N):
            for f in range(F):
                sequences_smoothed[i, :, f] = np.convolve(
                    sequences[i, :, f],
                    kernel,
                    mode='same'
                )

    return sequences_smoothed


def standardize_sequences(sequences, scaler, feature_columns):
    """
    Standardize sequences using fitted scaler.

    Args:
        sequences: (N, L, F) array
        scaler: fitted sklearn scaler
        feature_columns: list of feature names

    Returns:
        sequences_std: (N, L, F) standardized array
    """
    N, L, F = sequences.shape

    # Reshape to 2D
    sequences_flat = sequences.reshape(-1, F)  # (N*L, F)

    # Transform
    sequences_std_flat = scaler.transform(sequences_flat)

    # Reshape back
    sequences_std = sequences_std_flat.reshape(N, L, F)

    return sequences_std


def destandardize_sequences(sequences, scaler, feature_columns):
    """
    De-standardize sequences using fitted scaler.

    Args:
        sequences: (N, L, F) standardized array
        scaler: fitted sklearn scaler
        feature_columns: list of feature names

    Returns:
        sequences_raw: (N, L, F) de-standardized array
    """
    N, L, F = sequences.shape

    # Reshape to 2D
    sequences_flat = sequences.reshape(-1, F)  # (N*L, F)

    # Inverse transform
    sequences_raw_flat = scaler.inverse_transform(sequences_flat)

    # Reshape back
    sequences_raw = sequences_raw_flat.reshape(N, L, F)

    return sequences_raw

def serialize_scaler(scaler):
    """
    Serialize scaler to dictionary for saving.

    Args:
        scaler: Fitted scaler object

    Returns:
        Dictionary with scaler parameters
    """
    scaler_params = {
        'type': type(scaler).__name__,
        'mean': scaler.mean_.tolist() if hasattr(scaler, 'mean_') else None,
        'scale': scaler.scale_.tolist() if hasattr(scaler, 'scale_') else None,
        'var': scaler.var_.tolist() if hasattr(scaler, 'var_') else None,
        'min': scaler.min_.tolist() if hasattr(scaler, 'min_') else None,
        'data_min': scaler.data_min_.tolist() if hasattr(scaler, 'data_min_') else None,
        'data_max': scaler.data_max_.tolist() if hasattr(scaler, 'data_max_') else None,
        'center': scaler.center_.tolist() if hasattr(scaler, 'center_') else None,
    }

    return scaler_params
