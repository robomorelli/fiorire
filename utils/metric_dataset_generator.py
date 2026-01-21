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
        cfg: DictConfig,
        clean_sequences,
        feature_columns,
        scaler,
        original_anomaly_sequences=None,
        original_anomaly_labels=None,
        output_dir='./metric_datasets/',
        force_regenerate=False,
        plot_samples=True,
        plot_percentage=0.05
):
    """
    Generate metric dataset from pre-extracted sequences and save as PyTorch Dataset.

    Args:
        cfg: Configuration
        clean_sequences: [N, L, F] numpy array of clean, standardized sequences
        feature_columns: List of feature column names
        scaler: Fitted scaler (for potential destandardization)
        original_anomaly_sequences: [M, L, F] original anomalous sequences (optional)
        original_anomaly_labels: [M, 1, L] labels for original anomalies (optional)
        output_dir: Directory to save dataset
        force_regenerate: Force regeneration even if file exists
        plot_samples: Whether to generate comparison plots
        plot_percentage: Percentage of samples to plot

    Returns:
        filepath: Path to saved dataset file, or None if not generated

    NOTE: Input sequences MUST be standardized (from df_scaled)
    """
    import torch
    import numpy as np
    import os
    from datetime import datetime
    from dataset.sentinel import Dataset_seq, concatenate_datasets

    strategy = cfg.opt.get('anomaly_strategy', 'none')

    if strategy == 'none':
        print("\n   ℹ️  Strategy='none': No metric dataset")
        return None

    # Validate input
    if clean_sequences is None or len(clean_sequences) == 0:
        print("\n   ⚠️  No clean sequences provided!")
        return None

    print(f"\n   ✓ Received clean sequences: {clean_sequences.shape}")

    # Create filename
    model_name = cfg.model.get('name', 'unknown_model')
    dataset_name = cfg.dataset.get('name', 'dataset')
    exp_name = cfg.opt.get('exp_name', 'experiment')
    seed = cfg.opt.get('seed', 42)
    seq_len = cfg.dataset.seq_in_length

    # ✅ PREDICT is_standardized from config
    force_destd = cfg.opt.get('force_destandardization', False)

    if strategy == 'use_original':
        predicted_is_standardized = True
    elif strategy in ['corrupt_validation', 'both']:
        predicted_is_standardized = not force_destd
    else:
        predicted_is_standardized = True

    # Build filename with suffix
    scale_suffix = "" if predicted_is_standardized else "_original"
    filename = f"metric_{model_name}_{exp_name}_{dataset_name}_{seq_len}_{strategy}_seed{seed}{scale_suffix}_{cfg.opt.corruption_config.delta_mean}.pt"
    filepath = os.path.join(output_dir, filename)

    # ✅ Check if already exists
    if os.path.exists(filepath) and not force_regenerate:
        print(f"\n   ✓ Metric dataset already exists: {filepath}")
        print(f"     (Use force_regenerate=True to regenerate)")
        return filepath

    # ✅ Generate metric dataset based on strategy
    original_sequences_for_plot = None

    if strategy == 'corrupt_validation':
        metric_dataset, is_standardized = _create_corrupted_sequences_dataset(
            cfg=cfg,
            clean_sequences=clean_sequences,
            feature_columns=feature_columns,
            scaler=scaler,
            include_clean=True
        )
        original_sequences_for_plot = clean_sequences.copy()

    elif strategy == 'use_original':
        print("\n" + "!" * 80)
        print("⚠️  WARNING: Strategy 'use_original' is NOT FULLY TESTED!")
        print("   This strategy uses original anomalies from the dataset.")
        print("   Results may vary depending on data quality and anomaly distribution.")
        print("!" * 80 + "\n")

        metric_dataset, is_standardized = _create_original_anomaly_sequences_dataset(
            clean_sequences=clean_sequences,
            original_anomaly_sequences=original_anomaly_sequences,
            original_anomaly_labels=original_anomaly_labels,
            force_destandardization=force_destd,
            scaler=scaler,
            feature_columns=feature_columns,
            include_clean=True
        )

    elif strategy == 'both':
        print("\n" + "!" * 80)
        print("⚠️  WARNING: Strategy 'both' is NOT FULLY TESTED!")
        print("   This strategy combines corrupted validation + original anomalies.")
        print("   Results may vary depending on data quality and anomaly distribution.")
        print("!" * 80 + "\n")

        # Get ONLY corrupted (no clean)
        corrupted_dataset, is_std_corrupted = _create_corrupted_sequences_dataset(
            cfg=cfg,
            clean_sequences=clean_sequences,
            feature_columns=feature_columns,
            scaler=scaler,
            include_clean=False
        )

        # Get ONLY original anomalies (no clean)
        original_dataset, is_std_original = _create_original_anomaly_sequences_dataset(
            clean_sequences=clean_sequences,
            original_anomaly_sequences=original_anomaly_sequences,
            original_anomaly_labels=original_anomaly_labels,
            force_destandardization=force_destd,
            scaler=scaler,
            feature_columns=feature_columns,
            include_clean=False
        )

        # Verify standardization match
        if is_std_corrupted != is_std_original:
            raise ValueError(
                f"Standardization mismatch: corrupted={is_std_corrupted}, original={is_std_original}"
            )

        # Concatenate
        if corrupted_dataset is not None and original_dataset is not None:
            # Prepare clean sequences
            if not is_std_corrupted and scaler is not None:
                from preprocessing.scaling import inverse_transform_array
                clean_for_both = inverse_transform_array(clean_sequences, scaler, feature_columns)
            else:
                clean_for_both = clean_sequences

            # Combine sequences
            all_sequences = np.concatenate([
                clean_for_both,
                corrupted_dataset.sequences,
                original_dataset.sequences
            ], axis=0)

            # Combine labels
            N_clean = len(clean_sequences)
            L = clean_sequences.shape[1]

            clean_labels = np.zeros((N_clean, 1, L), dtype=np.float32)
            all_labels = np.concatenate([
                clean_labels,
                corrupted_dataset.anomaly_labels,
                original_dataset.anomaly_labels
            ], axis=0)

            # Create final dataset
            from dataset.sentinel import Dataset_seq
            metric_dataset = Dataset_seq(
                sequences=all_sequences,
                targets=all_sequences,
                anomaly_labels=all_labels,
                transform=None
            )

            # Combine metadata
            all_anomaly_types = (
                    ['normal'] * N_clean +
                    list(corrupted_dataset.anomaly_types) +
                    list(original_dataset.anomaly_types)
            )

            all_affected_channels = (
                    ['none'] * N_clean +
                    list(corrupted_dataset.affected_channels) +
                    list(original_dataset.affected_channels)
            )

            metric_dataset.anomaly_types = all_anomaly_types
            metric_dataset.affected_channels = all_affected_channels

            is_standardized = is_std_corrupted

            print(f"   ✓ Combined 'both' strategy: {len(metric_dataset)} total sequences")
            print(f"      - Clean: {N_clean}")
            print(f"      - Corrupted: {len(corrupted_dataset)}")
            print(f"      - Original: {len(original_dataset)}")
        elif corrupted_dataset is not None:
            metric_dataset = corrupted_dataset
            is_standardized = is_std_corrupted
            print(f"   ⚠️  Only corrupted dataset available")
        elif original_dataset is not None:
            metric_dataset = original_dataset
            is_standardized = is_std_original
            print(f"   ⚠️  Only original dataset available")
        else:
            print(f"   ⚠️  No datasets created!")
            return None

        original_sequences_for_plot = clean_sequences.copy()

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    if metric_dataset is None:
        print(f"   ⚠️  No metric dataset created")
        return None

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Scale info
    scale_info = "STANDARDIZED (ready for model)" if is_standardized else "ORIGINAL SCALE"

    print(f"\n📊 Saving metric dataset:")
    print(f"   - Strategy: {strategy}")
    print(f"   - Output: {filepath}")
    print(f"   - Data scale: {scale_info}")

    if not is_standardized:
        print(f"\n   ⚠️  WARNING: Data saved in ORIGINAL SCALE")
        print(f"      This metric dataset will require standardization before use")

    # Create metadata
    metadata = {
        'is_standardized': is_standardized,
        'strategy': strategy,
        'dataset_name': dataset_name,
        'exp_name': exp_name,
        'seed': seed,
        'seq_len': seq_len,
        'feature_columns': list(feature_columns),
        'num_sequences': len(metric_dataset),
        'creation_date': datetime.now().isoformat(),
    }

    # Add corruption config
    if strategy in ['corrupt_validation', 'both']:
        corruption_cfg = cfg.opt.get('corruption_config', {})
        metadata['corruption_config'] = {
            'anomalies_type': list(corruption_cfg.get('anomalies_type', [])),
            'delta_mean': float(corruption_cfg.get('delta_mean', 0.8)),
            'corruption_ratio': float(corruption_cfg.get('corruption_ratio', 1.0)),
        }

    # ✅ Save Dataset + metadata
    save_dict = {
        'dataset': metric_dataset,
        'metadata': metadata,
    }

    torch.save(save_dict, filepath)

    # Calculate anomaly stats
    if hasattr(metric_dataset, 'anomaly_labels') and metric_dataset.anomaly_labels is not None:
        labels = metric_dataset.anomaly_labels
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)
        n_anomalous_seqs = (labels.sum(dim=(1, 2)) > 0).sum().item()
        n_normal_seqs = len(metric_dataset) - n_anomalous_seqs
    else:
        n_normal_seqs = len(metric_dataset)
        n_anomalous_seqs = 0

    print(f"\n   ✅ Metric dataset saved:")
    print(f"      - Path: {filepath}")
    print(f"      - Total sequences: {len(metric_dataset)}")
    print(f"      - Normal sequences: {n_normal_seqs}")
    print(f"      - Anomalous sequences: {n_anomalous_seqs}")
    print(f"      - File size: {os.path.getsize(filepath) / 1024 ** 2:.2f} MB")
    print(f"      - is_standardized: {is_standardized}")

    # ✅ Generate plots if requested
    if plot_samples and original_sequences_for_plot is not None and strategy in ['corrupt_validation', 'both']:
        print(f"\n📊 Generating comparison plots...")

        # ✅ Use corruption mapping to extract PAIRED sequences
        if hasattr(metric_dataset, 'corruption_mapping'):
            mapping = metric_dataset.corruption_mapping
            original_indices = mapping['original_indices']
            n_corrupted = mapping['n_corrupted']

            # ✅ Extract ORIGINAL sequences for the corrupted ones
            original_part = original_sequences_for_plot[original_indices]  # [M, L, F]

            # ✅ Extract CORRUPTED sequences (last M in dataset)
            all_sequences = metric_dataset.sequences
            all_labels = metric_dataset.anomaly_labels
            corrupted_part = all_sequences[-n_corrupted:]  # [M, L, F]
            labels_part = all_labels[-n_corrupted:]  # [M, 1, L]

            # ✅ Extract metadata
            anomaly_types = metric_dataset.anomaly_types if hasattr(metric_dataset, 'anomaly_types') else None
            affected_channels = metric_dataset.affected_channels if hasattr(metric_dataset,
                                                                            'affected_channels') else None

            anomaly_types_part = anomaly_types[-n_corrupted:] if anomaly_types is not None else None
            affected_channels_part = affected_channels[-n_corrupted:] if affected_channels is not None else None

            # ✅ CRITICAL: Verify shapes match
            print(f"   ✓ Plotting alignment check:")
            print(f"      - Original sequences: {original_part.shape}")
            print(f"      - Corrupted sequences: {corrupted_part.shape}")
            print(f"      - Mapping pairs: {len(original_indices)}")

            if original_part.shape[0] != corrupted_part.shape[0]:
                print(f"   ❌ ERROR: Shape mismatch!")
                print(f"      - Original: {original_part.shape[0]}")
                print(f"      - Corrupted: {corrupted_part.shape[0]}")
                print(f"      Skipping plots")
            else:
                print(f"   ✓ Shapes match - generating plots")

                # ✅ NOW they correspond: original_part[i] ↔ corrupted_part[i]
                plot_corrupted_sequences_samples(
                    original_sequences=original_part,
                    corrupted_sequences=corrupted_part,
                    labels=labels_part,
                    anomaly_types=anomaly_types_part,
                    affected_channels=affected_channels_part,
                    feature_columns=feature_columns,
                    dataset_filepath=filepath,
                    sample_percentage=plot_percentage,
                    max_samples=10,
                    random_seed=mapping['seed']
                )
        else:
            print(f"   ⚠️  No corruption mapping found - skipping plots")
            print(f"      (Mapping is required for accurate plot alignment)")

    return filepath

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


def _create_corrupted_sequences_dataset(cfg, clean_sequences, feature_columns, scaler, include_clean=True):
    """Create dataset with corrupted sequences."""
    import numpy as np
    from dataset.sentinel import Dataset_seq

    print(f"\n   🔧 Creating corrupted validation dataset...")
    print(f"      - Input sequences: {clean_sequences.shape}")
    print(f"      - Include clean: {include_clean}")

    # Get config
    corruption_cfg = cfg.opt.get('corruption_config', {})
    corruption_ratio = corruption_cfg.get('corruption_ratio', 1.0)
    anomalies_type = corruption_cfg.get('anomalies_type', ['GWN'])
    delta_mean = corruption_cfg.get('delta_mean', 0.8)
    random_seed = corruption_cfg.get('random_seed', 123)
    target_channels = corruption_cfg.get('target_channels', None)

    force_destd = cfg.opt.get('force_destandardization', False)

    # Use WOMBAT - returns indices_to_corrupt
    corrupted_sequences, labels, anomaly_types, affected_channels, is_standardized, indices_to_corrupt = corrupt_sequences_wombat(
        sequences=clean_sequences,
        feature_columns=feature_columns,
        anomalies_type=anomalies_type,
        delta_mean=delta_mean,
        corruption_ratio=corruption_ratio,
        random_seed=random_seed,
        scaler=scaler,
        force_destandardization=force_destd,
        target_channels=target_channels
    )

    print(f"      ✓ Sequences corrupted: {corrupted_sequences.shape}")

    # ✅ FIX: Use indices directly instead of boolean indexing
    # Find which of the selected indices actually got corrupted
    was_corrupted_full = (labels.sum(axis=(1, 2)) > 0)  # [N_clean] boolean

    # Filter indices_to_corrupt to keep only those that were actually corrupted
    actually_corrupted_indices = []
    for idx in indices_to_corrupt:
        if was_corrupted_full[idx]:
            actually_corrupted_indices.append(idx)

    actually_corrupted_indices = np.array(actually_corrupted_indices)

    # ✅ Extract using direct indexing (preserves order!)
    corrupted_only = corrupted_sequences[actually_corrupted_indices]
    labels_corrupted = labels[actually_corrupted_indices]
    anomaly_types_corrupted = [anomaly_types[i] for i in actually_corrupted_indices]
    affected_channels_corrupted = [affected_channels[i] for i in actually_corrupted_indices]

    # ✅ Store the mapping (now in correct order!)
    original_indices_corrupted = actually_corrupted_indices

    print(f"      ✓ Corruption mapping:")
    print(f"         - Selected for corruption: {len(indices_to_corrupt)}")
    print(f"         - Actually corrupted: {len(original_indices_corrupted)}")

    N_clean = len(clean_sequences)
    L = clean_sequences.shape[1]

    if include_clean:
        # Standard: clean + corrupted
        all_sequences = np.concatenate([clean_sequences, corrupted_only], axis=0)

        clean_labels = np.zeros((N_clean, 1, L), dtype=np.float32)
        all_labels = np.concatenate([clean_labels, labels_corrupted], axis=0)

        # Metadata
        all_anomaly_types = ['normal'] * N_clean + anomaly_types_corrupted
        all_affected_channels = ['none'] * N_clean + affected_channels_corrupted

        print(f"      ✓ Combined dataset: {all_sequences.shape}")
        print(f"         - Clean: {N_clean} sequences")
        print(f"         - Corrupted: {len(corrupted_only)} sequences")
    else:
        # Only corrupted (for "both" strategy)
        all_sequences = corrupted_only
        all_labels = labels_corrupted

        # Metadata
        all_anomaly_types = anomaly_types_corrupted
        all_affected_channels = affected_channels_corrupted

        print(f"      ✓ Corrupted-only dataset: {all_sequences.shape}")
        print(f"         - Corrupted: {len(corrupted_only)} sequences")

    # If destandardized, apply inverse to clean too
    if include_clean and not is_standardized and scaler is not None:
        from preprocessing.scaling import inverse_transform_array
        print(f"      ℹ️  Clean sequences also destandardized for consistency")
        clean_sequences_destd = inverse_transform_array(clean_sequences, scaler, feature_columns)
        all_sequences[:N_clean] = clean_sequences_destd

    # Create dataset
    dataset = Dataset_seq(
        sequences=all_sequences,
        targets=all_sequences,
        anomaly_labels=all_labels,
        transform=None
    )

    # Metadata
    dataset.anomaly_types = all_anomaly_types
    dataset.affected_channels = all_affected_channels

    # ✅ Store corruption mapping
    dataset.corruption_mapping = {
        'original_indices': original_indices_corrupted,
        'seed': random_seed,
        'n_clean': N_clean,
        'n_corrupted': len(corrupted_only)
    }

    print(f"      ✓ Corruption mapping saved: {len(original_indices_corrupted)} pairs")

    return dataset, is_standardized


def plot_corrupted_sequences_samples(
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
    output_dir = dataset_path.parent / f"{dataset_name}_plots"
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