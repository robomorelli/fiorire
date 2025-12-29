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

def generate_and_save_metric_dataset(
        cfg: DictConfig,
        df_scaled,
        val_indices,
        feature_columns,
        scaler,
        output_dir='./metric_loaders/',
        force_regenerate=False,
        plot_samples=True,  # ✅ New parameter
        plot_percentage=0.05  # ✅ New parameter
):
    """
    Generate metric dataset and save as PyTorch Dataset.
    DataLoader will be created in the trainer with appropriate batch_size.

    DEFAULT: Data is STANDARDIZED (is_standardized=True)
    OPTIONAL: Set force_destandardization=True to save in ORIGINAL SCALE
    """
    strategy = cfg.opt.get('anomaly_strategy', 'none')

    if strategy == 'none':
        print("\n   ℹ️  Strategy='none': No metric dataset")
        return None

    # Create filename base
    dataset_name = cfg.dataset.get('name', 'dataset')
    exp_name = cfg.opt.get('exp_name', 'experiment')
    seed = cfg.opt.get('seed', 42)
    seq_len = cfg.dataset.seq_in_length
    anomaly_column = cfg.dataset.get('is_anomaly_column')

    # ✅ PREDICT is_standardized from config (before generating dataset)
    force_destd = cfg.opt.get('force_destandardization', False)

    if strategy == 'use_original':
        # Original anomalies are always from df_scaled → standardized
        predicted_is_standardized = True
    elif strategy in ['corrupt_validation', 'both']:
        # Depends on force_destandardization flag
        predicted_is_standardized = not force_destd
    else:
        predicted_is_standardized = True  # Default

    # ✅ Build filename with correct suffix
    if predicted_is_standardized:
        scale_suffix = ""
        scale_info = "STANDARDIZED (ready for model)"
    else:
        scale_suffix = "_original"
        scale_info = "ORIGINAL SCALE (needs standardization)"

    filename = f"metric_{exp_name}_{dataset_name}_{strategy}_seed{seed}{scale_suffix}.pt"
    filepath = os.path.join(output_dir, filename)

    # ✅ Check if already exists BEFORE generating
    if os.path.exists(filepath) and not force_regenerate:
        print(f"\n   ✓ Metric dataset already exists: {filepath}")
        print(f"     (Use force_regenerate=True to regenerate)")
        return filepath

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

    # Add suffix based on scale
    if is_standardized:
        scale_suffix = ""
        scale_info = "STANDARDIZED (ready for model)"
    else:
        scale_suffix = "_original"
        scale_info = "ORIGINAL SCALE (needs standardization)"


    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n📊 Saving metric dataset:")
    print(f"   - Strategy: {strategy}")
    print(f"   - Output: {filepath}")
    print(f"   - Data scale: {scale_info}")

    if not is_standardized:
        print(f"\n   ⚠️  WARNING: Data saved in ORIGINAL SCALE")
        print(f"      This metric dataset will require standardization before use in training")

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

    # ✅ Save Dataset + metadata
    save_dict = {
        'dataset': metric_dataset,
        'metadata': metadata,
    }

    torch.save(save_dict, filepath)

    # ✅ Calculate anomaly stats (gestisci sia numpy che torch)
    if hasattr(metric_dataset, 'anomaly_labels') and metric_dataset.anomaly_labels is not None:
        labels = metric_dataset.anomaly_labels

        # Convert to torch if numpy
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)

        # Now safely use torch operations
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
    if plot_samples and strategy in ['corrupt_validation', 'both']:
        from utils.load_dataset import extract_sequences_from_indices

        print(f"\n📊 Generating comparison plots...")

        # Re-extract original clean sequences for comparison
        original_sequences = extract_sequences_from_indices(
            df_scaled,
            val_indices,
            seq_len,
            feature_columns,
            perc_overlap=cfg.opt.get("metric_seq_overlap", 0)
        )

        # Extract data from dataset
        corrupted_sequences = metric_dataset.sequences
        labels = metric_dataset.anomaly_labels
        anomaly_types = metric_dataset.anomaly_types if hasattr(metric_dataset, 'anomaly_types') else None
        affected_channels = metric_dataset.affected_channels if hasattr(metric_dataset, 'affected_channels') else None

        # ✅ Extract only the corrupted part for plotting
        # Dataset structure: [all_clean, corrupted_duplicates]
        # We want to compare original clean with their corrupted versions

        corruption_ratio = cfg.opt.get('corruption_config', {}).get('corruption_ratio', 1.0)
        n_clean = len(original_sequences)
        n_corrupted = int(n_clean * corruption_ratio)

        if n_corrupted > 0:
            # Get corrupted sequences (last n_corrupted in dataset)
            corrupted_part = corrupted_sequences[-n_corrupted:]
            labels_part = labels[-n_corrupted:]
            anomaly_types_part = anomaly_types[-n_corrupted:] if anomaly_types is not None else None
            affected_channels_part = affected_channels[-n_corrupted:] if affected_channels is not None else None

            # Get corresponding original sequences
            # (the ones that were duplicated and corrupted)
            seed = cfg.opt.get('corruption_config', {}).get('random_seed', 123)
            np.random.seed(seed)
            indices_corrupted = np.random.choice(n_clean, size=n_corrupted, replace=False)
            original_part = original_sequences[indices_corrupted]

            # Plot
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
                random_seed=seed
            )
        else:
            print(f"   ⚠️  No corrupted sequences to plot (corruption_ratio=0)")

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
    Create dataset with BOTH normal and corrupted sequences.

    Strategy:
    - Extract ALL clean validation sequences (100% normal)
    - Duplicate corruption_ratio% of them
    - Corrupt the duplicates
    - Concatenate: [all_clean, corrupted_duplicates]

    Example with corruption_ratio=0.3:
    - 1000 clean sequences (normal)
    - + 300 duplicated and corrupted sequences (anomalous)
    - Total: 1300 sequences in metric dataset

    Returns:
        dataset: Dataset_seq object with sequences + labels
        is_standardized: True (default) or False (if forced)
    """
    from utils.anomaly_corruption import corrupt_sequences_wombat
    from utils.load_dataset import extract_sequences_from_indices

    print(f"   🔧 Creating WOMBAT-corrupted validation dataset...")

    # Extract ALL clean validation sequences (ALREADY SCALED)
    all_clean_sequences = extract_sequences_from_indices(
        df_scaled,
        val_indices,
        seq_len,
        feature_columns,
        perc_overlap=cfg.opt.get("metric_seq_overlap", 0)
    )  # [N, L, F] - ALREADY SCALED

    N_total = len(all_clean_sequences)

    print(f"      - Total clean sequences: {all_clean_sequences.shape}")
    print(f"      - Data already scaled (using training scaler)")

    # Get corruption config
    corruption_ratio = cfg.opt.get('corruption_config', {}).get('corruption_ratio', 1.0)
    random_seed = cfg.opt.get('corruption_config', {}).get('random_seed', 123)
    force_destd = cfg.opt.get('force_destandardization', False)

    if force_destd:
        print(f"      ⚠️  WARNING: force_destandardization=True detected!")
        print(f"      This will produce ORIGINAL SCALE data (not recommended for training)")

    # ✅ Determine number of sequences to corrupt
    n_to_corrupt = int(N_total * corruption_ratio)

    print(f"\n      📊 Corruption strategy:")
    print(f"         - Corruption ratio: {corruption_ratio:.1%}")
    print(f"         - Clean sequences (normal): {N_total}")
    print(f"         - Sequences to corrupt: {n_to_corrupt}")
    print(f"         - Total sequences: {N_total + n_to_corrupt}")

    if n_to_corrupt == 0:
        print(f"\n      ⚠️  Corruption ratio is 0 - no corrupted sequences will be added")
        # Return only clean sequences

        # Create labels (all zeros - no anomalies)
        labels = torch.zeros((N_total, seq_len, 1), dtype=torch.float32)
        anomaly_types = np.array(['normal'] * N_total, dtype=object)
        affected_channels = np.array(['none'] * N_total, dtype=object)

        dataset = Dataset_seq(
            sequences=torch.from_numpy(all_clean_sequences).float(),
            anomaly_labels=labels
        )
        dataset.anomaly_types = anomaly_types
        dataset.affected_channels = affected_channels

        is_standardized = True  # Clean sequences are standardized

        print(f"\n      ✅ Dataset created: {len(dataset)} sequences (all normal)")
        return dataset, is_standardized

    # ✅ Randomly select sequences to duplicate and corrupt
    np.random.seed(random_seed)
    indices_to_corrupt = np.random.choice(N_total, size=n_to_corrupt, replace=False)
    sequences_to_corrupt = all_clean_sequences[indices_to_corrupt]  # [n_to_corrupt, L, F]

    print(f"\n      💉 Corrupting {n_to_corrupt} duplicated sequences...")

    # Corrupt the duplicates
    corrupted_sequences, corrupted_labels, anomaly_types_corrupted, affected_channels_corrupted, is_standardized = corrupt_sequences_wombat(
        sequences=sequences_to_corrupt,
        feature_columns=feature_columns,
        cfg=cfg,
        standardization_mode="skip",
        scaler=scaler,
        force_destandardization=force_destd,
        random_seed=random_seed,
        verbose=True
    )

    # ✅ Create labels for clean sequences (all zeros)
    clean_labels = torch.zeros((N_total, seq_len, 1), dtype=torch.float32)
    clean_anomaly_types = np.array(['normal'] * N_total, dtype=object)
    clean_affected_channels = np.array(['none'] * N_total, dtype=object)

    # ✅ Concatenate: [all_clean, corrupted_duplicates]
    all_sequences = np.concatenate([
        all_clean_sequences,  # [N_total, L, F] - normal
        corrupted_sequences  # [n_to_corrupt, L, F] - anomalous
    ], axis=0)  # [N_total + n_to_corrupt, L, F]

    all_labels = torch.cat([
        clean_labels,  # [N_total, L, 1]
        torch.from_numpy(corrupted_labels).float()  # [n_to_corrupt, L, 1]
    ], dim=0)  # [N_total + n_to_corrupt, L, 1]

    all_anomaly_types = np.concatenate([
        clean_anomaly_types,  # [N_total]
        anomaly_types_corrupted  # [n_to_corrupt]
    ])

    all_affected_channels = np.concatenate([
        clean_affected_channels,  # [N_total]
        affected_channels_corrupted  # [n_to_corrupt]
    ])

    # Create dataset
    dataset = Dataset_seq(
        sequences=torch.from_numpy(all_sequences).float(),
        anomaly_labels=all_labels
    )

    # Store metadata
    dataset.anomaly_types = all_anomaly_types
    dataset.affected_channels = all_affected_channels

    print(f"\n      ✅ Combined dataset created:")
    print(f"         - Normal sequences: {N_total}")
    print(f"         - Anomalous sequences: {n_to_corrupt}")
    print(f"         - Total: {len(dataset)} sequences")
    print(f"         - Overall anomaly ratio: {all_labels.mean():.2%}")
    print(f"         - Metadata: anomaly_types, affected_channels")

    if is_standardized:
        print(f"         - ✓ Data is STANDARDIZED (ready for model)")
    else:
        print(f"         - ⚠️  Data is in ORIGINAL SCALE (requires standardization)")

    # Summary
    print(f"\n      📋 Anomaly distribution:")
    unique_types, counts = np.unique(all_anomaly_types[all_anomaly_types != 'normal'], return_counts=True)
    for atype, count in zip(unique_types, counts):
        print(f"         * {atype}: {count}")

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
        max_samples=100,
        random_seed=42
):
    """
    Plot comparison between original and corrupted sequences.

    Shows ONLY the affected channel (corrupted feature).
    """
    import matplotlib
    matplotlib.use('Agg')  # ✅ Non-interactive backend
    import matplotlib.pyplot as plt

    # ✅ Disable interactive mode
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
    has_anomaly = labels.sum(axis=(1, 2)) > 0  # [N]
    anomalous_indices = np.where(has_anomaly)[0]

    if len(anomalous_indices) == 0:
        print(f"   ⚠️  No anomalous sequences found to plot")
        return

    # Sample from anomalous sequences
    n_samples_actual = min(n_samples, len(anomalous_indices))
    sampled_indices = np.random.choice(anomalous_indices, size=n_samples_actual, replace=False)

    print(f"   - Anomalous sequences available: {len(anomalous_indices)}")
    print(f"   - Plotting: {n_samples_actual} sequences")

    # ✅ Convert feature_columns to list for indexing
    if isinstance(feature_columns, (list, tuple)):
        feature_columns_list = list(feature_columns)
    else:
        feature_columns_list = list(feature_columns)

    # Plot each sampled sequence
    for plot_idx, seq_idx in enumerate(sampled_indices):
        # Extract data for this sequence
        orig_seq = original_sequences[seq_idx]  # [L, F]
        corr_seq = corrupted_sequences[seq_idx]  # [L, F]
        label_seq = labels[seq_idx, :, 0]  # [L]
        anom_type = anomaly_types[seq_idx]
        aff_channel = affected_channels[seq_idx]

        # ✅ Find the index of the affected channel
        try:
            affected_idx = feature_columns_list.index(aff_channel)
        except ValueError:
            print(f"   ⚠️  Warning: Channel '{aff_channel}' not found in feature_columns. Skipping seq {seq_idx}")
            continue

        # ✅ Extract ONLY the affected feature
        orig_feature = orig_seq[:, affected_idx]  # [L]
        corr_feature = corr_seq[:, affected_idx]  # [L]
        diff_feature = corr_feature - orig_feature  # [L]

        # Create figure
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        fig.suptitle(
            f'Sequence {seq_idx} | Anomaly: {anom_type} | Affected: {aff_channel}',
            fontsize=14, fontweight='bold'
        )

        timesteps = np.arange(L)

        # ============================================================
        # Plot 1: Original vs Corrupted (ONLY affected feature)
        # ============================================================
        ax1 = axes[0]

        # Original
        ax1.plot(timesteps, orig_feature,
                 label=f'{aff_channel} (original)',
                 alpha=0.8, linewidth=2, color='blue')

        # Corrupted
        ax1.plot(timesteps, corr_feature,
                 label=f'{aff_channel} (corrupted)',
                 alpha=0.8, linewidth=2, color='red', linestyle='--')

        ax1.set_ylabel('Value', fontsize=12)
        ax1.set_title(f'Original vs Corrupted - {aff_channel}', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=11)
        ax1.grid(True, alpha=0.3)

        # Highlight anomalous regions
        anomaly_regions = np.where(label_seq > 0)[0]
        if len(anomaly_regions) > 0:
            ax1.axvspan(anomaly_regions[0], anomaly_regions[-1],
                        alpha=0.2, color='red', label='Anomaly region')

        # ============================================================
        # Plot 2: Difference (Corrupted - Original) for affected feature
        # ============================================================
        ax2 = axes[1]

        ax2.plot(timesteps, diff_feature,
                 label=f'{aff_channel}',
                 alpha=0.8, linewidth=2, color='purple')

        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
        ax2.set_ylabel('Difference', fontsize=12)
        ax2.set_title(f'Difference (Corrupted - Original) - {aff_channel}', fontsize=12, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=11)
        ax2.grid(True, alpha=0.3)

        # Highlight anomalous regions
        if len(anomaly_regions) > 0:
            ax2.axvspan(anomaly_regions[0], anomaly_regions[-1],
                        alpha=0.2, color='red')

        # ============================================================
        # Plot 3: Anomaly Labels
        # ============================================================
        ax3 = axes[2]

        ax3.fill_between(timesteps, 0, label_seq,
                         where=(label_seq > 0),
                         color='red', alpha=0.6, label='Anomaly')
        ax3.fill_between(timesteps, 0, 1 - label_seq,
                         where=(label_seq == 0),
                         color='green', alpha=0.3, label='Normal')

        ax3.set_xlabel('Timestep', fontsize=12)
        ax3.set_ylabel('Label', fontsize=12)
        ax3.set_title('Anomaly Labels', fontsize=12, fontweight='bold')
        ax3.set_ylim(-0.1, 1.1)
        ax3.set_yticks([0, 1])
        ax3.set_yticklabels(['Normal', 'Anomaly'])
        ax3.legend(loc='upper right', fontsize=11)
        ax3.grid(True, alpha=0.3, axis='x')

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