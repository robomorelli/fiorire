"""
Dataset loading and index generation utilities.
Minimal, focused on the refactored tensor-based system.
"""

from torchvision.transforms import transforms as T
from torchvision.transforms import Lambda

from config import *
import os
import re
from scipy import interpolate
from omegaconf import ListConfig
from torch.utils.data import SubsetRandomSampler
import json
from pathlib import Path
import pickle

from utils.metric_dataset_generator import generate_and_save_metric_dataset
from preprocessing.scaling import *


# =========================================================
# CORE FUNCTIONS - USATE NEL NUOVO SISTEMA
# =========================================================

def get_tune_value(tune_string, default=None):
    """
    Estrai valore da stringa tune_config.

    Args:
        tune_string: es. "tune.choice([1])" o "tune.choice([0.1, 0.2])"
        default: valore di default se parsing fallisce

    Returns:
        Primo valore della lista (o default)

    Examples:
        >>> get_tune_value("tune.choice([1])")
        1
        >>> get_tune_value("tune.choice([0.1, 0.2, 0.3])")
        0.1
        >>> get_tune_value("tune.uniform(0.0001, 0.001)")
        0.0001
    """

    if not isinstance(tune_string, str):
        # Già un valore concreto
        return tune_string

    # Pattern per tune.choice([...])
    match = re.search(r'tune\.choice\(\[(.*?)\]\)', tune_string)
    if match:
        values_str = match.group(1)
        # Parse la lista
        values = eval(f'[{values_str}]')  # Es: "1, 2, 3" -> [1, 2, 3]
        return values[0] if values else default

    # Pattern per tune.uniform(min, max)
    match = re.search(r'tune\.uniform\((.*?),\s*(.*?)\)', tune_string)
    if match:
        min_val = float(match.group(1))
        return min_val

    # Pattern per tune.loguniform(min, max)
    match = re.search(r'tune\.loguniform\((.*?),\s*(.*?)\)', tune_string)
    if match:
        min_val = float(match.group(1))
        return min_val

    # Pattern per tune.randint(min, max)
    match = re.search(r'tune\.randint\((.*?),\s*(.*?)\)', tune_string)
    if match:
        min_val = int(match.group(1))
        return min_val

    # Se non matcha nessun pattern, ritorna default
    return default

def load_and_preprocess_dataframe(cfg, data_path=None):
    """
    Load DataFrame and apply preprocessing:
    - Feature selection
    - Target handling
    - Column removal
    - NaN removal
    - Optional upsampling

    Args:
        cfg: Configuration
        data_path: Optional path override (if None, uses cfg.dataset.data_path)

    Returns:
        df: Preprocessed DataFrame
    """
    # 1. Load raw data
    # Use override path if provided, otherwise use config path
    if data_path is None:
        data_path = cfg.dataset.data_path

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

    # Drop exclude_columns immediately from the dataframe
    exclude_columns = cfg.dataset.get("exclude_columns", []) or []
    if exclude_columns:
        cols_to_drop = [c for c in exclude_columns if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            print(f"   🔧 Excluded columns (dropped): {cols_to_drop}")

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
    df_sel = df[columns]
    total_rows = len(df_sel)
    nan_per_col = df_sel.isnull().sum()
    rows_with_nan = df_sel.isnull().any(axis=1).sum()
    print(f"   📊 Shape before dropna: {df.shape}")
    print(f"   📊 Righe totali: {total_rows} | Righe con almeno un NaN: {rows_with_nan}")
    print(f"   📊 NaN per colonna (solo colonne con NaN):")
    nan_cols = nan_per_col[nan_per_col > 0]
    if len(nan_cols) == 0:
        print(f"      Nessun NaN nelle colonne selezionate.")
    else:
        for col, cnt in nan_cols.items():
            print(f"      {col}: {cnt} NaN ({100*cnt/total_rows:.2f}%)")
    df = df_sel.dropna()
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
        anomalous_data: Dict with 'window_start_indices', 'labels', 'all_indices_set' (if return_anomalies=True), else None
    """
    import numpy as np

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

    def get_anomaly_windows_with_labels(anomalous_idx, df, ano_col, seq_len, total_len):
        """
        Create windows around anomalous points and extract labels for each window.

        Args:
            anomalous_idx: Array of indices where df[ano_col] == 1
            df: Original dataframe with anomaly column
            ano_col: Name of anomaly column
            seq_len: Sequence length
            total_len: Total length of dataframe

        Returns:
            window_start_indices: np.array of valid window start indices [M]
            window_labels: np.array of binary labels [M, L] where L=seq_len
            window_indices_set: set of all indices in anomalous windows (for filtering)
        """
        # Dictionary to track which timesteps are anomalous in each window
        window_anomaly_positions = {}

        for anomaly_idx in anomalous_idx:
            # Find all valid windows that can contain this anomaly point
            # A window [start, start+seq_len) contains anomaly_idx if:
            # start <= anomaly_idx < start + seq_len
            # Therefore: anomaly_idx - seq_len + 1 <= start <= anomaly_idx

            earliest_start = max(0, anomaly_idx - seq_len + 1)
            latest_start = min(anomaly_idx, total_len - seq_len)

            if latest_start < 0:
                continue

            for start_idx in range(earliest_start, latest_start + 1):
                # Ensure window doesn't exceed dataframe
                if start_idx + seq_len > total_len:
                    continue

                if start_idx not in window_anomaly_positions:
                    window_anomaly_positions[start_idx] = set()

                # Position of anomaly within this window (relative to start)
                relative_pos = anomaly_idx - start_idx
                if 0 <= relative_pos < seq_len:
                    window_anomaly_positions[start_idx].add(relative_pos)

        # Convert to arrays
        window_start_indices = np.array(sorted(window_anomaly_positions.keys()), dtype=np.int64)

        if len(window_start_indices) == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32).reshape(0, seq_len), set()

        # Create label arrays [M, L] by directly reading from DataFrame
        window_labels = np.zeros((len(window_start_indices), seq_len), dtype=np.float32)

        for i, start_idx in enumerate(window_start_indices):
            # Extract actual labels from DataFrame for this window
            end_idx = start_idx + seq_len
            if end_idx <= len(df):
                label_values = df[ano_col].iloc[start_idx:end_idx].values
                window_labels[i, :] = label_values.astype(np.float32)

        # Also return set of all indices (for filtering - backward compatibility)
        window_indices_set = set()
        for start_idx in window_start_indices:
            window_indices_set.update(range(start_idx, min(start_idx + seq_len, total_len)))

        return window_start_indices, window_labels, window_indices_set

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
    # ANOMALY FILTERING WITH LABELS
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

    # ✅ NEW: Get anomaly windows WITH labels
    anomaly_window_start_indices, anomaly_window_labels, anomalous_window_indexes = \
        get_anomaly_windows_with_labels(
            anomalous_idx=full_anomalous_idx,
            df=df,
            ano_col=ano_col,
            seq_len=seq_len,
            total_len=total_len
        )

    print(f"      - Anomaly windows found: {len(anomaly_window_start_indices)}")
    if len(anomaly_window_start_indices) > 0:
        avg_anomaly_density = anomaly_window_labels.mean()
        print(f"      - Average anomaly density in windows: {avg_anomaly_density:.2%}")

    # Filter train/val to remove anomalous windows
    train_anomalous_indexes = np.intersect1d(full_anomalous_idx, train_indexes)
    val_anomalous_indexes = np.intersect1d(full_anomalous_idx, val_indexes)

    # Get window indices as sets for filtering
    _, _, train_window_set = get_anomaly_windows_with_labels(
        train_anomalous_indexes, df, ano_col, seq_len, total_len
    )
    _, _, val_window_set = get_anomaly_windows_with_labels(
        val_anomalous_indexes, df, ano_col, seq_len, total_len
    )

    train_normal_indexes = np.setdiff1d(train_indexes, list(train_window_set), assume_unique=True)
    val_normal_indexes = np.setdiff1d(val_indexes, list(val_window_set), assume_unique=True)

    # For scaler fitting: only features (exclude anomaly column)
    scaling_cols = df.columns.difference([ano_col]) if ano_col in df.columns else df.columns
    scaling_cols = [x for x in df.columns if x in scaling_cols]
    df_train_values_for_scaling = df[scaling_cols].iloc[train_normal_indexes].reset_index(drop=True)

    print(f"      - Train normal: {len(train_normal_indexes)}")
    print(f"      - Val normal: {len(val_normal_indexes)}")
    if len(anomalous_window_indexes) > 0:
        print(f"      - Anomalous window indices: {len(anomalous_window_indexes)}")

    # ✅ Return anomaly data as dict
    anomalous_data = {
        'window_start_indices': anomaly_window_start_indices,  # [M] start indices
        'labels': anomaly_window_labels,  # [M, L] binary labels
        'all_indices_set': anomalous_window_indexes  # set for filtering
    }

    return train_normal_indexes, val_normal_indexes, df_train_values_for_scaling, anomalous_data


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


# ✅ Global cache to prevent GC (module-level)
_RAY_OBJECT_CACHE = {}


# ============================================================
# MODIFIED: prepare_shared_configuration
# ============================================================

def parse_metadata_file(metadata_path):
    """
    Parse metadata.txt file and extract all information.
    """
    from pathlib import Path
    import json

    metadata_path = Path(metadata_path)

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    metadata = {}

    with open(metadata_path, 'r') as f:
        for line in f:
            line = line.strip()
            if ':' not in line:
                continue

            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()

            # Parse based on key
            if key == 'Experiment':
                metadata['exp_name'] = value

            elif key == 'Model':
                metadata['model_name'] = value

            elif key == 'Dataset':
                metadata['dataset_name'] = value

            elif key == 'Seed':
                metadata['seed'] = int(value)

            elif key == 'Filter anomalies':
                metadata['filter_anomalies'] = value.lower() == 'true'

            elif key == 'Sequence length':
                metadata['seq_len'] = int(value)

            elif key == 'Train indices':
                # Remove commas and convert to int
                metadata['train_size'] = int(value.replace(',', ''))

            elif key == 'Val indices':
                metadata['val_size'] = int(value.replace(',', ''))

            elif key == 'Scaler':
                metadata['scaler_type'] = value

            elif key == 'Scaler params':
                # ✅ Use JSON instead of eval (safer and handles multi-line)
                try:
                    metadata['scaler_params'] = json.loads(value)
                except json.JSONDecodeError as e:
                    print(f"   ⚠️  Could not parse scaler params as JSON: {e}")
                    # Try eval as fallback (for old format)
                    try:
                        metadata['scaler_params'] = eval(value)
                    except Exception as e2:
                        print(f"   ⚠️  Could not parse scaler params with eval: {e2}")
                        metadata['scaler_params'] = {'raw': value}

            elif key == 'Feature columns':
                # Parse comma-separated list
                metadata['feature_columns'] = [f.strip() for f in value.split(',')]

    # Validate required fields
    required = ['feature_columns']
    missing = [req for req in required if req not in metadata]

    if missing:
        raise ValueError(f"Required fields missing from metadata: {missing}")

    return metadata

def prepare_shared_configuration(cfg):
    """
    Prepare shared configuration by saving indices to disk.

    - If fine_tuning=1 AND paths exist: REUSE existing files (read from metadata.txt)
    - Otherwise: REGENERATE (delete old if exists)
    """
    import torch
    import os
    import numpy as np
    from pathlib import Path
    import pickle
    import shutil

    print("\n" + "=" * 80)
    print("📦 PREPARING SHARED CONFIGURATION (Disk-based)")
    print("=" * 80)

    # Extract config parameters
    seed = cfg.opt.get('seed', 42)
    filter_anomalies = cfg.opt.get('filter_anomalies', False)
    is_anomaly_column = cfg.dataset.get('is_anomaly_column', None)
    dataset_path = cfg.dataset.data_path
    seq_len = cfg.dataset.seq_in_length

    # Build experiment identifier
    model_name = cfg.model.get('name', 'unknown_model')
    dataset_name = cfg.dataset.get('name', 'dataset')
    exp_name = cfg.opt.get('exp_name', 'experiment')

    # ============================================================
    # CHECK FINE-TUNING MODE
    # ============================================================
    try:
        tune_str = cfg.tune_config.get('opt.fine_tuning', 'tune.choice([0])')
        is_fine_tuning = get_tune_value(tune_str, default=0)
    except:
        is_fine_tuning = cfg.opt.get('fine_tuning', 0)

    print(f"\n🔍 Mode: {'FINE-TUNING' if is_fine_tuning else 'TRAINING FROM SCRATCH'}")

    # ============================================================
    # Try to reuse existing files
    # ============================================================
    shared_config = check_existing_file(cfg, dataset_path, seq_len)
    if shared_config:
        return shared_config

    # ============================================================
    # TRAINING FROM SCRATCH MODE: Regenerate everything
    # ============================================================

    print(f"\n   🔄 REGENERATING all files (training from scratch)")

    # Add filter_anomalies suffix
    filter_anomalies_suffix = "filtered" if filter_anomalies else "unfiltered"

    # Base directory for ALL indices
    base_indices_dir = Path('./train_val_indices/')
    base_indices_dir.mkdir(exist_ok=True)

    # Experiment-specific subdirectory with filter suffix
    exp_subdir_name = f"{model_name}_{exp_name}_{dataset_name}_seed{seed}_{filter_anomalies_suffix}"
    exp_indices_dir = base_indices_dir / exp_subdir_name

    # Delete existing directory if present
    if exp_indices_dir.exists():
        print(f"\n   ⚠️  Experiment directory already exists: {exp_indices_dir}")
        print(f"   🗑️  Deleting existing directory...")
        try:
            shutil.rmtree(exp_indices_dir)
            print(f"   ✓ Deleted successfully")
        except Exception as e:
            print(f"   ❌ ERROR deleting directory: {e}")
            raise

    # Create fresh directory
    exp_indices_dir.mkdir(exist_ok=True)
    print(f"   ✓ Created fresh experiment directory: {exp_indices_dir}")

    # Define file paths
    indices_filename = 'train_val_indices.npz'
    scaler_filename = 'scaler.pkl'
    metadata_filename = 'metadata.txt'

    indices_path = exp_indices_dir / indices_filename
    scaler_path = exp_indices_dir / scaler_filename
    metadata_path = exp_indices_dir / metadata_filename

    # 1. Load data
    print("\n1️⃣ Loading dataset...")
    df = load_and_preprocess_dataframe(cfg)
    feature_columns = cfg.dataset.feats
    print(f"   ✓ Loaded: {df.shape}")
    print(f"   ✓ Features: {len(feature_columns)}")

    # 2. Split indices (NO OVERLAP - base indices only)
    print("\n2️⃣ Splitting train/val indices (no overlap)...")
    train_indexes, val_indexes, train_df_for_scaling, anomalous_data = (
        create_train_val_df_indexes(
            cfg=cfg,
            df=df,
            return_anomalies=filter_anomalies,
            ano_col=is_anomaly_column,
            seed=seed
        )
    )

    print(f"   ✓ Train base indices: {len(train_indexes):,}")
    print(f"   ✓ Val base indices: {len(val_indexes):,}")

    # Extract anomaly info if available
    anomaly_window_indices = None
    anomaly_window_labels = None

    if anomalous_data is not None:
        anomaly_window_indices = anomalous_data['window_start_indices']
        anomaly_window_labels = anomalous_data['labels']
        print(f"   ✓ Anomaly windows: {len(anomaly_window_indices)}")
        if len(anomaly_window_indices) > 0:
            avg_density = anomaly_window_labels.mean()
            print(f"   ✓ Average anomaly density: {avg_density:.2%}")

    # 3. Fit scaler on CLEAN train data
    print("\n3️⃣ Fitting scaler on CLEAN train data...")
    scaler, df_scaled, scaler_params = get_scaler(
        cfg=cfg,
        df_fit=train_df_for_scaling,
        df_transform=df
    )
    print(f"   ✓ Scaler fitted: {scaler.__class__.__name__}")
    print(f"   ✓ Scaler params: {scaler_params}")

    # 4. Generate and save metric dataset
    print("\n4️⃣ Generating metric dataset...")
    metric_dataset_path = None
    anomaly_strategy = cfg.opt.get('anomaly_strategy', 'none')

    if anomaly_strategy != 'none':
        print(f"   - Strategy: {anomaly_strategy}")

        # Apply overlap for metric dataset
        metric_seq_overlap = cfg.opt.get('metric_seq_overlap', 0)

        from trainer.utils import compute_indices_with_overlap
        val_indices_for_metric = compute_indices_with_overlap(
            base_indices=val_indexes,
            overlap=metric_seq_overlap,
            seq_len=seq_len
        )

        print(f"   ✓ Metric overlap: {metric_seq_overlap}")
        print(f"   ✓ Val indices for metric: {len(val_indices_for_metric)} (from {len(val_indexes)} base)")

        # Extract clean sequences
        print(f"   - Extracting clean sequences for metric dataset...")
        clean_sequences = extract_sequences_from_indices(
            df=df_scaled,
            indices=val_indices_for_metric,
            seq_len=seq_len,
            feature_columns=feature_columns
        )

        print(f"   ✓ Clean sequences extracted: {clean_sequences.shape}")

        # Extract original anomaly sequences if needed
        original_anomaly_sequences = None
        original_anomaly_labels = None

        if anomaly_strategy in ['use_original', 'both']:
            if anomaly_window_indices is not None and len(anomaly_window_indices) > 0:
                print(f"   - Extracting original anomaly sequences...")

                anomalous_indices_for_metric = compute_indices_with_overlap(
                    base_indices=anomaly_window_indices,
                    overlap=metric_seq_overlap,
                    seq_len=seq_len
                )

                print(f"   ✓ Anomalous indices for metric: {len(anomalous_indices_for_metric)}")

                original_anomaly_sequences = extract_sequences_from_indices(
                    df=df_scaled,
                    indices=anomalous_indices_for_metric,
                    seq_len=seq_len,
                    feature_columns=feature_columns,
                )

                # Extract labels
                if is_anomaly_column and is_anomaly_column in df.columns:
                    original_anomaly_labels = []
                    for idx in anomalous_indices_for_metric:
                        if idx + seq_len <= len(df):
                            label_seq = df[is_anomaly_column].iloc[idx:idx + seq_len].values
                            original_anomaly_labels.append(label_seq)

                    original_anomaly_labels = np.array(original_anomaly_labels)
                    original_anomaly_labels = original_anomaly_labels[:, np.newaxis, :]

                    print(f"   ✓ Original anomaly sequences: {original_anomaly_sequences.shape}")
                    print(f"   ✓ Original anomaly labels: {original_anomaly_labels.shape}")
                else:
                    print(f"   ⚠️  Anomaly column '{is_anomaly_column}' not found in DataFrame!")
                    original_anomaly_sequences = None
                    original_anomaly_labels = None
            else:
                print(f"   ⚠️  No anomalous indices found! Cannot use '{anomaly_strategy}' strategy")
                print(f"       Falling back to 'corrupt_validation' strategy")
                anomaly_strategy = 'corrupt_validation'

        # Generate metric dataset
        metric_dataset_path = generate_and_save_metric_dataset(
            cfg=cfg,
            clean_sequences=clean_sequences,
            feature_columns=feature_columns,
            scaler=scaler,
            original_anomaly_sequences=original_anomaly_sequences,
            original_anomaly_labels=original_anomaly_labels,
            output_dir='./metric_datasets/',
            force_regenerate=cfg.opt.get('force_regenerate', False),
            plot_samples=cfg.opt.get('plot_metric_samples', False),
            plot_percentage=cfg.opt.get('plot_metric_percentage', 0.05)
        )

        if metric_dataset_path and os.path.exists(metric_dataset_path):
            saved_dict = torch.load(metric_dataset_path, map_location='cpu')
            is_standardized = saved_dict['metadata'].get('is_standardized', True)

            print(f"   ✓ Metric dataset saved: {metric_dataset_path}")
            print(f"   ✓ Standardized: {is_standardized}")
            print(f"   ✓ Sequences: {len(saved_dict['dataset'])}")
        else:
            print(f"   ⚠️  Metric dataset generation failed or disabled")
    else:
        print(f"   - Anomaly strategy is 'none' - skipping metric dataset")

    # 5. Save indices and scaler to disk
    print("\n5️⃣ Saving indices and scaler to disk...")

    # Save indices (compressed NPZ)
    np.savez_compressed(
        indices_path,
        train_indices=train_indexes,
        val_indices=val_indexes
    )

    indices_size = os.path.getsize(indices_path) / (1024 * 1024)  # MB
    print(f"   ✓ Indices saved: {indices_path}")
    print(f"   ✓ File size: {indices_size:.2f} MB")
    print(f"   ✓ Train indices: {len(train_indexes):,}")
    print(f"   ✓ Val indices: {len(val_indexes):,}")

    # Save scaler
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    scaler_size = os.path.getsize(scaler_path) / 1024  # KB
    print(f"   ✓ Scaler saved: {scaler_path}")
    print(f"   ✓ File size: {scaler_size:.2f} KB")

    # Save metadata for reference
    with open(metadata_path, 'w') as f:
        f.write(f"Experiment: {exp_name}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Filter anomalies: {filter_anomalies}\n")
        f.write(f"Sequence length: {seq_len}\n")
        f.write(f"Train indices: {len(train_indexes):,}\n")
        f.write(f"Val indices: {len(val_indexes):,}\n")
        f.write(f"Scaler: {scaler.__class__.__name__}\n")

        # ✅ Use JSON for scaler_params (single line, no indent issues)
        f.write(f"Scaler params: {json.dumps(scaler_params)}\n")

        f.write(f"Feature columns: {', '.join(feature_columns)}\n")
    print(f"   ✓ Metadata saved: {metadata_path}")

    # 6. Build shared config
    shared_config = {
        # Paths to experiment-specific directory
        'indices_path': str(indices_path),
        'scaler_path': str(scaler_path),
        'experiment_dir': str(exp_indices_dir),

        # Dataset path
        'dataset_path': str(dataset_path),
        'metric_loader_path': metric_dataset_path,

        # Small metadata
        'scaler_params': scaler_params,
        'feature_columns': list(feature_columns),
        'seq_len': int(seq_len),

        # Statistics
        'train_size': int(len(train_indexes)),
        'val_size': int(len(val_indexes)),

        # Experiment identifier
        'exp_identifier': exp_subdir_name,
    }

    print("\n" + "=" * 80)
    print("✅ SHARED CONFIGURATION READY (REGENERATED)")
    print("=" * 80)
    print(f"   Mode: Training from scratch (regenerated all files)")
    print(f"   - Experiment directory: {exp_indices_dir}")
    print(f"   - Indices: {len(train_indexes):,} train, {len(val_indexes):,} val")
    print(f"   - Scaler: {scaler.__class__.__name__}")
    if metric_dataset_path:
        print(f"   - Metric dataset: {metric_dataset_path}")
    print(f"   - File sizes: {indices_size:.2f} MB + {scaler_size:.2f} KB")
    print(f"   ⚠️  All trials will read from: {exp_indices_dir}")
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


def extract_sequences_from_indices(df, indices, seq_len, feature_columns):
    """
    Extract sequences from DataFrame at given indices.

    Args:
        df: DataFrame (or already converted to numpy)
        indices: Start indices for sequences (ALREADY with overlap applied!)
        seq_len: Sequence length
        feature_columns: Feature column names

    Returns:
        sequences: [N, L, F] numpy array
    """
    import numpy as np

    # Convert to NumPy ONCE (if not already)
    if hasattr(df, 'values'):
        data = df[feature_columns].values  # [T, F]
    else:
        data = df  # Already numpy

    # Filter invalid indices (beyond data length)
    max_idx = len(data) - seq_len
    valid_indices = indices[indices <= max_idx]

    # ✅ NO downsampling here - indices already have overlap applied!
    # Vectorized extraction
    idx_array = valid_indices[:, None] + np.arange(seq_len)[None, :]  # [N, seq_len]
    sequences = data[idx_array]  # [N, seq_len, F]

    return sequences.astype(np.float32)


def get_transform(cfg):
    # Define the dataset name to apply specific transformations

    if cfg.model.name == "conv_ae1D":
        transform = T.Compose([
            T.ToTensor(),
            Lambda(lambda x: x.permute((0, 2, 1))),
            Lambda(lambda x: x.squeeze(0))])
    elif cfg.model.name == 'conv_ae2D':
        transform = T.Compose([
            T.ToTensor(),
            Lambda(lambda x: x.permute((0, 2, 1)))])
    else:
        transform = T.Compose([
            T.ToTensor(),
            Lambda(lambda x: x.squeeze(0))  # Removes channel dim added by ToTensor
        ])

    return transform


def create_dataloaders(cfg, train_dataset, val_dataset, train_sampler, val_sampler):
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

    print(f' USING N {cfg.resources.cpu_trial} Workers per trial')

    trainloader = DataLoader(
        train_dataset,
        batch_size=cfg.opt.batch_size,
        num_workers= cfg.resources.get('cpu_trial', 0),
        sampler=train_sampler
        #pin_memory=True if cfg.resources.gpu_trial else False
    )

    valloader = DataLoader(
        val_dataset,
        batch_size=cfg.opt.batch_size,
        num_workers= cfg.resources.get('cpu_trial', 0),
        sampler=val_sampler
        #pin_memory=True if cfg.resources.gpu_trial else False
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


def create_datasets_from_splits(df_scaled, splits, cfg, transform=None):
    """
    Universal dataset creation from arbitrary number of index splits.

    Creates a Dataset_seq_df for each split provided.

    Args:
        df_scaled: Scaled DataFrame
        splits: Dict mapping split names to config dicts with 'indices' and 'shuffle'
                Example: {
                    'train': {'indices': train_idx, 'shuffle': True},
                    'val': {'indices': val_idx, 'shuffle': False}
                }
        cfg: Configuration object
        transform: Optional transform override (if None, uses get_transform(cfg))

    Returns:
        datasets: Dict mapping split names to Dataset_seq_df instances
        samplers: Dict mapping split names to samplers

    Example:
        >>> splits = {
        ...     'train': {'indices': train_indices, 'shuffle': True},
        ...     'val': {'indices': val_indices, 'shuffle': False}
        ... }
        >>> datasets, samplers = create_datasets_from_splits(
        ...     df_scaled=df_scaled,
        ...     splits=splits,
        ...     cfg=cfg
        ... )
        >>> train_dataset = datasets['train']
        >>> val_dataset = datasets['val']
    """
    from dataset.sentinel import Dataset_seq_df
    from utils.load_dataset import get_transform

    if not isinstance(splits, dict) or len(splits) == 0:
        raise ValueError("splits must be a non-empty dict")

    # Validate splits format
    for split_name, split_config in splits.items():
        if not isinstance(split_config, dict):
            raise ValueError(
                f"Invalid format for split '{split_name}': "
                f"expected dict with 'indices' and 'shuffle', got {type(split_config)}"
            )
        if 'indices' not in split_config:
            raise ValueError(
                f"Missing 'indices' key in split '{split_name}'"
            )

    # Get transform if not provided
    if transform is None:
        transform = get_transform(cfg)

    # Create samplers for all splits
    samplers = get_samplers_from_index_sets(splits, seed=cfg.opt.get('seed', 42))

    # Dataset arguments (common for all splits)
    dataset_args = dict(
        df=df_scaled,
        target=cfg.dataset.target,
        sequence_length=cfg.dataset.seq_in_length,
        out_window=cfg.dataset.seq_out_length,
        forecast=cfg.dataset.forecast,
        remove_target=cfg.dataset.remove_target,
        transform=transform,
        is_anomaly_column=cfg.dataset.get('is_anomaly_column', None)
    )

    # Create datasets for all splits
    datasets = {}
    for split_name, sampler in samplers.items():
        datasets[split_name] = Dataset_seq_df(
            **dataset_args,
            sampler=sampler,
            indices=sampler.indices
        )

    # Print summary
    print(f"\n   ✓ Datasets created (on-the-fly extraction):")
    for split_name, dataset in datasets.items():
        shuffle_status = "shuffled" if splits[split_name].get('shuffle', False) else "sequential"
        print(f"      - {split_name}: {len(dataset)} sequences ({shuffle_status})")

    return datasets, samplers


def get_samplers_from_index_sets(index_sets: dict, seed=42):
    """
    Creates SubsetRandomSamplers for multiple index lists with per-split shuffle control.

    Args:
        index_sets (dict): Dictionary where keys are split names (e.g., 'train', 'val')
                          and values are dicts with:
                          - 'indices': list/array of indices (REQUIRED)
                          - 'shuffle': bool (whether to shuffle this split, default: False)
        seed (int): Random seed for reproducibility (default: 42)

    Returns:
        dict: Dictionary with same keys, values are SubsetRandomSampler objects.

    Example:
        >>> index_sets = {
        ...     'train': {'indices': [0,1,2,...,1000], 'shuffle': True},
        ...     'val': {'indices': [1001,1002,...,1200], 'shuffle': False}
        ... }
        >>> samplers = get_samplers_from_index_sets(index_sets, seed=42)
        >>> train_sampler = samplers['train']  # Shuffled indices
        >>> val_sampler = samplers['val']      # Sequential indices
    """
    import numpy as np
    from torch.utils.data import SubsetRandomSampler

    np.random.seed(seed)
    samplers = {}

    for key, config in index_sets.items():
        # Validate config format
        if not isinstance(config, dict):
            raise ValueError(
                f"Invalid format for '{key}': expected dict with 'indices' and 'shuffle', "
                f"got {type(config)}"
            )

        # Extract indices (required)
        if 'indices' not in config:
            raise ValueError(f"Missing required 'indices' key for split '{key}'")

        index_list = config['indices']

        # Extract shuffle flag (optional, default False)
        shuffle = config.get('shuffle', False)

        # Convert to numpy array
        idxs = np.array(sorted(index_list))

        # Shuffle if requested
        if shuffle:
            np.random.shuffle(idxs)

        # Create sampler
        samplers[key] = SubsetRandomSampler(idxs)

    return samplers


def apply_step_get_samplers_from_index_sets(index_sets: dict, shuffle=False, seed=101):
    """
    Creates SubsetRandomSamplers for multiple index lists, each with a specified step.

    Args:
        index_sets (dict): A dictionary where keys are names (e.g., 'train', 'val') and
                           values are tuples of (index_list, step_size).
        shuffle (bool): Whether to shuffle indices for each sampler.

    Returns:
        dict: A dictionary with the same keys as index_sets, where values are SubsetRandomSampler objects.
    """
    np.random.seed(seed)
    samplers = {}

    for key, (index_list, step, shuffle) in index_sets.items():
        index_list = np.array(sorted(index_list))
        idxs = index_list[::step] if step > 0 else index_list
        if shuffle:
            np.random.shuffle(idxs)
        samplers[key] = SubsetRandomSampler(idxs)

    return samplers


def check_existing_file(cfg, dataset_path, seq_len):
    metric_dataset_path = cfg.dataset.get('metric_dataset_path', None)
    train_val_indices_path = cfg.dataset.get('train_val_indices_path', None)

    print(f"\n   Checking for existing files...")
    print(f"   - metric_dataset_path: {metric_dataset_path}")
    print(f"   - train_val_indices_path: {train_val_indices_path}")

    # Validate paths exist
    if (metric_dataset_path and os.path.exists(metric_dataset_path) and
            train_val_indices_path and os.path.exists(train_val_indices_path)):

        print(f"\n   ✅ REUSING existing files")

        # ========================================================
        # FIND FILES IN DIRECTORY
        # ========================================================

        indices_dir = Path(train_val_indices_path)
        if indices_dir.is_file():
            indices_dir = indices_dir.parent

        # Find files
        train_val_indices_file = None
        scaler_file = None
        metadata_file = None

        for f in indices_dir.iterdir():
            if 'train_val' in f.name.lower() and f.suffix in ['.npz', '.npy']:
                train_val_indices_file = f
            elif 'scaler' in f.name.lower() and f.suffix == '.pkl':
                scaler_file = f
            elif 'metadata' in f.name.lower() and f.suffix == '.txt':
                metadata_file = f

        # Validate required files exist
        if train_val_indices_file is None or not train_val_indices_file.exists():
            raise FileNotFoundError(f"❌ train_val_indices file not found in {indices_dir}")

        if scaler_file is None or not scaler_file.exists():
            raise FileNotFoundError(f"❌ scaler.pkl not found in {indices_dir}")

        # ========================================================
        # LOAD METADATA (Primary source of truth)
        # ========================================================

        if metadata_file and metadata_file.exists():
            print(f"\n   📂 Loading metadata from: {metadata_file}")
            metadata = parse_metadata_file(metadata_file)

            print(f"   ✓ Experiment: {metadata.get('exp_name')}")
            print(f"   ✓ Model: {metadata.get('model_name')}")
            print(f"   ✓ Dataset: {metadata.get('dataset_name')}")
            print(f"   ✓ Seed: {metadata.get('seed')}")
            print(f"   ✓ Features: {len(metadata.get('feature_columns', []))}")

            # Extract key info from metadata
            feature_columns = metadata.get('feature_columns')
            scaler_params = metadata.get('scaler_params')
            seq_len_from_meta = metadata.get('seq_len')
            train_size = metadata.get('train_size')
            val_size = metadata.get('val_size')

            if feature_columns is None:
                raise ValueError("❌ Feature columns not found in metadata!")

            # Use seq_len from metadata if available, else from config
            if seq_len_from_meta:
                seq_len = seq_len_from_meta
                print(f"   ✓ Sequence length (from metadata): {seq_len}")
            else:
                print(f"   ⚠️  Sequence length not in metadata, using config: {seq_len}")

        else:
            print(f"\n   ⚠️  Metadata file not found!")
            print(f"   ⚠️  Falling back to config (not recommended)")

            # Fallback to config
            feature_columns = cfg.dataset.feats
            scaler_params = None
            train_size = None
            val_size = None

        # ========================================================
        # LOAD INDICES
        # ========================================================

        print(f"\n   📂 Loading indices from: {train_val_indices_file}")
        indices_data = np.load(train_val_indices_file)
        train_indexes = indices_data['train_indices']
        val_indexes = indices_data['val_indices']

        # Use loaded sizes if not from metadata
        if train_size is None:
            train_size = len(train_indexes)
        if val_size is None:
            val_size = len(val_indexes)

        print(f"   ✓ Train indices: {train_size:,}")
        print(f"   ✓ Val indices: {val_size:,}")

        # ========================================================
        # LOAD SCALER
        # ========================================================

        print(f"\n   📂 Loading scaler from: {scaler_file}")
        with open(scaler_file, 'rb') as f:
            scaler = pickle.load(f)

        print(f"   ✓ Scaler: {scaler.__class__.__name__}")

        # If scaler_params not from metadata, extract from scaler
        if scaler_params is None:
            if hasattr(scaler, 'mean_'):
                scaler_params = {
                    'mean': scaler.mean_.tolist() if hasattr(scaler.mean_, 'tolist') else None,
                    'std': scaler.scale_.tolist() if hasattr(scaler, 'scale_') else None
                }
            else:
                scaler_params = {'type': scaler.__class__.__name__}

        # Validate feature count
        if len(feature_columns) != scaler.n_features_in_:
            raise ValueError(
                f"Feature mismatch! Metadata has {len(feature_columns)} "
                f"but scaler expects {scaler.n_features_in_}"
            )

        # Get experiment directory
        exp_indices_dir = train_val_indices_file.parent

        print(f"\n   ✓ Experiment directory: {exp_indices_dir}")

        # ========================================================
        # BUILD SHARED CONFIG (from loaded data)
        # ========================================================
        shared_config = {
            # Paths to existing files
            'indices_path': str(train_val_indices_file),
            'scaler_path': str(scaler_file),
            'experiment_dir': str(exp_indices_dir),

            # Dataset path
            'dataset_path': str(dataset_path),
            'metric_loader_path': str(metric_dataset_path),

            # Metadata (from metadata.txt)
            'scaler_params': scaler_params,
            'feature_columns': list(feature_columns),
            'seq_len': int(seq_len),

            # Statistics (from metadata.txt or loaded data)
            'train_size': int(train_size),
            'val_size': int(val_size),

            # Experiment identifier
            'exp_identifier': exp_indices_dir.name,
        }

        print("\n" + "=" * 80)
        print("✅ SHARED CONFIGURATION READY (REUSED)")
        print("=" * 80)
        print(f"   Mode: Fine-tuning (reusing existing files)")
        print(f"   - Experiment directory: {exp_indices_dir}")
        print(f"   - Indices: {train_size:,} train, {val_size:,} val")
        print(f"   - Scaler: {scaler.__class__.__name__}")
        print(f"   - Features: {len(feature_columns)}")
        print(f"   - Sequence length: {seq_len}")
        print(f"   - Metric dataset: {metric_dataset_path}")
        print(f"   ✓ All trials will read from: {exp_indices_dir}")
        print("=" * 80)

        return shared_config

    else:
        print(f"\n   ⚠️  Existing files not found or incomplete!")
        print(f"       Falling back to REGENERATE mode")
        is_fine_tuning = False  # Fallback to regenerate

        return False