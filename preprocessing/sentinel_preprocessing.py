import pandas as pd
from omegaconf import ListConfig
from sklearn.preprocessing import StandardScaler, RobustScaler
from dataset.sentinel import Dataset_seq
from torch.utils.data import SubsetRandomSampler, DataLoader
import numpy as np
from scipy import interpolate

import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg' if you have PyQt5 installed
import matplotlib.pyplot as plt

def extract_sequences(df, indices, seq_len, feature_cols):
    sequences = []
    for idx in indices:
        window = df.iloc[idx:idx+seq_len][feature_cols].values
        sequences.append(window)
    return np.stack(sequences)  # shape: (num_seqs, seq_len, num_features)

def get_samplers_from_index_sets(index_sets: dict, shuffle=False, seed=101):
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

def get_scaler(cfg, df_fit=None, df_transform=None):
    scaler_cfg = cfg['dataset'].get('scaler', None)
    if not scaler_cfg:
        return None, df_transform, None  # No scaler, return original df_transform and None for params

    columns_fit = df_fit.columns.difference([cfg.dataset.is_anomaly_column]) if cfg.dataset.is_anomaly_column in df_fit.columns else df_fit.columns
    df_fit = df_fit[columns_fit]

    if df_transform is not None:
        columns_transform = df_transform.columns.difference([cfg.dataset.is_anomaly_column]) if cfg.dataset.is_anomaly_column in df_transform.columns else df_transform.columns
        df_transform = df_transform[columns_transform]
        assert df_fit.columns.equals(df_transform.columns), "DataFrames for fitting and transforming must have the same columns."

    # Extract scaler name and params
    scaler_name = scaler_cfg.split('-')[0]

    if scaler_name == 'StandardScaler':
        scaler = StandardScaler()
    elif scaler_name == 'RobustScaler':
        # Translate config keys to sklearn-compatible ones
        q1 = float(scaler_cfg.split('-')[1])
        q2 = float(scaler_cfg.split('-')[2])
        scaler = RobustScaler(quantile_range=(q1 * 100, q2 * 100))
    else:
        raise ValueError(f"Scaler '{scaler_name}' not supported.")

    if df_fit is not None:
        # Fit only on clean training data
        scaler.fit(df_fit)
    else:
        df_scaled = None
        scaler_params = None
        return scaler, df_scaled, scaler_params    # df_scaled=None, scaler_params=None

    if df_fit is not None and df_transform is not None:
        df_scaled = pd.DataFrame(scaler.transform(df_transform), columns=df_transform.columns)
        scaler_params = serialize_scaler(scaler)
        return scaler, df_scaled, scaler_params
    else:
        df_scaled = None
        scaler_params = None
        return scaler, df_scaled, scaler_params  # df_scaled=None, scaler_params=None


def serialize_scaler(scaler):
    if isinstance(scaler, StandardScaler):
        return {
            'type': 'StandardScaler',
            'mean_': scaler.mean_.tolist(),
            'scale_': scaler.scale_.tolist(),
            'var_': scaler.var_.tolist(),
            'n_samples_seen_': int(scaler.n_samples_seen_)}

    elif isinstance(scaler, RobustScaler):
        return {
            'type': 'RobustScaler',
            'center_': scaler.center_.tolist(),
            'scale_': scaler.scale_.tolist(),
            'quantile_range': scaler.quantile_range}

    else:
        raise ValueError(f"Cannot serialize unknown scaler type: {type(scaler)}")

def deserialize_scaler(scaler_params):
    scaler_type = scaler_params['type']

    if scaler_type == 'StandardScaler':
        scaler = StandardScaler()
        scaler.mean_ = np.array(scaler_params['mean_'])
        scaler.scale_ = np.array(scaler_params['scale_'])
        scaler.var_ = np.array(scaler_params['var_'])
        return scaler

    elif scaler_type == 'RobustScaler':
        scaler = RobustScaler()
        scaler.center_ = np.array(scaler_params['center_'])
        scaler.scale_ = np.array(scaler_params['scale_'])
        scaler.quantile_range = tuple(scaler_params['quantile_range'])
        return scaler

    else:
        raise ValueError(f"Cannot deserialize unknown scaler type: {scaler_type}")

def create_train_val_df_indexes(
    cfg,
    df,
    return_anomalies=False,
    ano_col="ia_anomaly",
    seed=42,
    min_chunk_size=None,
    max_chunks=200,
    tolerance=0.01,
):
    """
     Split a dataframe into training/validation sets by chunk sampling,
     ensuring the validation ratio is near the target and that each chunk
     is at least `min_chunk_size` samples long.
     """

    def valid_chunk_splits(
            total_len,
            val_ratio=0.2,
            max_chunks=100,
            min_chunk_size=None,
            tolerance=0.01,
    ):
        """
        Trova la configurazione di chunking che:
        - rispetta la dimensione minima del chunk
        - produce un rapporto di validation vicino al target entro la tolleranza
        - massimizza il numero di chunk (validation più sparsa)
        """

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

                # Mantieni la soluzione con il numero massimo di chunk validi
                if (
                        best_solution is None
                        or candidate["num_chunks"] > best_solution["num_chunks"]
                ):
                    best_solution = candidate

        if best_solution is None:
            raise ValueError(
                "Nessuna configurazione valida trovata — aumenta max_chunks o riduci min_chunk_size/tolerance."
            )

        return best_solution

    # ---------------------------------------------------------------------
    # Prepare configuration
    seq_len = cfg.dataset.seq_in_length
    val_ratio = 1 - cfg.dataset.train_val_split
    np.random.seed(seed)
    df = df.reset_index(drop=True)
    total_len = len(df)

    min_chunk_size = cfg.dataset.seq_in_length*cfg.dataset.seq_in_length_into_chunk if min_chunk_size is None else min_chunk_size

    # Find a valid chunking configuration
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

    # ---------------------------------------------------------------------
    # Randomly choose validation chunks
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

    # ---------------------------------------------------------------------
    # If anomalies not needed, return here
    if not return_anomalies:
        df_train_values_for_scaling = df.iloc[train_indexes].reset_index(drop=True)
        return train_indexes, val_indexes, df_train_values_for_scaling, None

    # ---------------------------------------------------------------------
    # Handle anomaly-based indexing
    full_anomalous_idx = df[df[ano_col] == 1].index.to_numpy()

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

    train_anomalous_windows = get_anomaly_window_indexes(train_anomalous_indexes, total_len)
    val_anomalous_windows = get_anomaly_window_indexes(val_anomalous_indexes, total_len)

    train_normal_indexes = np.setdiff1d(train_indexes, list(train_anomalous_windows), assume_unique=True)
    val_normal_indexes = np.setdiff1d(val_indexes, list(val_anomalous_windows), assume_unique=True)

    anomalous_window_indexes = list(get_anomaly_window_indexes(full_anomalous_idx, total_len))

    scaling_cols = df.columns.difference([ano_col]) if ano_col in df.columns else df.columns
    df_train_values_for_scaling = df[scaling_cols].iloc[train_normal_indexes].reset_index(drop=True)

    return train_normal_indexes, val_normal_indexes, df_train_values_for_scaling, anomalous_window_indexes

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
    Upsample an entire DataFrame column-wise and optionally apply
    slight augmentation *only* to interpolated points.

    If `ano_col` is provided (binary 0/1 anomaly label or list of them),
    those columns are *not interpolated* but propagated step-wise:
      - propagate_mode='forward': fill interpolated points with the
        anomaly value of the *start* of each interval.
      - propagate_mode='backward': fill interpolated points with the
        anomaly value of the *end* of each interval.

    Handles datetime indexes (with or without timezone) and numeric indexes.

    Parameters
    ----------
    df_low : pd.DataFrame
        Time-indexed DataFrame (datetime or numeric index).
    factor : int
        Upsampling factor (e.g., 4 = 4x more points).
    method : str
        Interpolation method ('linear', 'cubic', ...).
    do_augmentation : bool
        Whether to add small noise/scaling to interpolated points.
    jitter_std : float
        Noise amplitude relative to each column's std.
    scale_range : tuple
        Random scaling range for interpolated points.
    ano_col : str or list, optional
        Name or list of binary anomaly columns to propagate without interpolation.
    propagate_mode : str, default='forward'
        'forward' → use the first value of each interval,
        'backward' → use the last value of each interval.

    Returns
    -------
    pd.DataFrame
        Upsampled (and optionally augmented) DataFrame.
    """

    if factor <= 1:
        return df_low.copy()

    df_low = df_low.sort_index()
    n = len(df_low)

    # --- Normalize ano_col to list
    if ano_col is None:
        ano_cols = []
    elif isinstance(ano_col, str):
        ano_cols = [ano_col]
    else:
        ano_cols = list(ano_col)

    # --- Convert index to numeric (seconds or raw)
    if isinstance(df_low.index, pd.DatetimeIndex):
        idx = pd.DatetimeIndex(df_low.index)
        if idx.tz is not None:
            idx = idx.tz_convert(None)
        x_low = idx.view(np.int64) / 1e9  # seconds
        idx_type = "datetime"
    else:
        x_low = df_low.index.to_numpy(dtype=float)
        idx_type = "numeric"

    # --- Build new timeline
    x_min, x_max = x_low[0], x_low[-1]
    n_new = (n - 1) * factor + 1
    x_high = np.linspace(x_min, x_max, n_new)

    # --- Identify original vs interpolated points
    #tol = (x_high[1] - x_high[0]) / 10
    #is_original = np.isclose(x_high[:, None], x_low[None, :], atol=tol).any(axis=1)

    upsampled_cols = {}

    for col in df_low.columns:
        y_low = df_low[col].to_numpy()

        # Handle anomaly columns separately
        if col in ano_cols:
            if propagate_mode == "backward":
                y_high = np.repeat(y_low, factor)
                y_high = np.append(y_low[0], y_high[:-1])  # shift backward
            else:  # forward (default)
                y_high = np.repeat(y_low, factor)
                y_high = np.append(y_high, y_low[-1])
            y_high = y_high[:n_new]
            upsampled_cols[col] = y_high.astype(int)
            continue

        # --- Regular interpolation for numeric columns
        col_method = "linear" if (method == "cubic" and len(df_low) < 4) else method
        f = interpolate.interp1d(
            x_low, y_low, kind=col_method, bounds_error=False, fill_value="extrapolate"
        )
        y_high = f(x_high)

        # --- Augmentation only on interpolated points
        if do_augmentation:
            std = np.nanstd(y_low)
            if std == 0 or np.isnan(std):
                std = 1.0
            noise = np.random.normal(0, jitter_std * std, len(y_high))
            scale = np.random.uniform(scale_range[0], scale_range[1])
            #mask = ~is_original
            #y_high[mask] = y_high[mask] * scale + noise[mask]
            y_high = y_high * scale + noise

        upsampled_cols[col] = y_high

    # --- Build output index
    if idx_type == "datetime":
        new_index = pd.to_datetime(x_high, unit="s")
    else:
        new_index = x_high

    df_high = pd.DataFrame(upsampled_cols, index=new_index)
    df_high.index.name = df_low.index.name
    return df_high


def get_scaled_train_val_dataloader(cfg, df, seq_len=40, filter_anomalies=True, transform=None, ano_col=None
                                    ,scale=True, scaler=None):

    # target can be a list of columns or a single column
    cfg.dataset.target = (
        cfg.dataset.target
        if isinstance(cfg.dataset.target, (list, ListConfig))
        else [cfg.dataset.target] if cfg.dataset.target
        else None
    )

    columns = cfg.dataset.feats + [
        x for x in cfg.dataset.target if x not in cfg.dataset.feats
    ] if cfg.dataset.target else cfg.dataset.feats

    cfg.dataset.target = columns if cfg.dataset.target is None else cfg.dataset.target
    # Add anomaly column if specified to searcher anomalous sequences
    columns = columns + [ano_col] if ano_col and ano_col in df.columns else columns

    df = df[columns].dropna()
    if cfg.dataset.dataset_subset:
        df = df.iloc[:cfg.dataset.dataset_subset, :]

    if cfg.dataset.get('upsample_factor', 0) > 1:
        print(f"🔧 Upsampling data by factor of {cfg.dataset.upsample_factor}")
        up_factor = cfg.dataset.get('upsample_factor')
        method = cfg.dataset.get('upsample_method', 'cubic')
        augmentation = cfg.dataset.get('upsample_augmentation', False)

        df = upsample_and_augment(df_low = df, factor=up_factor, method=method, do_augmentation=augmentation, ano_col=ano_col)

    # Use anomaly-aware split if anomaly column exists in the data
    use_anomaly_split = ano_col in df.columns and filter_anomalies
    # To do use decorator on the same function to divide ano searches behaviour
    # Train df_scaling is without eventual ano_col, so it can be used to fit the scaler
    train_indexes, val_indexes, train_df_for_scaling, anomalous_indexes = (
                    create_train_val_df_indexes(cfg=cfg, df=df, return_anomalies=use_anomaly_split, ano_col=ano_col, seed=42))

    # Scaling
    if scale:
        if scaler is None:
            scaler, df_scaled, scaler_params = get_scaler(cfg, df_fit=train_df_for_scaling, df_transform=df)
        else:
            scaling_cols = df.columns.difference([ano_col]) if ano_col in df.columns else df.columns
            assert list(scaling_cols) == list(scaler.feature_names_in_)
            scaled_values = scaler.transform(df[scaling_cols].values)
            ano_col = ano_col if isinstance(ano_col, (list, ListConfig)) else [ano_col] if ano_col else None
            all_df_values = np.concatenate((scaled_values, df[ano_col].values), axis=1) if ano_col else scaled_values
            df_scaled = pd.DataFrame(all_df_values, columns=df.columns)
            scaler_params = serialize_scaler(scaler)
    else:
        scaler = None
        df_scaled = df.copy()
        scaler_params = None

    # Create index sets and samplers
    step = seq_len - int(seq_len * cfg.dataset.perc_overlap)
    index_sets = {
        "train": (train_indexes, step, cfg.dataset.shuffle_train),
        "val": (val_indexes, step, False),
    }

    if anomalous_indexes is not None:
        metric_indexes = np.union1d(val_indexes, anomalous_indexes)
        index_sets["metric"] = (metric_indexes, seq_len)

    samplers = get_samplers_from_index_sets(index_sets)

    # Datasets
    dataset_args = dict(df=df_scaled,  target=cfg.dataset.target,
                        sequence_length=cfg.dataset.seq_in_length, out_window=cfg.dataset.seq_out_length,
                        forecast=cfg.dataset.forecast,
                        remove_target=cfg.dataset.remove_target,
                        transform=transform)

    train_dataset = Dataset_seq(**dataset_args, sampler=samplers['train'], indices=samplers['train'].indices)
    val_dataset = Dataset_seq(**dataset_args, sampler=samplers['val'], indices=samplers['val'].indices)

    trainloader = DataLoader(train_dataset, batch_size=cfg.opt.batch_size, sampler=samplers["train"])
    valloader = DataLoader(val_dataset, batch_size=cfg.opt.batch_size, sampler=samplers["val"])

    if "metric" in samplers:
        dataset_args['is_anomaly_column'] = ano_col
        metric_dataset = Dataset_seq(**dataset_args)
        metrics_loader = DataLoader(metric_dataset, batch_size=cfg.opt.batch_size, sampler=samplers["metric"])
    else:
        metrics_loader = None

    return trainloader, valloader, metrics_loader, scaler, scaler_params


def get_scaled_dataloader(cfg, df, seq_len=40, transform=None, ano_col=None, scale=True, scaler=None):

    # target can be a list of columns or a single column
    cfg.dataset.target = (
        cfg.dataset.target
        if isinstance(cfg.dataset.target, (list, ListConfig))
        else [cfg.dataset.target] if cfg.dataset.target
        else None
    )

    columns = cfg.dataset.feats + [
        x for x in cfg.dataset.target if x not in cfg.dataset.feats
    ] if cfg.dataset.target else cfg.dataset.feats

    cfg.dataset.target = columns if cfg.dataset.target is None else cfg.dataset.target
    # Add anomaly column if specified to searcher anomalous sequences
    columns = columns + [ano_col] if ano_col and ano_col in df.columns else columns

    df = df[columns].dropna()
    if cfg.dataset.dataset_subset:
        df = df.iloc[:cfg.dataset.dataset_subset, :]

    if scale:
        if scaler is None:
            scaler, df_scaled, scaler_params = get_scaler(cfg, df_fit=df, df_transform=df)
        else:
            scaling_cols = df.columns.difference([ano_col]) if ano_col in df.columns else df.columns
            assert list(scaling_cols) == list(scaler.feature_names_in_)
            scaled_values = scaler.transform(df[scaling_cols].values)
            ano_col = ano_col if isinstance(ano_col, list) else [ano_col] if ano_col else None
            all_df_values = np.concatenate((scaled_values, df[ano_col].values), axis=1) if ano_col else scaled_values
            df_scaled = pd.DataFrame(all_df_values, columns=df.columns)
            scaler_params = serialize_scaler(scaler)
    else:
        scaler = None
        df_scaled = df.copy()
        scaler_params = None

    indexes_sets = {}
    metric_indexes = list(range(len(df_scaled) - seq_len + 1))
    indexes_sets["dataset"] = (metric_indexes, seq_len, False)
    samplers = get_samplers_from_index_sets(indexes_sets)

    # Datasets
    dataset_args = dict(df=df_scaled,  target=cfg.dataset.target,
        sequence_length=cfg.dataset.seq_in_length, out_window=cfg.dataset.seq_out_length,
        is_anomaly_column=ano_col, remove_target=cfg.dataset.remove_target,
        forecast=cfg.dataset.forecast, transform=transform)

    dataset = Dataset_seq(**dataset_args, sampler=samplers["dataset"], indices=samplers['dataset'].indices)
    loader = DataLoader(dataset, batch_size=cfg.opt.batch_size)

    return loader, scaler, scaler_params





































def create_train_val_df(cfg, df, seed=42):

    num_chunks = cfg.dataset.chunks_num
    val_ratio = 1 - cfg.dataset.train_val_split

    np.random.seed(seed)  # Set seed for reproducibility
    df = df.reset_index(drop=True)  # preserve original index for window detection

    # Split into chunks for validation selection
    chunks = np.arange(num_chunks)
    chunk_size = len(df) // num_chunks
    val_chunk_num = int(num_chunks * val_ratio)
    np.random.shuffle(chunks)
    val_chunk_idxs = chunks[:val_chunk_num]

    val_indexes = []
    for i in val_chunk_idxs:
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < num_chunks - 1 else len(df)
        val_indexes.extend(range(start, end))

    all_indexes = np.arange(len(df))
    val_indexes = np.array(val_indexes)
    train_indexes = np.setdiff1d(all_indexes, val_indexes, assume_unique=True)

    # Split into train and val
    df_train = df.iloc[train_indexes].copy().reset_index(drop=True)
    df_val = df.iloc[val_indexes].copy().reset_index(drop=True)

    # inverse order split dataframe and index: split before the indexes and the the dataframe
    # remove df split and use only indexes

    return df_train, df_val


def create_train_val_df_with_ano(cfg, df, seq_len, ano_col='is_anomaly', num_chunks=12, val_ratio=0.1, seed=42):
    np.random.seed(seed)  # Set seed for reproducibility
    df = df.reset_index(drop=True)  # preserve original index for window detection

    # Split into chunks for validation selection
    chunks = np.arange(num_chunks)
    chunk_size = len(df) // num_chunks
    val_chunk_num = int(num_chunks * val_ratio)
    np.random.shuffle(chunks)
    val_chunk_idxs = chunks[:val_chunk_num]

    val_indexes = []
    for i in val_chunk_idxs:
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < num_chunks - 1 else len(df)
        val_indexes.extend(range(start, end))

    all_indexes = np.arange(len(df))
    val_indexes = np.array(val_indexes)
    train_indexes = np.setdiff1d(all_indexes, val_indexes, assume_unique=True)

    # Split into train and val
    df_train = df.iloc[train_indexes].copy().reset_index(drop=True)
    df_val = df.iloc[val_indexes].copy().reset_index(drop=True)

    # Identify anomaly indices within each split
    train_anomalous_idx = df_train[df_train[ano_col] == 1].index.to_numpy()
    val_anomalous_idx = df_val[df_val[ano_col] == 1].index.to_numpy()

    # Helper function to get anomaly window indices
    def get_anomaly_window_indexes(anomalous_idx, total_len):
        window_indexes = set()
        starting_idx = seq_len
        for i in anomalous_idx:
            start = max(i - starting_idx, 0)
            end = min(start + seq_len, total_len)
            start = max(end - seq_len, 0)  # adjust if end clipped
            window_indexes.update(range(start, end))
        return window_indexes

    # Get anomaly window indexes
    train_window_idxs = get_anomaly_window_indexes(train_anomalous_idx, len(df_train))
    val_window_idxs = get_anomaly_window_indexes(val_anomalous_idx, len(df_val))

    # Extract anomaly windows if any
    if len(train_window_idxs) > 0:
        df_train_ano = df_train.loc[sorted(train_window_idxs)].copy().reset_index(drop=True)
        df_train = df_train.drop(index=train_window_idxs).reset_index(drop=True)
    else:
        df_train_ano = pd.DataFrame(columns=df_train.columns)

    if len(val_window_idxs) > 0:
        df_val_ano = df_val.loc[sorted(val_window_idxs)].copy().reset_index(drop=True)
        df_val = df_val.drop(index=val_window_idxs).reset_index(drop=True)
    else:
        df_val_ano = pd.DataFrame(columns=df_val.columns)

    if cfg.dataset.limit_data is not None:
        val_limit = int(cfg.dataset.limit_data * cfg.dataset.val_ratio)
        df_val = df.iloc[:val_limit]
        train_limit = int(cfg.dataset.limit_data * (1 - cfg.dataset.val_ratio))
        df_train = df.iloc[:train_limit]

    return df_train, df_val, df_train_ano, df_val_ano


def get_scaled_train_val_df(cfg, df, seq_len=32, ano_col=None):
    ano_col = cfg.dataset.ano_columns
    cfg.dataset.target = (
        cfg.dataset.target
        if isinstance(cfg.dataset.target, list)
        else [cfg.dataset.target] if cfg.dataset.target
        else None
    )

    columns = cfg.dataset.feats + [
        x for x in cfg.dataset.target if x not in cfg.dataset.feats
    ] if cfg.dataset.target else cfg.dataset.feats
    columns = columns + [ano_col] if ano_col else columns

    dataRaw = df[columns].dropna()

    if cfg.dataset.dataset_subset:
        dataRaw = dataRaw.iloc[:cfg.dataset.dataset_subset, :]

    df = dataRaw.copy()

    # Use anomaly-aware split if anomaly column exists in the data
    use_anomaly_split = ano_col in df.columns

    if use_anomaly_split:
        train_df, val_df, train_df_ano, val_df_ano = create_train_val_df_with_ano(
            cfg, df, seq_len=seq_len, ano_col=ano_col, seed=42)
    else:
        train_df, val_df = create_train_val_df(cfg, df, seed=42)
        train_df_ano = val_df_ano = pd.DataFrame(columns=df.columns)
        train_anomalous_idxs = val_anomalous_idxs = []

    # Scaling
    scaler = get_scaler(cfg)
    if scaler:
        # Fit only on clean training data
        scaler.fit(train_df.values)

        train_df_scaled = pd.DataFrame(scaler.transform(train_df.values), columns=train_df.columns)
        val_df_scaled = pd.DataFrame(scaler.transform(val_df.values), columns=val_df.columns)
        df_scaled = pd.DataFrame(scaler.transform(df.values), columns=df.columns)

        train_df_ano_scaled = pd.DataFrame(scaler.transform(train_df_ano.values), columns=train_df.columns) if not train_df_ano.empty else train_df_ano
        val_df_ano_scaled = pd.DataFrame(scaler.transform(val_df_ano.values), columns=val_df.columns) if not val_df_ano.empty else val_df_ano

        scaler_params = serialize_scaler(scaler)
    else:
        # If no scaler used
        train_df_scaled = train_df
        val_df_scaled = val_df
        train_df_ano_scaled = train_df_ano
        val_df_ano_scaled = val_df_ano
        df_scaled = df
        scaler_params = None

    return (
        train_df_scaled,
        val_df_scaled,
        train_df_ano_scaled,
        val_df_ano_scaled,
        scaler,
        df_scaled,  # unscaled full dataset
        scaler_params
    )




def get_scaled_df(cfg, df, scale=False, scaler=None, add_columns=None):

    cfg.dataset.target = cfg.dataset.target if isinstance(cfg.dataset.target, list) else [cfg.dataset.target] if cfg.dataset.target else None
    columns = cfg.dataset.feats + [x for x in cfg.dataset.target if x not in cfg.dataset.feats] if cfg.dataset.target else cfg.dataset.feats
    if add_columns:
        columns += add_columns if isinstance(add_columns, list) else [add_columns]

    dataRaw = df[columns].dropna()
    if cfg.dataset.dataset_subset:
        dataRaw = dataRaw.iloc[:cfg.dataset.dataset_subset, :]

    df = dataRaw.copy()

    if scale:
        if scaler:
            df_scaled = scaler.transform(df.values)
            df = pd.DataFrame(df_scaled, columns=df.columns)
            scaler_params = serialize_scaler(scaler)
        else:
            scaler = get_scaler(cfg)
            df_scaled = scaler.fit_transform(df.values)
            df = pd.DataFrame(df_scaled, columns=df.columns)

            scaler_params = serialize_scaler(scaler)
    else:
        scaler = None
        scaler_params = None

    return df, scaler, df, scaler_params


def get_train_val_samplers(cfg, df):
    # return trian, val and scaler
    np.random.seed(101)
    step = cfg.dataset.seq_in_length - int(cfg.dataset.seq_in_length * cfg.dataset.perc_overlap)
    print('using step', step)
    print('perc_overlap', cfg.dataset.perc_overlap)
    print('using sequence length', cfg.dataset.seq_in_length)
    step = step if step > 0 else 1
    print('step', step)

    dataset_size = len(df)
    idxs = np.arange(0, dataset_size, step)
    print('idxs', idxs[:10])
    print('len idxs', len(idxs))

    train_split_idx = int(np.floor(cfg.dataset.train_val_split * len(idxs)))
    train_idx, val_idx = idxs[:train_split_idx], idxs[train_split_idx:]

    if cfg.dataset.shuffle_train:
        np.random.shuffle(train_idx)

    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    return train_sampler, val_sampler


