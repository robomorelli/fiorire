import pandas as pd
from omegaconf import ListConfig
from sklearn.preprocessing import StandardScaler, RobustScaler
from dataset.sentinel import Dataset_seq
from torch.utils.data import SubsetRandomSampler, DataLoader
import numpy as np

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

def create_train_val_df_indexes(cfg, df, return_anomalies=False, ano_col='is_anomaly', seed=42):
    """
    Splits a DataFrame into training and validation sets using a chunk-based split strategy,
    while identifying and separating anomalous windows if target columns indicate anomalies.

    This function returns the **index positions** (not the actual data) of clean (non-anomalous)
    sequences for both training and validation and, if required, the training dartaframe to fit the scaler.
    These train val and anomalous indexes correspond to **all data points**
    and can be used to generate sequence windows with a given step size inside a custom Dataset
    or DataLoader.

    Anomalous sequences are defined as any window of `seq_len` points that contains at least
    one anomaly (i.e., one row where `is_anomaly == 1`). These anomalous window indexes can be
    used separately for evaluation or excluded from training.

    Returns:
        train_normal_idxs (np.ndarray): Indexes of clean data in the training set.
        val_normal_idxs (np.ndarray): Indexes of clean data in the validation set.
        df_train_values_for_scaling (pd.DataFrame): Clean (non-anomalous) training data, useful for fitting the scaler.
        Remove also ano target column if it exists.
        anomalous_window_idxs (List[int]): Indexes of all windows in the dataset affected by anomalies.

    Notes:
        - `train_normal_idxs` and `val_normal_idxs` refer to **individual rows**, not sequence windows.
          They are intended to be used with a sliding window generator that samples sequences from
          the full data using a step size.
        - `anomalous_window_idxs` covers all indices that fall within a sequence containing at least
          one anomaly — i.e., these are *not* safe for training the model.
    """
    seq_len = cfg.dataset.seq_in_length
    num_chunks = min(cfg.dataset.chunks_num, 3)
    val_ratio = 1 - cfg.dataset.train_val_split
    np.random.seed(seed)
    df = df.reset_index(drop=True)

    # Validation set chunk selection
    chunks = np.arange(num_chunks)
    chunk_size = len(df) // num_chunks
    val_chunk_num = int(np.ceil(num_chunks * val_ratio))
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

    if not return_anomalies:
        df_train_values_for_scaling = df.iloc[train_indexes].reset_index(drop=True)
        return train_indexes, val_indexes, df_train_values_for_scaling, None

    # Otherwise, also compute anomalous windows
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

    # search for anomalous windows in the training and validation sets
    train_anomalous_windows = get_anomaly_window_indexes(train_anomalous_indexes, len(df))
    val_anomalous_windows = get_anomaly_window_indexes(val_anomalous_indexes, len(df))

    # Get indexes of clean data in train and val sets
    train_normal_indexes = np.setdiff1d(train_indexes, list(train_anomalous_windows), assume_unique=True)
    val_normal_indexes= np.setdiff1d(val_indexes, list(val_anomalous_windows), assume_unique=True)

    # Get anomalous window indexes for the full dataset
    anomalous_window_indexes = list(get_anomaly_window_indexes(full_anomalous_idx, len(df)))
    # Create DataFrame for training values to fit the scaler
    scaling_cols = df.columns.difference([ano_col]) if ano_col in df.columns else df.columns
    df_train_values_for_scaling = df[scaling_cols].iloc[train_normal_indexes].reset_index(drop=True)

    return train_normal_indexes, val_normal_indexes, df_train_values_for_scaling, anomalous_window_indexes


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


