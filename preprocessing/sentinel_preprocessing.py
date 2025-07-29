import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from torch.utils.data import SubsetRandomSampler
import numpy as np

def serialize_scaler(scaler):
    if isinstance(scaler, StandardScaler):
        return {
            'type': 'StandardScaler',
            'mean_': scaler.mean_.tolist(),
            'scale_': scaler.scale_.tolist(),
            'var_': scaler.var_.tolist(),
            'n_samples_seen_': int(scaler.n_samples_seen_)
        }

    elif isinstance(scaler, RobustScaler):
        return {
            'type': 'RobustScaler',
            'center_': scaler.center_.tolist(),
            'scale_': scaler.scale_.tolist(),
            'quantile_range': scaler.quantile_range,
        }

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

def get_scaler(cfg):
    scaler_cfg = cfg['dataset'].get('scaler', None)
    if not scaler_cfg:
        return None

    # Extract scaler name and params
    scaler_name, scaler_params = list(scaler_cfg.items())[0]

    if scaler_name == 'StandardScaler':
        return StandardScaler()
    elif scaler_name == 'RobustScaler':
        # Translate config keys to sklearn-compatible ones
        q1 = scaler_params.get('qr_1', 0.25)
        q2 = scaler_params.get('qr_2', 0.75)
        return RobustScaler(quantile_range=(q1 * 100, q2 * 100))
    else:
        raise ValueError(f"Scaler '{scaler_name}' not supported.")


def create_train_val_df_with_ano(cfg, df, seq_len, num_chunks=12, val_ratio=0.1, seed=42):
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
    train_anomalous_idx = df_train[df_train['is_anomaly'] == 1].index.to_numpy()
    val_anomalous_idx = df_val[df_val['is_anomaly'] == 1].index.to_numpy()

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

def get_scaled_train_val_df(cfg, df):
    dataRaw = df[cfg.dataset.feats].dropna()
    if cfg.dataset.dataset_subset:
        dataRaw = dataRaw.iloc[:cfg.dataset.dataset_subset, :]

    df = dataRaw.copy()
    train_df, val_df = create_train_val_df(cfg, df, seed=42)

    scaler = get_scaler(cfg)

    if scaler:
        scaler.fit(train_df)
        train_df_scaled = scaler.transform(train_df.values)
        val_df_scaled = scaler.transform(val_df.values)

        train_df = pd.DataFrame(train_df_scaled, columns=train_df.columns)
        val_df = pd.DataFrame(val_df_scaled, columns=val_df.columns)

        scaler_params = serialize_scaler(scaler)
    else:
        scaler = None
        scaler_params = None

    return train_df, val_df, scaler, df, scaler_params



def get_train_val_samplers(cfg, df):
    # return trian, val and scaler
    np.random.seed(101)
    step = cfg.dataset.sequence_length - int(cfg.dataset.sequence_length * cfg.dataset.perc_overlap)
    step = step if step > 0 else 1
    print('step', step)

    dataset_size = len(df)
    idxs = np.arange(0, dataset_size, step)
    print('idxs', idxs[:10])
    print('len idxs', len(idxs))

    train_split_idx = int(np.floor(cfg.dataset.train_val_split * len(idxs)))
    train_idx, val_idx = idxs[:train_split_idx], idxs[train_split_idx:]

    if cfg.dataset.shuffle_train:
        print('train idx before shuffle', train_idx[:10])
        np.random.shuffle(train_idx)
        print('train idx after shuffle', train_idx[:10])

    print('train idx', train_idx[:10])
    print('val idx', val_idx[:10])

    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    return train_sampler, val_sampler
