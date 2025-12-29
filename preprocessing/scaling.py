"""
Data scaling utilities.
Handles scaler fitting and application for time series data.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


def get_scaler(scaler_type: str = 'StandardScaler'):
    """
    Get scaler instance by name.

    Args:
        scaler_type: Type of scaler ('StandardScaler', 'MinMaxScaler', 'RobustScaler')

    Returns:
        Scaler instance
    """
    if scaler_type == 'StandardScaler':
        return StandardScaler()
    elif scaler_type == 'MinMaxScaler':
        return MinMaxScaler()
    elif scaler_type == 'RobustScaler':
        return RobustScaler()
    else:
        raise ValueError(f"Unknown scaler: {scaler_type}")


def fit_scaler_on_indices(df, indices, seq_len, feature_columns, cfg):
    """
    Fit scaler ONLY on data covered by given indices.

    This ensures scaler is fitted only on training data,
    avoiding data leakage from validation set.

    Args:
        df: Raw DataFrame
        indices: Training indices (starting positions of sequences)
        seq_len: Sequence length
        feature_columns: Columns to scale
        cfg: Configuration

    Returns:
        Fitted scaler object
    """
    scaler_type = cfg.dataset.get('scaler', 'StandardScaler')
    scaler = get_scaler(scaler_type)

    # Collect all data points covered by indices
    all_rows = []
    for start_idx in indices:
        end_idx = start_idx + seq_len
        if end_idx <= len(df):
            all_rows.extend(range(start_idx, end_idx))

    # Get unique rows (in case of overlap in indices)
    unique_rows = sorted(set(all_rows))

    # Fit on these rows only
    train_data = df.iloc[unique_rows][feature_columns].values
    scaler.fit(train_data)

    print(f"      - Scaler type: {scaler_type}")
    print(f"      - Fitted on: {len(unique_rows)} data points")
    print(f"      - Features: {len(feature_columns)}")

    return scaler


def apply_scaler(df, scaler, feature_columns):
    """
    Apply fitted scaler to DataFrame.

    Args:
        df: Raw DataFrame
        scaler: Fitted scaler object
        feature_columns: Columns to scale

    Returns:
        Scaled DataFrame
    """
    df_scaled = df.copy()

    # Transform
    scaled_values = scaler.transform(df[feature_columns].values)

    # Assign back column by column (pandas-safe)
    for i, col in enumerate(feature_columns):
        df_scaled[col] = scaled_values[:, i]

    print(f"   ✓ Scaled {len(feature_columns)} features")

    return df_scaled


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


def get_scaler(cfg, df_fit=None, df_transform=None):
    scaler_cfg = cfg['dataset'].get('scaler', None)
    if not scaler_cfg:
        return None, df_transform, None  # No scaler, return original df_transform and None for params

    # Identify anomaly flag column if present
    anomaly_col = getattr(cfg.dataset, "is_anomaly_column", None)

    # Prepare df_fit (exclude anomaly column)
    if df_fit is not None:
        if anomaly_col and anomaly_col in df_fit.columns:
            df_fit_no_anomaly = df_fit.drop(columns=[anomaly_col])
        else:
            df_fit_no_anomaly = df_fit.copy()
    else:
        df_fit_no_anomaly = None

    # Prepare df_transform (exclude anomaly column)
    if df_transform is not None:
        if anomaly_col and anomaly_col in df_transform.columns:
            df_transform_no_anomaly = df_transform.drop(columns=[anomaly_col])
        else:
            df_transform_no_anomaly = df_transform.copy()
    else:
        df_transform_no_anomaly = None

    # Sanity check: columns must match
    if df_fit_no_anomaly is not None and df_transform_no_anomaly is not None:
        assert df_fit_no_anomaly.columns.equals(df_transform_no_anomaly.columns), \
            "DataFrames for fitting and transforming must have the same columns."

    # --- Extract scaler name and parameters ---
    parts = scaler_cfg.split('-')
    scaler_name = parts[0]

    if scaler_name == 'StandardScaler':
        scaler = StandardScaler()
    elif scaler_name == 'RobustScaler':
        # Check if quantiles are provided
        if len(parts) >= 3:
            q1 = float(parts[1])
            q2 = float(parts[2])
        else:
            q1, q2 = 0.25, 0.75  # default
        scaler = RobustScaler(quantile_range=(q1 * 100, q2 * 100))
    else:
        raise ValueError(f"Scaler '{scaler_name}' not supported.")

    # --- Fit the scaler ---
    if df_fit_no_anomaly is not None:
        scaler.fit(df_fit_no_anomaly)
    else:
        return scaler, None, None

    # --- Transform if df_transform provided ---
    if df_transform_no_anomaly is not None:
        df_scaled = pd.DataFrame(
            scaler.transform(df_transform_no_anomaly),
            columns=df_transform_no_anomaly.columns,
            index=df_transform_no_anomaly.index
        )

        # Re-attach the anomaly flag column if present
        if anomaly_col and anomaly_col in df_transform.columns:
            df_scaled[anomaly_col] = df_transform[anomaly_col]

        # Serialize parameters (if you have a helper)
        scaler_params = serialize_scaler(scaler)
        return scaler, df_scaled, scaler_params
    else:
        return scaler, None, None



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