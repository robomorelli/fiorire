import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter, gaussian_filter1d
import pandas as pd
from omegaconf import ListConfig



def apply_smoothing_groups(df, smooth_configs, feature_cols, ano_col=None):
    """
    Apply different smoothing to different feature groups.

    Args:
        df: DataFrame to smooth
        smooth_configs: List of smoothing config dicts, each with:
            - mode: smoothing method
            - features: 'all' or list of columns
            - kernel_size: window size
            - pad_mode: 'drop', 'forward_fill', 'reflect', 'edge' (default: 'drop')
            - ... (method-specific params)
        feature_cols: List of feature column names
        ano_col: Anomaly column name

    Returns:
        df: DataFrame with smoothed features
    """
    df_smoothed = df.copy()
    smoothed_cols = set()  # Track which columns have been smoothed

    for i, smooth_cfg in enumerate(smooth_configs):
        print(f"\n   📦 Smoothing group {i + 1}/{len(smooth_configs)}:")

        mode = smooth_cfg.get('mode', 'mean')
        features = smooth_cfg.get('features', 'all')
        kernel_size = smooth_cfg.get('kernel_size', 10)
        pad_mode = smooth_cfg.get('pad_mode', 'drop')  # 'drop', 'ffill', 'reflect', 'edge'

        # Determine columns for this group
        if features == 'all' or features == ['all']:
            # All features not yet smoothed
            cols_to_smooth = [c for c in feature_cols if c not in smoothed_cols]
        elif isinstance(features, (list, ListConfig)):
            cols_to_smooth = [f for f in features if f in df.columns and f not in smoothed_cols]
        else:
            cols_to_smooth = [features] if features in df.columns and features not in smoothed_cols else []

        # Remove anomaly column
        if ano_col and ano_col in cols_to_smooth:
            cols_to_smooth.remove(ano_col)

        if not cols_to_smooth:
            print(f"      ⚠️  No columns to smooth in this group")
            continue

        print(f"      - Mode: {mode}")
        print(f"      - Kernel: {kernel_size}")
        print(f"      - Pad mode: {pad_mode}")
        print(f"      - Columns: {cols_to_smooth}")

        # Apply smoothing with padding strategy
        df_smoothed = apply_smoothing_with_padding(
            df_smoothed,
            cols_to_smooth,
            mode,
            kernel_size,
            pad_mode,
            smooth_cfg
        )

        # Mark as smoothed
        smoothed_cols.update(cols_to_smooth)

    # Check for unsmoothed features
    unsmoothed = [c for c in feature_cols if c not in smoothed_cols and c != ano_col]
    if unsmoothed:
        print(f"\n   ℹ️  Unsmoothed features: {unsmoothed}")

    return df_smoothed


def apply_smoothing_with_padding(df, cols_to_smooth, mode, kernel_size, pad_mode, smooth_cfg):
    """
    Apply smoothing with different padding strategies to avoid data loss.
    """
    from scipy.signal import savgol_filter
    from scipy.ndimage import median_filter, gaussian_filter1d
    import numpy as np

    df_smoothed = df.copy()

    # ✅ FIX: Map pad_mode to scipy mode correctly
    pad_mode_to_scipy = {
        'reflect': 'mirror',  # ← FIX: scipy usa 'mirror' non 'reflect'
        'edge': 'nearest',
        'constant': 'constant',
        'wrap': 'wrap',
        'mirror': 'mirror',
        'nearest': 'nearest',
        'drop': 'nearest',  # Use nearest then drop NaN
        'ffill': 'nearest'  # Use nearest then ffill
    }

    scipy_mode = pad_mode_to_scipy.get(pad_mode, 'nearest')

    print(f"         - Padding: {pad_mode} → scipy mode: {scipy_mode}")

    for col in cols_to_smooth:
        signal = df[col].values

        try:
            # Apply smoothing based on mode
            if mode == 'mean':
                # Rolling mean with padding
                if pad_mode == 'ffill':
                    smoothed = df[col].rolling(
                        window=kernel_size,
                        min_periods=1,
                        center=True
                    ).mean()
                elif pad_mode == 'drop':
                    smoothed = df[col].rolling(
                        window=kernel_size,
                        min_periods=kernel_size,
                        center=True
                    ).mean()
                else:  # reflect, edge, etc.
                    pad_width = kernel_size // 2
                    if pad_mode == 'reflect' or pad_mode == 'mirror':
                        padded = np.pad(signal, pad_width, mode='reflect')
                    elif pad_mode == 'edge' or pad_mode == 'nearest':
                        padded = np.pad(signal, pad_width, mode='edge')
                    elif pad_mode == 'wrap':
                        padded = np.pad(signal, pad_width, mode='wrap')
                    else:
                        padded = np.pad(signal, pad_width, mode='edge')

                    smoothed = pd.Series(padded).rolling(
                        window=kernel_size,
                        min_periods=kernel_size,
                        center=True
                    ).mean().values[pad_width:-pad_width]

            elif mode == 'median':
                # Scipy median filter
                smoothed = median_filter(signal, size=kernel_size, mode=scipy_mode)

            elif mode == 'gaussian':
                sigma = kernel_size / 6
                smoothed = gaussian_filter1d(signal, sigma, mode=scipy_mode)

            elif mode == 'savgol':
                polyorder = min(3, kernel_size - 1)
                if kernel_size % 2 == 0:
                    kernel_size += 1
                    print(f"         - Adjusted kernel_size to {kernel_size} (must be odd for savgol)")

                # ✅ FIX: Use correct scipy mode
                smoothed = savgol_filter(signal, kernel_size, polyorder, mode=scipy_mode)

            elif mode == 'ewm':
                # EWM doesn't lose data at edges
                smoothed = df[col].ewm(span=kernel_size, min_periods=1).mean()

            else:
                print(f"         ⚠️  Unknown mode '{mode}', using mean")
                smoothed = df[col].rolling(window=kernel_size, min_periods=1, center=True).mean()

            df_smoothed[col] = smoothed

        except Exception as e:
            print(f"         ❌ Error smoothing column '{col}': {e}")
            print(f"         → Keeping original values for this column")
            # Keep original values on error
            df_smoothed[col] = df[col]

    # Handle NaN based on pad_mode
    if pad_mode == 'drop':
        initial_rows = len(df_smoothed)
        df_smoothed = df_smoothed.dropna()
        dropped = initial_rows - len(df_smoothed)
        if dropped > 0:
            print(f"         → Dropped {dropped} rows with NaN ({100 * dropped / initial_rows:.4f}%)")
    elif pad_mode == 'ffill':
        # Forward fill then backward fill to handle all NaN
        df_smoothed = df_smoothed.fillna(method='ffill').fillna(method='bfill')
        print(f"         → Forward/backward filled NaN values")
    # For scipy modes (mirror, nearest, etc.), no NaN should be present
    else:
        # Check if there are any NaN (shouldn't be with proper scipy modes)
        nan_count = df_smoothed[cols_to_smooth].isnull().sum().sum()
        if nan_count > 0:
            print(f"         ⚠️  Found {nan_count} NaN values, forward filling...")
            df_smoothed = df_smoothed.fillna(method='ffill').fillna(method='bfill')

    return df_smoothed


def apply_smoothing_to_dataframe(df, cfg, feature_columns):
    """
    Apply smoothing to a DataFrame (wrapper around existing apply_smoothing_groups).

    Args:
        df: DataFrame with time series data
        cfg: configuration
        feature_columns: list of columns to smooth

    Returns:
        df_smoothed: DataFrame with smoothed data
    """
    smooth_cfg = cfg.dataset.get('smooth', None)

    if smooth_cfg is None:
        print(f"      ℹ️  No smoothing configured - returning original data")
        return df.copy()

    print(f"      🔧 Applying smoothing to dataframe...")

    # Get anomaly column
    ano_col = cfg.dataset.get('is_anomaly_column')

    # Supporta sia singolo dict che lista di dict
    if isinstance(smooth_cfg, dict):
        smooth_configs = [smooth_cfg]
    else:
        smooth_configs = list(smooth_cfg)

    # Use your existing function
    df_smoothed = apply_smoothing_groups(df, smooth_configs, feature_columns, ano_col)

    return df_smoothed



def apply_smoothing_to_sequences(sequences, cfg, feature_columns):
    """
    Apply smoothing to 3D sequences array.
    Replicates the exact logic of apply_smoothing_groups + apply_smoothing_with_padding.

    Args:
        sequences: (N, L, F) array
        cfg: configuration object
        feature_columns: list of feature names

    Returns:
        sequences_smoothed: (N, L, F) array
    """
    smooth_cfg = cfg.dataset.get('smooth', None)

    if smooth_cfg is None:
        print(f"      ℹ️  No smoothing configured - returning original data")
        return sequences.copy()

    print(f"      🔧 Applying smoothing to sequences...")

    N, L, F = sequences.shape
    sequences_smoothed = sequences.copy()

    # Get anomaly column index (if exists)
    ano_col = cfg.dataset.get('is_anomaly_column')
    ano_col_idx = feature_columns.index(ano_col) if ano_col and ano_col in feature_columns else None

    # Support both single dict and list of dicts
    if isinstance(smooth_cfg, dict):
        smooth_configs = [smooth_cfg]
    else:
        smooth_configs = list(smooth_cfg)

    smoothed_indices = set()  # Track which feature indices have been smoothed

    for i, smooth_config in enumerate(smooth_configs):
        print(f"\n   📦 Smoothing group {i + 1}/{len(smooth_configs)}:")

        mode = smooth_config.get('mode', 'mean')
        features = smooth_config.get('features', 'all')
        kernel_size = smooth_config.get('kernel_size', 10)
        pad_mode = smooth_config.get('pad_mode', 'drop')

        # Determine feature indices for this group
        if features == 'all' or features == ['all']:
            # All features not yet smoothed
            indices_to_smooth = [idx for idx in range(F) if idx not in smoothed_indices]
        elif isinstance(features, list):
            indices_to_smooth = [
                feature_columns.index(f)
                for f in features
                if f in feature_columns and feature_columns.index(f) not in smoothed_indices
            ]
        else:
            idx = feature_columns.index(features) if features in feature_columns else None
            indices_to_smooth = [idx] if idx is not None and idx not in smoothed_indices else []

        # Remove anomaly column index
        if ano_col_idx is not None and ano_col_idx in indices_to_smooth:
            indices_to_smooth.remove(ano_col_idx)

        if not indices_to_smooth:
            print(f"      ⚠️  No features to smooth in this group")
            continue

        cols_names = [feature_columns[idx] for idx in indices_to_smooth]
        print(f"      - Mode: {mode}")
        print(f"      - Kernel: {kernel_size}")
        print(f"      - Pad mode: {pad_mode}")
        print(f"      - Features: {cols_names}")

        # Apply smoothing with padding strategy (SAME as dataframe)
        sequences_smoothed = apply_smoothing_with_padding_sequences(
            sequences_smoothed,
            indices_to_smooth,
            mode,
            kernel_size,
            pad_mode,
            smooth_config
        )

        # Mark as smoothed
        smoothed_indices.update(indices_to_smooth)

    # Check for unsmoothed features
    unsmoothed_indices = [
        idx for idx in range(F)
        if idx not in smoothed_indices and idx != ano_col_idx
    ]
    if unsmoothed_indices:
        unsmoothed_names = [feature_columns[idx] for idx in unsmoothed_indices]
        print(f"\n   ℹ️  Unsmoothed features: {unsmoothed_names}")

    return sequences_smoothed


def apply_smoothing_with_padding_sequences(sequences, indices_to_smooth, mode, kernel_size, pad_mode, smooth_cfg):
    """
    Apply smoothing with padding strategies to sequences.
    Replicates the exact logic of apply_smoothing_with_padding but for 3D arrays.

    Args:
        sequences: (N, L, F) array
        indices_to_smooth: list of feature indices to smooth
        mode: smoothing mode
        kernel_size: window size
        pad_mode: padding mode
        smooth_cfg: config dict with method-specific params

    Returns:
        sequences_smoothed: (N, L, F) array
    """
    N, L, F = sequences.shape
    sequences_smoothed = sequences.copy()

    # Map pad_mode to scipy mode (SAME as dataframe)
    pad_mode_to_scipy = {
        'reflect': 'mirror',
        'edge': 'nearest',
        'constant': 'constant',
        'wrap': 'wrap',
        'mirror': 'mirror',
        'nearest': 'nearest',
        'drop': 'nearest',
        'ffill': 'nearest'
    }

    scipy_mode = pad_mode_to_scipy.get(pad_mode, 'nearest')
    print(f"         - Padding: {pad_mode} → scipy mode: {scipy_mode}")

    for f_idx in indices_to_smooth:
        try:
            # Process each sequence for this feature
            for seq_idx in range(N):
                signal = sequences[seq_idx, :, f_idx]

                # Apply smoothing based on mode (SAME logic as dataframe)
                if mode == 'mean':
                    if pad_mode == 'ffill':
                        smoothed = pd.Series(signal).rolling(
                            window=kernel_size,
                            min_periods=1,
                            center=True
                        ).mean().values
                    elif pad_mode == 'drop':
                        smoothed = pd.Series(signal).rolling(
                            window=kernel_size,
                            min_periods=kernel_size,
                            center=True
                        ).mean().values
                    else:  # reflect, edge, wrap, etc.
                        pad_width = kernel_size // 2
                        if pad_mode == 'reflect' or pad_mode == 'mirror':
                            padded = np.pad(signal, pad_width, mode='reflect')
                        elif pad_mode == 'edge' or pad_mode == 'nearest':
                            padded = np.pad(signal, pad_width, mode='edge')
                        elif pad_mode == 'wrap':
                            padded = np.pad(signal, pad_width, mode='wrap')
                        else:
                            padded = np.pad(signal, pad_width, mode='edge')

                        smoothed = pd.Series(padded).rolling(
                            window=kernel_size,
                            min_periods=kernel_size,
                            center=True
                        ).mean().values[pad_width:-pad_width]

                elif mode == 'median':
                    smoothed = median_filter(signal, size=kernel_size, mode=scipy_mode)

                elif mode == 'gaussian':
                    sigma = kernel_size / 6
                    smoothed = gaussian_filter1d(signal, sigma, mode=scipy_mode)

                elif mode == 'savgol':
                    polyorder = min(3, kernel_size - 1)
                    if kernel_size % 2 == 0:
                        kernel_size += 1

                    smoothed = savgol_filter(signal, kernel_size, polyorder, mode=scipy_mode)

                elif mode == 'ewm':
                    smoothed = pd.Series(signal).ewm(span=kernel_size, min_periods=1).mean().values

                else:
                    print(f"         ⚠️  Unknown mode '{mode}', using mean")
                    smoothed = pd.Series(signal).rolling(
                        window=kernel_size,
                        min_periods=1,
                        center=True
                    ).mean().values

                sequences_smoothed[seq_idx, :, f_idx] = smoothed

        except Exception as e:
            print(f"         ❌ Error smoothing feature index {f_idx}: {e}")
            print(f"         → Keeping original values for this feature")
            # Keep original values on error
            sequences_smoothed[:, :, f_idx] = sequences[:, :, f_idx]

    # Handle NaN based on pad_mode (SAME as dataframe)
    if pad_mode == 'drop':
        # For sequences, we'd need to identify which timesteps have NaN
        # This is complex, so we'll use ffill instead
        print(f"         ⚠️  'drop' mode not fully supported for sequences, using ffill")
        for f_idx in indices_to_smooth:
            for seq_idx in range(N):
                series = pd.Series(sequences_smoothed[seq_idx, :, f_idx])
                sequences_smoothed[seq_idx, :, f_idx] = series.fillna(method='ffill').fillna(method='bfill').values

    elif pad_mode == 'ffill':
        for f_idx in indices_to_smooth:
            for seq_idx in range(N):
                series = pd.Series(sequences_smoothed[seq_idx, :, f_idx])
                sequences_smoothed[seq_idx, :, f_idx] = series.fillna(method='ffill').fillna(method='bfill').values
        print(f"         → Forward/backward filled NaN values")
    else:
        # Check for NaN (shouldn't be with proper scipy modes)
        nan_mask = np.isnan(sequences_smoothed[:, :, indices_to_smooth])
        nan_count = nan_mask.sum()
        if nan_count > 0:
            print(f"         ⚠️  Found {nan_count} NaN values, forward filling...")
            for f_idx in indices_to_smooth:
                for seq_idx in range(N):
                    series = pd.Series(sequences_smoothed[seq_idx, :, f_idx])
                    sequences_smoothed[seq_idx, :, f_idx] = series.fillna(method='ffill').fillna(method='bfill').values

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
    import numpy as np

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
    import numpy as np

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
    Serialize scaler to dictionary (uses existing function logic).

    Args:
        scaler: Fitted scaler object

    Returns:
        Dictionary with scaler parameters
    """
    from sklearn.preprocessing import StandardScaler, RobustScaler

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
            'quantile_range': scaler.quantile_range
        }

    else:
        raise ValueError(f"Cannot serialize unknown scaler type: {type(scaler)}")


def deserialize_scaler(scaler_params):
    """
    Deserialize scaler from dictionary (uses existing function logic).

    Args:
        scaler_params: Dictionary with scaler parameters

    Returns:
        Fitted scaler object
    """
    import numpy as np
    from sklearn.preprocessing import StandardScaler, RobustScaler

    scaler_type = scaler_params['type']

    if scaler_type == 'StandardScaler':
        scaler = StandardScaler()
        scaler.mean_ = np.array(scaler_params['mean_'])
        scaler.scale_ = np.array(scaler_params['scale_'])
        scaler.var_ = np.array(scaler_params['var_'])
        scaler.n_samples_seen_ = scaler_params.get('n_samples_seen_', len(scaler.mean_))
        return scaler

    elif scaler_type == 'RobustScaler':
        scaler = RobustScaler()
        scaler.center_ = np.array(scaler_params['center_'])
        scaler.scale_ = np.array(scaler_params['scale_'])
        scaler.quantile_range = tuple(scaler_params['quantile_range'])
        return scaler

    else:
        raise ValueError(f"Cannot deserialize unknown scaler type: {scaler_type}")

''' 
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


def apply_smoothing_to_dataframe(df, cfg, feature_columns):
    """
    Apply smoothing to a DataFrame (wrapper around existing apply_smoothing_groups).

    Args:
        df: DataFrame with time series data
        cfg: configuration
        feature_columns: list of columns to smooth

    Returns:
        df_smoothed: DataFrame with smoothed data
    """
    smooth_cfg = cfg.dataset.get('smooth', None)

    if smooth_cfg is None:
        print(f"      ℹ️  No smoothing configured - returning original data")
        return df.copy()

    print(f"      🔧 Applying smoothing to dataframe...")

    # Get anomaly column
    ano_col = cfg.dataset.get('is_anomaly_column')

    # Supporta sia singolo dict che lista di dict
    if isinstance(smooth_cfg, dict):
        smooth_configs = [smooth_cfg]
    else:
        smooth_configs = list(smooth_cfg)

    # Use your existing function
    df_smoothed = apply_smoothing_groups(df, smooth_configs, feature_columns, ano_col)

    return df_smoothed




def apply_smoothing_with_padding(df, cols_to_smooth, mode, kernel_size, pad_mode, smooth_cfg):
    """
    Apply smoothing with different padding strategies to avoid data loss.
    """


    df_smoothed = df.copy()

    # ✅ FIX: Map pad_mode to scipy mode correctly
    pad_mode_to_scipy = {
        'reflect': 'mirror',  # ← FIX: scipy usa 'mirror' non 'reflect'
        'edge': 'nearest',
        'constant': 'constant',
        'wrap': 'wrap',
        'mirror': 'mirror',
        'nearest': 'nearest',
        'drop': 'nearest',  # Use nearest then drop NaN
        'ffill': 'nearest'  # Use nearest then ffill
    }

    scipy_mode = pad_mode_to_scipy.get(pad_mode, 'nearest')

    print(f"         - Padding: {pad_mode} → scipy mode: {scipy_mode}")

    for col in cols_to_smooth:
        signal = df[col].values

        try:
            # Apply smoothing based on mode
            if mode == 'mean':
                # Rolling mean with padding
                if pad_mode == 'ffill':
                    smoothed = df[col].rolling(
                        window=kernel_size,
                        min_periods=1,
                        center=True
                    ).mean()
                elif pad_mode == 'drop':
                    smoothed = df[col].rolling(
                        window=kernel_size,
                        min_periods=kernel_size,
                        center=True
                    ).mean()
                else:  # reflect, edge, etc.
                    pad_width = kernel_size // 2
                    if pad_mode == 'reflect' or pad_mode == 'mirror':
                        padded = np.pad(signal, pad_width, mode='reflect')
                    elif pad_mode == 'edge' or pad_mode == 'nearest':
                        padded = np.pad(signal, pad_width, mode='edge')
                    elif pad_mode == 'wrap':
                        padded = np.pad(signal, pad_width, mode='wrap')
                    else:
                        padded = np.pad(signal, pad_width, mode='edge')

                    smoothed = pd.Series(padded).rolling(
                        window=kernel_size,
                        min_periods=kernel_size,
                        center=True
                    ).mean().values[pad_width:-pad_width]

            elif mode == 'median':
                # Scipy median filter
                smoothed = median_filter(signal, size=kernel_size, mode=scipy_mode)

            elif mode == 'gaussian':
                sigma = kernel_size / 6
                smoothed = gaussian_filter1d(signal, sigma, mode=scipy_mode)

            elif mode == 'savgol':
                polyorder = min(3, kernel_size - 1)
                if kernel_size % 2 == 0:
                    kernel_size += 1
                    print(f"         - Adjusted kernel_size to {kernel_size} (must be odd for savgol)")

                # ✅ FIX: Use correct scipy mode
                smoothed = savgol_filter(signal, kernel_size, polyorder, mode=scipy_mode)

            elif mode == 'ewm':
                # EWM doesn't lose data at edges
                smoothed = df[col].ewm(span=kernel_size, min_periods=1).mean()

            else:
                print(f"         ⚠️  Unknown mode '{mode}', using mean")
                smoothed = df[col].rolling(window=kernel_size, min_periods=1, center=True).mean()

            df_smoothed[col] = smoothed

        except Exception as e:
            print(f"         ❌ Error smoothing column '{col}': {e}")
            print(f"         → Keeping original values for this column")
            # Keep original values on error
            df_smoothed[col] = df[col]

    # Handle NaN based on pad_mode
    if pad_mode == 'drop':
        initial_rows = len(df_smoothed)
        df_smoothed = df_smoothed.dropna()
        dropped = initial_rows - len(df_smoothed)
        if dropped > 0:
            print(f"         → Dropped {dropped} rows with NaN ({100 * dropped / initial_rows:.4f}%)")
    elif pad_mode == 'ffill':
        # Forward fill then backward fill to handle all NaN
        df_smoothed = df_smoothed.fillna(method='ffill').fillna(method='bfill')
        print(f"         → Forward/backward filled NaN values")
    # For scipy modes (mirror, nearest, etc.), no NaN should be present
    else:
        # Check if there are any NaN (shouldn't be with proper scipy modes)
        nan_count = df_smoothed[cols_to_smooth].isnull().sum().sum()
        if nan_count > 0:
            print(f"         ⚠️  Found {nan_count} NaN values, forward filling...")
            df_smoothed = df_smoothed.fillna(method='ffill').fillna(method='bfill')

    return df_smoothed
    
    
def apply_smoothing_to_sequences(sequences, cfg, feature_columns):
    """
    Apply smoothing to 3D sequences array.
    Replicates the exact logic of apply_smoothing_groups + apply_smoothing_with_padding.

    Args:
        sequences: (N, L, F) array
        cfg: configuration object
        feature_columns: list of feature names

    Returns:
        sequences_smoothed: (N, L, F) array
    """
    smooth_cfg = cfg.dataset.get('smooth', None)

    if smooth_cfg is None:
        print(f"      ℹ️  No smoothing configured - returning original data")
        return sequences.copy()

    print(f"      🔧 Applying smoothing to sequences...")

    N, L, F = sequences.shape
    sequences_smoothed = sequences.copy()

    # Get anomaly column index (if exists)
    ano_col = cfg.dataset.get('is_anomaly_column')
    ano_col_idx = feature_columns.index(ano_col) if ano_col and ano_col in feature_columns else None

    # Support both single dict and list of dicts
    if isinstance(smooth_cfg, dict):
        smooth_configs = [smooth_cfg]
    else:
        smooth_configs = list(smooth_cfg)

    smoothed_indices = set()  # Track which feature indices have been smoothed

    for i, smooth_config in enumerate(smooth_configs):
        print(f"\n   📦 Smoothing group {i + 1}/{len(smooth_configs)}:")

        mode = smooth_config.get('mode', 'mean')
        features = smooth_config.get('features', 'all')
        kernel_size = smooth_config.get('kernel_size', 10)
        pad_mode = smooth_config.get('pad_mode', 'drop')

        # Determine feature indices for this group
        if features == 'all' or features == ['all']:
            # All features not yet smoothed
            indices_to_smooth = [idx for idx in range(F) if idx not in smoothed_indices]
        elif isinstance(features, list):
            indices_to_smooth = [
                feature_columns.index(f)
                for f in features
                if f in feature_columns and feature_columns.index(f) not in smoothed_indices
            ]
        else:
            idx = feature_columns.index(features) if features in feature_columns else None
            indices_to_smooth = [idx] if idx is not None and idx not in smoothed_indices else []

        # Remove anomaly column index
        if ano_col_idx is not None and ano_col_idx in indices_to_smooth:
            indices_to_smooth.remove(ano_col_idx)

        if not indices_to_smooth:
            print(f"      ⚠️  No features to smooth in this group")
            continue

        cols_names = [feature_columns[idx] for idx in indices_to_smooth]
        print(f"      - Mode: {mode}")
        print(f"      - Kernel: {kernel_size}")
        print(f"      - Pad mode: {pad_mode}")
        print(f"      - Features: {cols_names}")

        # Apply smoothing with padding strategy (SAME as dataframe)
        sequences_smoothed = apply_smoothing_with_padding_sequences(
            sequences_smoothed,
            indices_to_smooth,
            mode,
            kernel_size,
            pad_mode,
            smooth_config
        )

        # Mark as smoothed
        smoothed_indices.update(indices_to_smooth)

    # Check for unsmoothed features
    unsmoothed_indices = [
        idx for idx in range(F)
        if idx not in smoothed_indices and idx != ano_col_idx
    ]
    if unsmoothed_indices:
        unsmoothed_names = [feature_columns[idx] for idx in unsmoothed_indices]
        print(f"\n   ℹ️  Unsmoothed features: {unsmoothed_names}")

    return sequences_smoothed


def apply_smoothing_with_padding_sequences(sequences, indices_to_smooth, mode, kernel_size, pad_mode, smooth_cfg):
    """
    Apply smoothing with padding strategies to sequences.
    Replicates the exact logic of apply_smoothing_with_padding but for 3D arrays.

    Args:
        sequences: (N, L, F) array
        indices_to_smooth: list of feature indices to smooth
        mode: smoothing mode
        kernel_size: window size
        pad_mode: padding mode
        smooth_cfg: config dict with method-specific params

    Returns:
        sequences_smoothed: (N, L, F) array
    """
    N, L, F = sequences.shape
    sequences_smoothed = sequences.copy()

    # Map pad_mode to scipy mode (SAME as dataframe)
    pad_mode_to_scipy = {
        'reflect': 'mirror',
        'edge': 'nearest',
        'constant': 'constant',
        'wrap': 'wrap',
        'mirror': 'mirror',
        'nearest': 'nearest',
        'drop': 'nearest',
        'ffill': 'nearest'
    }

    scipy_mode = pad_mode_to_scipy.get(pad_mode, 'nearest')
    print(f"         - Padding: {pad_mode} → scipy mode: {scipy_mode}")

    for f_idx in indices_to_smooth:
        try:
            # Process each sequence for this feature
            for seq_idx in range(N):
                signal = sequences[seq_idx, :, f_idx]

                # Apply smoothing based on mode (SAME logic as dataframe)
                if mode == 'mean':
                    if pad_mode == 'ffill':
                        smoothed = pd.Series(signal).rolling(
                            window=kernel_size,
                            min_periods=1,
                            center=True
                        ).mean().values
                    elif pad_mode == 'drop':
                        smoothed = pd.Series(signal).rolling(
                            window=kernel_size,
                            min_periods=kernel_size,
                            center=True
                        ).mean().values
                    else:  # reflect, edge, wrap, etc.
                        pad_width = kernel_size // 2
                        if pad_mode == 'reflect' or pad_mode == 'mirror':
                            padded = np.pad(signal, pad_width, mode='reflect')
                        elif pad_mode == 'edge' or pad_mode == 'nearest':
                            padded = np.pad(signal, pad_width, mode='edge')
                        elif pad_mode == 'wrap':
                            padded = np.pad(signal, pad_width, mode='wrap')
                        else:
                            padded = np.pad(signal, pad_width, mode='edge')

                        smoothed = pd.Series(padded).rolling(
                            window=kernel_size,
                            min_periods=kernel_size,
                            center=True
                        ).mean().values[pad_width:-pad_width]

                elif mode == 'median':
                    smoothed = median_filter(signal, size=kernel_size, mode=scipy_mode)

                elif mode == 'gaussian':
                    sigma = kernel_size / 6
                    smoothed = gaussian_filter1d(signal, sigma, mode=scipy_mode)

                elif mode == 'savgol':
                    polyorder = min(3, kernel_size - 1)
                    if kernel_size % 2 == 0:
                        kernel_size += 1

                    smoothed = savgol_filter(signal, kernel_size, polyorder, mode=scipy_mode)

                elif mode == 'ewm':
                    smoothed = pd.Series(signal).ewm(span=kernel_size, min_periods=1).mean().values

                else:
                    print(f"         ⚠️  Unknown mode '{mode}', using mean")
                    smoothed = pd.Series(signal).rolling(
                        window=kernel_size,
                        min_periods=1,
                        center=True
                    ).mean().values

                sequences_smoothed[seq_idx, :, f_idx] = smoothed

        except Exception as e:
            print(f"         ❌ Error smoothing feature index {f_idx}: {e}")
            print(f"         → Keeping original values for this feature")
            # Keep original values on error
            sequences_smoothed[:, :, f_idx] = sequences[:, :, f_idx]

    # Handle NaN based on pad_mode (SAME as dataframe)
    if pad_mode == 'drop':
        # For sequences, we'd need to identify which timesteps have NaN
        # This is complex, so we'll use ffill instead
        print(f"         ⚠️  'drop' mode not fully supported for sequences, using ffill")
        for f_idx in indices_to_smooth:
            for seq_idx in range(N):
                series = pd.Series(sequences_smoothed[seq_idx, :, f_idx])
                sequences_smoothed[seq_idx, :, f_idx] = series.fillna(method='ffill').fillna(method='bfill').values

    elif pad_mode == 'ffill':
        for f_idx in indices_to_smooth:
            for seq_idx in range(N):
                series = pd.Series(sequences_smoothed[seq_idx, :, f_idx])
                sequences_smoothed[seq_idx, :, f_idx] = series.fillna(method='ffill').fillna(method='bfill').values
        print(f"         → Forward/backward filled NaN values")
    else:
        # Check for NaN (shouldn't be with proper scipy modes)
        nan_mask = np.isnan(sequences_smoothed[:, :, indices_to_smooth])
        nan_count = nan_mask.sum()
        if nan_count > 0:
            print(f"         ⚠️  Found {nan_count} NaN values, forward filling...")
            for f_idx in indices_to_smooth:
                for seq_idx in range(N):
                    series = pd.Series(sequences_smoothed[seq_idx, :, f_idx])
                    sequences_smoothed[seq_idx, :, f_idx] = series.fillna(method='ffill').fillna(method='bfill').values

    return sequences_smoothed


'''