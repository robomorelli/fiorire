import numpy as np
import pandas as pd
from typing import List, Dict
import argparse
import os
from omegaconf import OmegaConf
from anomalies_injection.utils import load_data
from anomalies_injection.utils import ANOMALIES_REGISTRY, make_json_safe, to_json_serializable
import json
from datetime import datetime

"""
WOMBAT-Compliant MULTI-CHANNEL Anomaly Injection Pipeline

Key feature: Can inject anomalies across MULTIPLE channels simultaneously.
Uses LAZY FITTING for memory efficiency (fits on-demand, not pre-fitting).

Flow:
1. Load data + global standardization
2. Random window selection + random MULTI-CHANNEL selection + random anomaly assignment
3. LAZY FIT on-demand (only when an anomaly-channel combination is needed)
4. Insert anomalies into dataframe (same anomaly across selected channels)
5. De-standardize and save

Memory efficiency: Instead of pre-fitting N×M combinations, only fits what's actually used.
Example: With 16 channels × 5 anomalies = 80 possible combinations, but only ~30 might be used.
"""


# =======================================================================
# STANDARDIZATION HANDLER
# =======================================================================
class StandardizationHandler:
    """Z-score standardization with exact restoration of non-anomalous values"""

    def __init__(self):
        self.mean_dict = {}
        self.std_dict = {}
        self.feature_columns = None

    def fit_transform(self, df: pd.DataFrame, feature_columns: list = None) -> pd.DataFrame:
        """Standardize numeric columns (z-score: mean=0, std=1)"""
        df_standardized = df.copy()
        if feature_columns is None:
            feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        self.feature_columns = feature_columns

        for col in feature_columns:
            self.mean_dict[col] = df[col].mean()
            self.std_dict[col] = df[col].std()
            df_standardized[col] = (df[col] - self.mean_dict[col]) / self.std_dict[col]

        print(f"\n✓ Standardization completed for {len(feature_columns)} features")
        for col in feature_columns[:3]:
            print(f"    - {col}: μ={self.mean_dict[col]:.4f}, σ={self.std_dict[col]:.4f}")
        if len(feature_columns) > 3:
            print(f"    ... and {len(feature_columns) - 3} more features")

        return df_standardized

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reverse z-score normalization"""
        df_destandardized = df.copy()
        for col in self.feature_columns:
            df_destandardized[col] = (df[col] * self.std_dict[col]) + self.mean_dict[col]
        print(f"\n✓ De-standardization completed")
        return df_destandardized


# =======================================================================
# MULTI-CHANNEL WOMBAT ANOMALY INJECTOR
# =======================================================================
class MultiChannelWombatAnomalyInjector:
    """
    WOMBAT-style anomaly injection with MULTI-CHANNEL support.

    Can inject the same anomaly type across multiple channels simultaneously
    in the same time window.

    Process:
    1. Extract ALL windows from standardized data for fitting
    2. Fit all anomaly types on all channels (N×M fittings)
    3. Randomly select windows to make anomalous
    4. For each window:
       - Randomly select 1-K channels (min_channels to max_channels)
       - Randomly select an anomaly type
       - Apply SAME anomaly to ALL selected channels
       - Reinsert distorted windows
    """

    def __init__(
            self,
            anomaly_types: List[str],
            anomaly_percentage: float,
            window_length: int,
            delta: float,
            min_channels: int = 1,
            max_channels: int = 4,
            channel_prob_decay: float = 0.7,
            random_seed: int = None,
            anomaly_registry: dict = None
    ):
        """
        Parameters
        ----------
        anomaly_types : List[str]
            List of anomaly types to use (e.g., ['GWN', 'Constant', 'Step'])
        anomaly_percentage : float
            Percentage of points to make anomalous
        window_length : int
            Fixed window length
        delta : float
            Deviation intensity
        min_channels : int
            Minimum number of channels to perturb per window
        max_channels : int
            Maximum number of channels to perturb per window
        channel_prob_decay : float
            Probability for 1 channel (e.g., 0.7 = 70% chance of 1 channel)
            Remaining probability (0.3) is distributed exponentially across 2+ channels
        random_seed : int
            For reproducibility
        """
        self.anomaly_types = anomaly_types
        self.anomaly_percentage = anomaly_percentage
        self.window_length = window_length
        self.delta = delta
        self.min_channels = min_channels
        self.max_channels = max_channels
        self.channel_prob_decay = channel_prob_decay
        self.random_seed = random_seed

        # Get anomaly registry
        self.anomaly_registry = anomaly_registry if anomaly_registry is not None else ANOMALIES_REGISTRY

        # Validate anomaly types
        for anom_type in anomaly_types:
            if anom_type not in self.anomaly_registry:
                raise ValueError(f"Anomaly type '{anom_type}' not found in registry!")

        if random_seed is not None:
            np.random.seed(random_seed)

        # Storage for fitted anomalies: fitted_anomalies[channel][anomaly_type] = fitted_obj
        self.fitted_anomalies: Dict[str, Dict[str, object]] = {}

        print(f"\n✓ Multi-Channel Anomaly Injector initialized")
        print(f"  - Types: {anomaly_types}")
        print(f"  - Window length: {window_length}")
        print(f"  - Delta: {delta:.2f}")
        print(f"  - Target anomaly %: {anomaly_percentage:.1f}%")
        print(f"  - Channels per window: {min_channels}-{max_channels}")
        print(f"  - P(1 channel): {channel_prob_decay:.0%}, P(2+ channels): {1 - channel_prob_decay:.0%}")

    def _select_num_channels(self) -> int:
        """
        Probabilistically determine how many channels to perturb.

        Distribution:
        - P(1 channel) = channel_prob_decay (e.g., 0.7 = 70%)
        - P(2+ channels) = 1 - channel_prob_decay (e.g., 0.3 = 30%)
        - Among 2+ channels, exponential decay with factor (1 - channel_prob_decay)

        Example with channel_prob_decay=0.7, max_channels=4:
        - P(1) = 0.70 (70.0%)
        - P(2) = 0.30 × 0.70 = 0.21 (21.0%)
        - P(3) = 0.30 × 0.30 × 0.70 ≈ 0.063 (6.3%)
        - P(4) = 0.30 × 0.30 × 0.30 ≈ 0.027 (2.7%)

        Returns
        -------
        int
            Number of channels to perturb (between 1 and max_channels)
        """
        # First roll: decide if we use exactly 1 channel
        if np.random.random() < self.channel_prob_decay:
            return 1

        # If not 1 channel, start from 2 and probabilistically add more
        # Each additional channel has probability (1 - channel_prob_decay)
        n_channels = 2
        remaining_prob = 1 - self.channel_prob_decay

        # Decide between 2, 3, or 4 channels with exponential decay
        for _ in range(self.max_channels - 2):
            if np.random.random() < remaining_prob:
                n_channels += 1
            else:
                break

        return min(n_channels, self.max_channels)

    def _extract_all_windows_for_channel(
            self,
            data: np.ndarray,
            channel_name: str
    ) -> np.ndarray:
        """
        Extract all non-overlapping windows for a single channel.

        WOMBAT requirement: Data is already globally standardized (mean≈0, std≈1).
        Do NOT normalize each window individually!

        Parameters
        ----------
        data : np.ndarray
            1D array of GLOBALLY standardized channel data
        channel_name : str
            Channel name (for logging)

        Returns
        -------
        np.ndarray
            Shape [N, window_length] - windows extracted as-is (already standardized)
        """
        n_points = len(data)
        n_windows = n_points // self.window_length

        # Extract windows WITHOUT additional normalization
        windows = np.array([
            data[i * self.window_length:(i + 1) * self.window_length]
            for i in range(n_windows)
        ])

        # Verify WOMBAT requirements on the ENTIRE set of windows
        global_mean = windows.mean()
        global_std = windows.std()

        print(f"    ✓ {channel_name}: {n_windows} windows, shape {windows.shape}")
        print(f"      Global mean: {global_mean:.6f}, std: {global_std:.6f}")

        return windows

    def _get_or_fit_anomaly(
            self,
            channel: str,
            anomaly_type: str,
            df_standardized: pd.DataFrame
    ) -> object:
        """
        LAZY FITTING: Get fitted anomaly object, fitting on-demand if needed.

        This is more memory-efficient than pre-fitting all N×M combinations.
        Only fits anomaly types that are actually used.

        Parameters
        ----------
        channel : str
            Channel name
        anomaly_type : str
            Anomaly type name
        df_standardized : pd.DataFrame
            Globally standardized dataframe

        Returns
        -------
        object
            Fitted anomaly object ready to use .distort()
        """
        # Check if already fitted
        if channel in self.fitted_anomalies:
            if anomaly_type in self.fitted_anomalies[channel]:
                return self.fitted_anomalies[channel][anomaly_type]
        else:
            self.fitted_anomalies[channel] = {}

        # Need to fit: extract windows for this channel
        print(f"      → Fitting {anomaly_type} on {channel} (on-demand)...")

        data = df_standardized[channel].values
        windows = self._extract_all_windows_for_channel(data, channel)

        # Create and fit anomaly
        anomaly_class = self.anomaly_registry[anomaly_type]
        anomaly_obj = anomaly_class(self.delta)
        anomaly_obj.fit(windows)

        # Cache for future use
        self.fitted_anomalies[channel][anomaly_type] = anomaly_obj

        return anomaly_obj

    def inject_anomalies(
            self,
            df_standardized: pd.DataFrame,
            feature_columns: List[str]
    ) -> pd.DataFrame:
        """
        STEP 2: Inject MULTI-CHANNEL anomalies with LAZY FITTING

        Process:
        1. Calculate how many windows to make anomalous
        2. Randomly select windows
        3. For each selected window:
           - Randomly select K channels (min_channels to max_channels)
           - Randomly select an anomaly type
           - LAZY FIT if needed (on-demand)
           - Apply SAME anomaly to ALL K channels in this window
           - Extract window, apply distortion, reinsert

        Parameters
        ----------
        df_standardized : pd.DataFrame
            Globally standardized dataframe
        feature_columns : List[str]
            List of feature columns

        Returns
        -------
        pd.DataFrame
            Dataframe with anomalies injected (still standardized)
        """
        print(f"\n{'=' * 70}")
        print("MULTI-CHANNEL ANOMALY INJECTION (with lazy fitting)")
        print('=' * 70)

        df_result = df_standardized.copy()
        df_result['is_anomaly'] = 0
        df_result['anomaly_type'] = ''
        df_result['affected_channels'] = ''

        n_points = len(df_result)
        n_windows = n_points // self.window_length

        # Calculate target number of anomalous windows
        n_anomalous_points_target = int(n_points * self.anomaly_percentage / 100)
        n_anomalous_windows_target = max(1, n_anomalous_points_target // self.window_length)

        # Don't exceed total available windows
        n_anomalous_windows = min(n_anomalous_windows_target, n_windows)

        print(f"\n  Dataset info:")
        print(f"    - Total points: {n_points}")
        print(f"    - Total windows: {n_windows}")
        print(f"    - Target anomalous windows: {n_anomalous_windows}")

        # Randomly select which windows to make anomalous
        window_indices = np.arange(n_windows)
        np.random.shuffle(window_indices)
        selected_window_indices = window_indices[:n_anomalous_windows]

        print(f"\n  Injecting multi-channel anomalies into {n_anomalous_windows} windows...")
        print(f"  (Fitting anomalies on-demand as needed)")

        anomaly_stats = {anom_type: 0 for anom_type in self.anomaly_types}
        channel_stats = {channel: 0 for channel in feature_columns}
        multi_channel_distribution = {i: 0 for i in range(1, self.max_channels + 1)}

        # Detailed anomaly log for JSON export
        anomalies_log = []

        # Process each selected window
        for win_idx in selected_window_indices:
            # Calculate position in dataframe
            start_idx = win_idx * self.window_length
            end_idx = start_idx + self.window_length

            # Randomly determine how many channels to perturb
            n_channels_to_perturb = self._select_num_channels()
            n_channels_to_perturb = min(n_channels_to_perturb, len(feature_columns))

            # Randomly select K channels WITHOUT replacement
            selected_channels = np.random.choice(
                feature_columns,
                size=n_channels_to_perturb,
                replace=False
            ).tolist()

            # Randomly select ONE anomaly type (same for all selected channels)
            anomaly_type = np.random.choice(self.anomaly_types)

            # Track for statistics
            multi_channel_distribution[n_channels_to_perturb] += 1
            anomaly_stats[anomaly_type] += 1

            # Apply the SAME anomaly to ALL selected channels
            distorted_successfully = []

            for channel in selected_channels:
                try:
                    # LAZY FIT: Get or fit anomaly on-demand
                    anomaly_obj = self._get_or_fit_anomaly(channel, anomaly_type, df_standardized)

                    # Extract the window from standardized data
                    # WOMBAT: Use globally standardized data as-is, NO per-window normalization!
                    window_data = df_result[channel].values[start_idx:end_idx]

                    # Apply distortion (WOMBAT expects [M, Length], we have [1, Length])
                    # Window is already in the correct standardized space
                    window_distorted = anomaly_obj.distort(window_data.reshape(1, -1))[0]

                    # Reinsert into dataframe (use .iloc for positional indexing)
                    col_idx = df_result.columns.get_loc(channel)
                    df_result.iloc[start_idx:end_idx, col_idx] = window_distorted

                    # Update channel statistics
                    channel_stats[channel] += 1
                    distorted_successfully.append(channel)

                except Exception as e:
                    print(f"    ✗ Error injecting {anomaly_type} in {channel} at window {win_idx}: {e}")

            # Mark as anomalous (only if at least one channel was distorted)
            if len(distorted_successfully) > 0:
                anomaly_col_idx = df_result.columns.get_loc('is_anomaly')
                type_col_idx = df_result.columns.get_loc('anomaly_type')
                channels_col_idx = df_result.columns.get_loc('affected_channels')

                df_result.iloc[start_idx:end_idx, anomaly_col_idx] = 1
                df_result.iloc[start_idx:end_idx, type_col_idx] = anomaly_type
                df_result.iloc[start_idx:end_idx, channels_col_idx] = ','.join(distorted_successfully)

                # Log detailed information for JSON export
                anomaly_record = {
                    'anomaly_id': len(anomalies_log),
                    'window_index': int(win_idx),
                    'start_index': int(start_idx),
                    'end_index': int(end_idx),
                    'anomaly_type': anomaly_type,
                    'affected_channels': distorted_successfully,
                    'n_channels_affected': len(distorted_successfully),
                    'window_length': int(self.window_length),
                    'delta': float(self.delta)
                }
                anomalies_log.append(anomaly_record)

        # Report statistics
        total_anomalous_points = (df_result['is_anomaly'] == 1).sum()
        actual_percentage = (total_anomalous_points / n_points) * 100

        # Report how many anomalies were actually fitted
        total_fitted = sum(len(anomalies) for anomalies in self.fitted_anomalies.values())
        max_possible = len(self.anomaly_types) * len(feature_columns)

        print(f"\n  Injection complete:")
        print(f"    - Anomalous points: {total_anomalous_points} ({actual_percentage:.2f}%)")
        print(
            f"    - Fitted anomalies: {total_fitted}/{max_possible} combinations ({total_fitted / max_possible * 100:.1f}%)")

        print(f"\n  Anomaly type distribution:")
        for anom_type, count in anomaly_stats.items():
            print(f"    - {anom_type}: {count} windows")

        print(f"\n  Channel distribution (total perturbations):")
        for channel, count in sorted(channel_stats.items(), key=lambda x: x[1], reverse=True):
            print(f"    - {channel}: {count} windows")

        print(f"\n  Multi-channel distribution:")
        for n_ch, count in sorted(multi_channel_distribution.items()):
            if count > 0:
                pct = (count / n_anomalous_windows) * 100
                print(f"    - {n_ch} channel(s): {count} windows ({pct:.1f}%)")

        return df_result, anomalies_log

    def run_pipeline(
            self,
            df_standardized: pd.DataFrame,
            feature_columns: List[str]
    ) -> tuple:
        """
        Complete pipeline with LAZY FITTING

        No pre-fitting phase - anomalies are fitted on-demand as needed.
        This is much more memory-efficient for large datasets.

        Parameters
        ----------
        df_standardized : pd.DataFrame
            Globally standardized dataframe
        feature_columns : List[str]
            List of feature columns

        Returns
        -------
        tuple
            (df_with_anomalies: pd.DataFrame, anomalies_log: list)
        """
        # Directly inject with lazy fitting
        df_with_anomalies, anomalies_log = self.inject_anomalies(df_standardized, feature_columns)

        return df_with_anomalies, anomalies_log


# =======================================================================
# MAIN PIPELINE
# =======================================================================
def main(args):
    """
    Complete workflow:
    1. Load data
    2. Global standardization
    3. Lazy fitting on demand
    4. Random MULTI-CHANNEL injection
    5. De-standardization
    6. Restore original non-anomalous values
    7. Save
    """
    # Load configuration
    cfg = OmegaConf.load(args.conf_file)
    data_path = cfg.dataset.data_path

    print(f"\n{'=' * 70}")
    print("WOMBAT MULTI-CHANNEL ANOMALY INJECTION PIPELINE")
    print('=' * 70)

    # ===================================================================
    # STEP 1: Load dataset
    # ===================================================================
    print(f"\n{'=' * 70}")
    print("STEP 1: LOAD DATA")
    print('=' * 70)

    df_original = load_data(cfg)
    print(f"✓ Loaded {df_original.shape[0]} rows × {df_original.shape[1]} cols")

    # Backup for later restoration
    df_backup = df_original.copy()
    features = cfg.dataset.feats

    # ===================================================================
    # STEP 2: Global standardization
    # ===================================================================
    print(f"\n{'=' * 70}")
    print("STEP 2: GLOBAL STANDARDIZATION")
    print('=' * 70)

    # Check statistics BEFORE standardization
    print(f"\nStatistics BEFORE standardization:")
    for col in features[:min(3, len(features))]:
        mean_before = df_original[col].mean()
        std_before = df_original[col].std()
        min_before = df_original[col].min()
        max_before = df_original[col].max()
        print(f"  - {col}:")
        print(f"      mean={mean_before:.4f}, std={std_before:.4f}")
        print(f"      range=[{min_before:.4f}, {max_before:.4f}]")
    if len(features) > 3:
        print(f"    ... and {len(features) - 3} more features")

    handler = StandardizationHandler()
    df_standardized = handler.fit_transform(df_original, feature_columns=features)

    # Verify
    print(f"\nVerification (first 3 features):")
    for col in features[:min(3, len(features))]:
        print(f"  - {col}: mean={df_standardized[col].mean():.6f}, std={df_standardized[col].std():.6f}")

    # ===================================================================
    # STEP 3: WOMBAT Multi-Channel Pipeline (LAZY FITTING + INJECTION)
    # ===================================================================
    print(f"\n{'=' * 70}")
    print("STEP 3: WOMBAT MULTI-CHANNEL PIPELINE (with lazy fitting)")
    print('=' * 70)

    injector = MultiChannelWombatAnomalyInjector(
        anomaly_types=cfg.dataset.anomalies_type,
        anomaly_percentage=cfg.dataset.anomaly_percentage,
        window_length=cfg.dataset.window_mean,
        delta=cfg.dataset.delta_mean,
        min_channels=cfg.dataset.get('min_channels', 1),
        max_channels=cfg.dataset.get('max_channels', 4),
        channel_prob_decay=cfg.dataset.get('channel_prob_decay', 0.7),
        random_seed=cfg.dataset.random_seed,
        anomaly_registry=ANOMALIES_REGISTRY
    )

    df_with_anomalies_std, anomalies_log = injector.run_pipeline(df_standardized, features)

    # ===================================================================
    # STEP 4: De-standardization
    # ===================================================================
    print(f"\n{'=' * 70}")
    print("STEP 4: DE-STANDARDIZATION")
    print('=' * 70)

    df_destandardized = handler.inverse_transform(df_with_anomalies_std)

    # ===================================================================
    # STEP 5: Restore original non-anomalous values
    # ===================================================================
    print(f"\n{'=' * 70}")
    print("STEP 5: RESTORE ORIGINAL NON-ANOMALOUS VALUES")
    print('=' * 70)

    anomaly_mask = df_with_anomalies_std['is_anomaly'].astype(bool)

    for col in features:
        df_destandardized.loc[~anomaly_mask, col] = df_backup.loc[~anomaly_mask, col]

    print(f"✓ Original values preserved for non-anomalous points")

    # Verification
    print(f"\nVerification (first 3 features):")
    for col in features[:min(3, len(features))]:
        non_anomalous_identical = np.allclose(
            df_destandardized.loc[~anomaly_mask, col].values,
            df_backup.loc[~anomaly_mask, col].values,
            rtol=1e-10
        )
        if non_anomalous_identical:
            print(f"  ✓ {col}: Non-anomalous values IDENTICAL to original")
        else:
            max_diff = np.max(np.abs(
                df_destandardized.loc[~anomaly_mask, col].values -
                df_backup.loc[~anomaly_mask, col].values
            ))
            print(f"  ⚠ {col}: Max difference = {max_diff:.2e}")

    # ===================================================================
    # STEP 6: Save final dataset
    # ===================================================================
    print(f"\n{'=' * 70}")
    print("STEP 6: STATISTICAL VALIDATION & SAVE")
    print('=' * 70)

    # Check statistics AFTER de-standardization (with anomalies)
    print(f"\nStatistics AFTER de-standardization (with anomalies):")
    for col in features[:min(3, len(features))]:
        mean_after = df_destandardized[col].mean()
        std_after = df_destandardized[col].std()
        min_after = df_destandardized[col].min()
        max_after = df_destandardized[col].max()

        # Compare with original statistics
        mean_original = df_backup[col].mean()
        std_original = df_backup[col].std()

        mean_diff = abs(mean_after - mean_original)
        std_diff = abs(std_after - std_original)

        print(f"  - {col}:")
        print(f"      mean={mean_after:.4f} (Δ={mean_diff:.4f}), std={std_after:.4f} (Δ={std_diff:.4f})")
        print(f"      range=[{min_after:.4f}, {max_after:.4f}]")

        # Warning if statistics changed too much (>10% for mean, >20% for std)
        if (mean_diff / abs(mean_original) > 0.1) if mean_original != 0 else (mean_diff > 0.1):
            print(f"      ⚠ WARNING: Mean changed by {mean_diff / abs(mean_original) * 100:.1f}%")
        if (std_diff / std_original > 0.2) if std_original != 0 else (std_diff > 0.2):
            print(f"      ⚠ WARNING: Std changed by {std_diff / std_original * 100:.1f}%")

    if len(features) > 3:
        print(f"    ... and {len(features) - 3} more features")

    n_points = len(df_backup)
    n_total = len(df_destandardized)
    n_anomalous = anomaly_mask.sum()

    print(f"\nℹ Small changes in statistics are EXPECTED due to anomalies ({n_anomalous / n_total * 100:.2f}% of data)")


    # ===================================================================
    # Save detailed anomalies log as JSON
    # ===================================================================
    dir_path = os.path.dirname(data_path)
    base_name, ext = os.path.splitext(os.path.basename(data_path))
    output_path = os.path.join(dir_path, f"{base_name}_with_anomalies{ext}")

    # Create comprehensive metadata
    anomalies_metadata = {
        'metadata': {
            'creation_date': datetime.now().isoformat(),
            'dataset_name': cfg.dataset.name,
            'original_data_path': data_path,
            'output_data_path': output_path,
            'total_points': int(n_total),
            'total_windows': int(n_points // cfg.dataset.window_mean),
            'window_length': int(cfg.dataset.window_mean),
            'features': features,
            'n_features': len(features),
            'random_seed': cfg.dataset.random_seed
        },
        'injection_config': {
            'anomaly_types': cfg.dataset.anomalies_type,
            'anomaly_percentage_target': float(cfg.dataset.anomaly_percentage),
            'anomaly_percentage_actual': float(n_anomalous / n_total * 100),
            'delta': float(cfg.dataset.delta_mean),
            'min_channels': cfg.dataset.get('min_channels', 1),
            'max_channels': cfg.dataset.get('max_channels', 4),
            'channel_prob_decay': cfg.dataset.get('channel_prob_decay', 0.7)
        },
        'summary': {
            'total_anomalies_injected': len(anomalies_log),
            'total_anomalous_points': int(n_anomalous),
            'anomaly_type_distribution': {
                anom_type: sum(1 for a in anomalies_log if a['anomaly_type'] == anom_type)
                for anom_type in cfg.dataset.anomalies_type
            },
            'channel_distribution': {
                channel: sum(1 for a in anomalies_log if channel in a['affected_channels'])
                for channel in features
            },
            'multi_channel_distribution': {
                f'{i}_channels': sum(1 for a in anomalies_log if a['n_channels_affected'] == i)
                for i in range(1, cfg.dataset.get('max_channels', 4) + 1)
            }
        },
        'anomalies': anomalies_log
    }


    # save same format as input
    if ext == ".pkl":
        df_destandardized.to_pickle(output_path)
    else:
        df_destandardized.to_csv(output_path, index=False)
    print(f"✓ Saved dataset: {output_path}")

    # --- Convert and save metadata ---
    anomalies_metadata_clean = to_json_serializable(anomalies_metadata)
    json_path = os.path.join(dir_path, f"{base_name}_anomalies_info.json")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(anomalies_metadata_clean, f, indent=2, ensure_ascii=False)

    print(f"✓ Saved anomalies metadata: {json_path}")

    print(f"✓ Saved: {output_path}")

    print(f"✓ Saved anomalies info: {json_path}")
    print(f"  - Total anomalies logged: {len(anomalies_log)}")

    # ===================================================================
    # STEP 7: Final statistical comparison
    # ===================================================================
    print(f"\n{'=' * 70}")
    print("STATISTICAL COMPARISON: Original vs Final (with anomalies)")
    print('=' * 70)

    print(
        f"\n{'Channel':<20} {'Original Mean':<15} {'Final Mean':<15} {'Δ Mean':<10} "
        f"{'Original Std':<15} {'Final Std':<15} {'Δ Std':<10} {'Anomalous Points':<20}"
    )
    print("-" * 130)

    max_mean_change = 0
    max_std_change = 0

    # Compute anomaly counts per feature
    anomaly_counts = {}
    for col in features:
        anomaly_counts[col] = int(df_destandardized.loc[df_destandardized['is_anomaly'] == 1, col].count())

    for col in features:
        mean_orig = df_backup[col].mean()
        mean_final = df_destandardized[col].mean()
        std_orig = df_backup[col].std()
        std_final = df_destandardized[col].std()

        mean_change = abs(mean_final - mean_orig) / abs(mean_orig) * 100 if mean_orig != 0 else 0
        std_change = abs(std_final - std_orig) / std_orig * 100 if std_orig != 0 else 0

        max_mean_change = max(max_mean_change, mean_change)
        max_std_change = max(max_std_change, std_change)

        print(
            f"{col:<20} "
            f"{mean_orig:<15.4f} {mean_final:<15.4f} {mean_change:<9.2f}% "
            f"{std_orig:<15.4f} {std_final:<15.4f} {std_change:<9.2f}% "
            f"{anomaly_counts[col]:<20}"
        )

    print("-" * 130)
    print(f"\nMaximum changes: Mean={max_mean_change:.2f}%, Std={max_std_change:.2f}%")
    print(f"ℹ These changes are due to the {n_anomalous / n_total * 100:.2f}% anomalous points injected")

    print(f"\n{'=' * 70}")
    print("✓ PIPELINE COMPLETED SUCCESSFULLY")
    print('=' * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="WOMBAT-style MULTI-CHANNEL anomaly injection"
    )
    parser.add_argument(
        '--conf_file', '-c',
        type=str,
        default='./dataset_configuration/fiorire_1.yaml',
        help='Path to configuration file'
    )
    args = parser.parse_args()
    main(args)