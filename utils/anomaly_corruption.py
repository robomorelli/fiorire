"""
WOMBAT-based anomaly injection for validation sequences.
Uses existing WOMBAT anomaly classes and follows standardization workflow.
"""

import numpy as np
from typing import Tuple, Dict, Optional, List
from omegaconf import DictConfig, ListConfig
from tqdm import tqdm

# Import WOMBAT anomaly classes
from wombats.anomalies.increasing import *
from wombats.anomalies.invariant import *
from wombats.anomalies.decreasing import *

# WOMBAT Registry
ANOMALIES_REGISTRY = {
    'GWN': GWN,
    'Constant': Constant,
    'Step': Step,
    'Impulse': Impulse,
    'GNN': GNN,
}


def print_standardization_summary(sequences: np.ndarray, feature_columns: List[str], title: str = "Data Summary"):
    """
    Print statistical summary to verify standardization.

    Args:
        sequences: [N, L, F] sequences
        feature_columns: Feature names
        title: Title for the summary
    """
    N, L, F = sequences.shape

    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")
    print(f"Shape: {sequences.shape}")
    print(f"\nPer-feature statistics:")
    print(f"{'Feature':<20} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
    print("-" * 80)

    for f_idx, col in enumerate(feature_columns):
        data = sequences[:, :, f_idx].flatten()
        mean = np.mean(data)
        std = np.std(data)
        min_val = np.min(data)
        max_val = np.max(data)

        print(f"{col:<20} {mean:12.6f} {std:12.6f} {min_val:12.6f} {max_val:12.6f}")

    print("="*80 + "\n")


class StandardizationHandler:
    """Handle standardization and inverse transformation."""

    def __init__(self):
        self.mean_dict: Dict[str, float] = {}
        self.std_dict: Dict[str, float] = {}
        self.feature_columns: List[str] = None

    def from_scaler(self, scaler, feature_columns: List[str]):
        """
        Initialize from existing sklearn scaler.

        Args:
            scaler: Fitted sklearn scaler (StandardScaler, MinMaxScaler, etc.)
            feature_columns: Feature names
        """
        self.feature_columns = feature_columns

        # Extract mean and std from scaler
        if hasattr(scaler, 'mean_'):
            # StandardScaler
            for i, col in enumerate(feature_columns):
                self.mean_dict[col] = float(scaler.mean_[i])
                self.std_dict[col] = float(scaler.scale_[i])
        elif hasattr(scaler, 'data_min_'):
            # MinMaxScaler - convert to StandardScaler-like params
            for i, col in enumerate(feature_columns):
                # For MinMaxScaler, we approximate
                data_range = scaler.data_max_[i] - scaler.data_min_[i]
                self.mean_dict[col] = float(scaler.data_min_[i])
                self.std_dict[col] = float(data_range) if data_range > 0 else 1.0
        else:
            raise ValueError(f"Unsupported scaler type: {type(scaler)}")

        print(f"   ✓ Loaded scaler parameters for {len(feature_columns)} features")

    def fit_transform(self, sequences: np.ndarray, feature_columns: List[str]) -> np.ndarray:
        """
        Standardize sequences by fitting on data.

        Args:
            sequences: [N, L, F] raw sequences
            feature_columns: Feature names

        Returns:
            standardized: [N, L, F] standardized sequences
        """
        N, L, F = sequences.shape
        self.feature_columns = feature_columns
        standardized = sequences.copy()

        for f_idx, col in enumerate(feature_columns):
            # Compute mean/std across all sequences
            data = sequences[:, :, f_idx].flatten()
            m = np.mean(data)
            s = np.std(data)

            if s == 0 or np.isnan(s):
                s = 1.0

            self.mean_dict[col] = float(m)
            self.std_dict[col] = float(s)

            # Standardize
            standardized[:, :, f_idx] = (sequences[:, :, f_idx] - m) / s

        print(f"   ✓ Fitted and standardized {F} features")
        return standardized

    def transform(self, sequences: np.ndarray) -> np.ndarray:
        """
        Apply standardization using already fitted parameters.

        Args:
            sequences: [N, L, F] raw sequences

        Returns:
            standardized: [N, L, F] standardized sequences
        """
        if not self.mean_dict or not self.std_dict:
            raise RuntimeError("Handler not fitted! Use fit_transform() or from_scaler() first.")

        N, L, F = sequences.shape
        standardized = sequences.copy()

        for f_idx, col in enumerate(self.feature_columns):
            m = self.mean_dict[col]
            s = self.std_dict[col]
            standardized[:, :, f_idx] = (sequences[:, :, f_idx] - m) / s

        print(f"   ✓ Applied standardization to {F} features")
        return standardized

    def inverse_transform(self, sequences: np.ndarray) -> np.ndarray:
        """
        De-standardize sequences.

        Args:
            sequences: [N, L, F] standardized sequences

        Returns:
            destandardized: [N, L, F] original scale
        """
        if not self.mean_dict or not self.std_dict:
            raise RuntimeError("Handler not fitted!")

        N, L, F = sequences.shape
        destandardized = sequences.copy()

        for f_idx, col in enumerate(self.feature_columns):
            m = self.mean_dict[col]
            s = self.std_dict[col]
            destandardized[:, :, f_idx] = sequences[:, :, f_idx] * s + m

        print(f"   ✓ De-standardized {F} features")
        return destandardized


class WOMBATInjector:
    """
    WOMBAT-style injector with lazy fitting and caching.
    """

    def __init__(self, cfg: DictConfig):
        corruption_cfg = cfg.opt.get('corruption_config', {})

        self.anomalies_type = corruption_cfg.get('anomalies_type', ['GWN', 'Step', 'Impulse'])
        self.delta_mean = corruption_cfg.get('delta_mean', 0.8)
        self.min_channels = corruption_cfg.get('min_channels', 1)
        self.max_channels = corruption_cfg.get('max_channels', 1)
        self.corruption_ratio = corruption_cfg.get('corruption_ratio', 1.0)
        self.random_seed = corruption_cfg.get('random_seed', None)
        self.target_channels = corruption_cfg.get('target_channels', None)

        # Fitted injectors cache: {(channel_idx, anomaly_type): fitted_object}
        self.fitted_cache: Dict[Tuple[int, str], object] = {}

        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        print(f"\n🧪 WOMBAT Injector Configuration:")
        print(f"   - Anomaly types: {self.anomalies_type}")
        print(f"   - Delta: {self.delta_mean}")
        print(f"   - Channels per sequence: {self.min_channels}-{self.max_channels}")
        print(f"   - Corruption ratio: {self.corruption_ratio:.1%}")

    def _get_or_fit(
        self,
        channel_idx: int,
        anomaly_type: str,
        all_sequences_std: np.ndarray,
        window_length: int
    ):
        """
        Get or fit anomaly injector for (channel, type).

        Args:
            channel_idx: Channel index
            anomaly_type: Anomaly type
            all_sequences_std: [N, L, F] all standardized sequences
            window_length: Sequence length

        Returns:
            Fitted anomaly object
        """
        cache_key = (channel_idx, anomaly_type)

        # Check cache
        if cache_key in self.fitted_cache:
            return self.fitted_cache[cache_key]

        # Create new injector
        if anomaly_type not in ANOMALIES_REGISTRY:
            raise ValueError(f"Unknown anomaly type: {anomaly_type}")

        anomaly_cls = ANOMALIES_REGISTRY[anomaly_type]
        anomaly_obj = anomaly_cls(delta=self.delta_mean)

        # Extract channel data: [N, L]
        channel_data = all_sequences_std[:, :, channel_idx]  # [N, L]

        # Fit on ALL sequences
        try:
            anomaly_obj.fit(channel_data)  # WOMBAT fit expects [N, L]
        except Exception as e:
            # Fallback: fit on single sequence
            tqdm.write(f"      ⚠️  Fitting {anomaly_type} on channel {channel_idx} failed: {e}")
            tqdm.write(f"      ↳ Using fallback (single sequence)")
            anomaly_obj.fit(channel_data[:1])

        # Cache
        self.fitted_cache[cache_key] = anomaly_obj

        return anomaly_obj

    def inject(
        self,
        sequences_std: np.ndarray,
        feature_columns: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Inject anomalies into standardized sequences.

        Args:
            sequences_std: [N, L, F] standardized sequences
            feature_columns: Feature names

        Returns:
            corrupted_std: [N, L, F] corrupted standardized sequences
            labels: [N, L, 1] binary labels
            anomaly_types: [N] anomaly type per sequence
            affected_channels: [N] affected channel per sequence
        """
        # ✅ FIX: Convert OmegaConf ListConfig to Python list
        if isinstance(feature_columns, ListConfig):
            feature_columns = list(feature_columns)

        N, L, F = sequences_std.shape

        # Available channels
        if self.target_channels is not None:
            available_channels = [
                feature_columns.index(ch) for ch in self.target_channels
                if ch in feature_columns
            ]
        else:
            available_channels = list(range(F))

        # Initialize outputs
        corrupted_std = sequences_std.copy()
        labels = np.zeros((N, L, 1), dtype=np.float32)
        anomaly_types = np.array(['normal'] * N, dtype=object)
        affected_channels_idx = np.full(N, -1, dtype=np.int32)

        # Select sequences to corrupt
        n_to_corrupt = int(N * self.corruption_ratio)
        indices_to_corrupt = np.random.choice(N, size=n_to_corrupt, replace=False)

        print(f"\n   📊 Injecting anomalies into {n_to_corrupt}/{N} sequences...")

        # Inject with progress bar
        for seq_idx in tqdm(indices_to_corrupt, desc="   💉 Injecting", unit="seq", ncols=100):
            # Choose anomaly type
            anomaly_type = np.random.choice(self.anomalies_type)

            # Choose ONE channel
            channel_idx = np.random.choice(available_channels)

            # Track if this is a new fit
            cache_key = (channel_idx, anomaly_type)
            is_new_fit = cache_key not in self.fitted_cache

            if is_new_fit:
                tqdm.write(f"      🔧 Fitting {anomaly_type} on channel {channel_idx}... ✓")

            # Get fitted injector
            injector = self._get_or_fit(
                channel_idx=channel_idx,
                anomaly_type=anomaly_type,
                all_sequences_std=sequences_std,
                window_length=L
            )

            # Extract sequence for this channel
            original_seq = sequences_std[seq_idx, :, channel_idx].copy()  # [L]

            # Inject anomaly
            try:
                distorted_seq = injector.distort(original_seq.reshape(1, -1))[0]  # WOMBAT expects [1, L]
            except Exception as e:
                tqdm.write(f"      ⚠️  Injection failed for seq {seq_idx}: {e}")
                distorted_seq = original_seq.copy()

            # Compute mask (where changed)
            mask = (original_seq != distorted_seq)

            if mask.any():
                # Update corrupted sequences
                corrupted_std[seq_idx, :, channel_idx] = distorted_seq

                # Update labels
                labels[seq_idx, mask, 0] = 1

                # Update metadata
                anomaly_types[seq_idx] = anomaly_type
                affected_channels_idx[seq_idx] = channel_idx

        # Summary
        total_anomalous = labels.sum()
        print(f"\n   ✅ Injection complete:")
        print(f"      - Corrupted sequences: {n_to_corrupt}/{N}")
        print(f"      - Unique injectors fitted: {len(self.fitted_cache)}")
        print(f"      - Anomalous timesteps: {total_anomalous:.0f}/{N*L} ({total_anomalous/(N*L)*100:.1f}%)")

        # ✅ FIX: Cast to int() for OmegaConf compatibility
        affected_channels_names = np.array([
            feature_columns[int(idx)] if idx >= 0 else 'none'
            for idx in affected_channels_idx
        ], dtype=object)

        return corrupted_std, labels, anomaly_types, affected_channels_names


def corrupt_sequences_wombat(
    sequences: np.ndarray,
    feature_columns: List[str],
    cfg: DictConfig,
    standardization_mode: str = "skip",
    scaler: Optional[object] = None,
    random_seed: Optional[int] = None,
    force_destandardization: bool = False,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    """
    Complete WOMBAT corruption pipeline with flexible standardization.

    Args:
        sequences: [N, L, F] raw sequences
        feature_columns: Feature names
        cfg: Configuration
        standardization_mode:
            - "fit_new": Fit new scaler on these sequences
            - "use_provided": Use provided scaler parameters
            - "skip": Assume sequences already standardized, no transform
        scaler: Fitted scaler object (required if mode="use_provided")
        random_seed: Random seed
        force_destandardization: If True with mode="skip", force de-standardization
        verbose: Print statistical summaries

    Returns:
        corrupted_sequences: [N, L, F] corrupted sequences
        labels: [N, L, 1] binary labels
        anomaly_types: [N] anomaly type per sequence
        affected_channels: [N] affected channel names
        is_standardized: True if output is still standardized
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    print(f"\n{'='*80}")
    print("WOMBAT CORRUPTION PIPELINE")
    print(f"{'='*80}")
    print(f"Input: {sequences.shape} sequences")
    print(f"Standardization mode: {standardization_mode}")
    if standardization_mode == "skip" and force_destandardization:
        print(f"Force de-standardization: True")

    # Summary BEFORE standardization
    if verbose:
        print_standardization_summary(sequences, feature_columns, "BEFORE Standardization")

    handler = StandardizationHandler()

    # =========================================================
    # Step 1: Standardization
    # =========================================================
    if standardization_mode == "fit_new":
        print(f"\n📊 Step 1: Standardization (fitting new scaler)")
        sequences_std = handler.fit_transform(sequences, feature_columns)

    elif standardization_mode == "use_provided":
        if scaler is None:
            raise ValueError("scaler must be provided when mode='use_provided'")

        print(f"\n📊 Step 1: Standardization (using provided scaler)")
        handler.from_scaler(scaler, feature_columns)
        sequences_std = handler.transform(sequences)

    elif standardization_mode == "skip":
        print(f"\n📊 Step 1: Standardization (skipped - assuming already standardized)")
        sequences_std = sequences.copy()
        # Still need scaler for de-standardization later
        if scaler is not None:
            handler.from_scaler(scaler, feature_columns)
        else:
            if force_destandardization:
                raise ValueError("force_destandardization=True requires scaler to be provided")
            print("   ⚠️  WARNING: No scaler provided - de-standardization will not be possible")

    else:
        raise ValueError(f"Unknown standardization_mode: {standardization_mode}")

    # Summary AFTER standardization
    if verbose:
        print_standardization_summary(sequences_std, feature_columns, "AFTER Standardization (should be ~mean=0, std=1)")

    # =========================================================
    # Step 2: Inject (on standardized data)
    # =========================================================
    print(f"\n💉 Step 2: Anomaly Injection")
    injector = WOMBATInjector(cfg)
    corrupted_std, labels, anomaly_types, affected_channels = injector.inject(
        sequences_std,
        feature_columns
    )

    # Summary AFTER injection (standardized)
    if verbose:
        print_standardization_summary(corrupted_std, feature_columns, "AFTER Injection (still standardized)")

    # =========================================================
    # Step 3: De-standardization & Preserve
    # =========================================================
    is_standardized = False

    if standardization_mode in ["fit_new", "use_provided"]:
        print(f"\n📈 Step 3: De-standardization")
        corrupted_sequences = handler.inverse_transform(corrupted_std)

        # ✅ Preserve original values for normal timesteps
        print(f"   ↳ Preserving original values for normal timesteps...")
        normal_mask = (labels[:, :, 0] == 0)
        for f_idx in range(sequences.shape[2]):
            corrupted_sequences[:, :, f_idx][normal_mask] = sequences[:, :, f_idx][normal_mask]

        is_standardized = False

    elif standardization_mode == "skip":
        if force_destandardization and scaler is not None:
            print(f"\n📈 Step 3: De-standardization (forced)")

            # De-standardizza TUTTO (corrupted E sequences originali)
            corrupted_sequences = handler.inverse_transform(corrupted_std)
            sequences_destd = handler.inverse_transform(sequences)

            # ✅ Preserve (ora entrambi in scala originale)
            print(f"   ↳ Preserving original values for normal timesteps...")
            normal_mask = (labels[:, :, 0] == 0)
            for f_idx in range(sequences.shape[2]):
                corrupted_sequences[:, :, f_idx][normal_mask] = sequences_destd[:, :, f_idx][normal_mask]

            is_standardized = False
        else:
            print(f"\n📈 Step 3: De-standardization (skipped - keeping standardized)")
            corrupted_sequences = corrupted_std

            # ✅ Preserve (entrambi in scala standardizzata - safety net)
            print(f"   ↳ Preserving original values for normal timesteps (safety check)...")
            normal_mask = (labels[:, :, 0] == 0)
            for f_idx in range(sequences.shape[2]):
                corrupted_sequences[:, :, f_idx][normal_mask] = sequences[:, :, f_idx][normal_mask]

            is_standardized = True

    # Summary AFTER de-standardization
    if verbose and not is_standardized:
        print_standardization_summary(corrupted_sequences, feature_columns,
                                     "AFTER De-standardization (back to original scale)")

    print(f"\n{'='*80}")
    print("CORRUPTION COMPLETE")
    if is_standardized:
        print("⚠️  OUTPUT IS STILL STANDARDIZED")
    else:
        print("✓ OUTPUT IS IN ORIGINAL SCALE")
    print(f"{'='*80}\n")

    return corrupted_sequences, labels, anomaly_types, affected_channels, is_standardized