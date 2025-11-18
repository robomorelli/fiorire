#!/usr/bin/env python3
"""
Adaptive WOMBAT-style multi-channel anomaly injector
- Delta & window adaptive schedule with resets at configurable percentages
- Lazy fitting of anomaly objects (fitted on-demand)
- Saves .pkl if input was .pkl
- Final statistics table includes anomalous point counts per feature
"""

# Your project utilities (assumed available)
from anomalies_injection.utils import load_data, ANOMALIES_REGISTRY, make_json_safe, sample_and_plot_anomalies_with_labels

import os
import json
import random
import argparse
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
import shutil

# External libs referenced in original script (placeholders if not imported elsewhere)
from omegaconf import OmegaConf
from omegaconf import DictConfig, ListConfig

# --- PLACEHOLDERS / EXPECTED TO BE PROVIDED IN YOUR ENVIRONMENT ---
# ANOMALIES_REGISTRY, load_data, sample_and_plot_anomalies should be available
# in your codebase exactly as they were in the original project. If not, import
# them appropriately.


# -------------------------
# Standardization Handler
# -------------------------
class StandardizationHandler:
    """
    Z-score standardization with exact restoration of non-anomalous values.
    Stores per-feature mean/std from the original DataFrame.
    """

    def __init__(self):
        self.mean_dict: Dict[str, float] = {}
        self.std_dict: Dict[str, float] = {}
        self.feature_columns: List[str] = None

    def fit_transform(self, df: pd.DataFrame, feature_columns: List[str] = None) -> pd.DataFrame:
        df_std = df.copy(deep=True)
        if feature_columns is None:
            feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        self.feature_columns = feature_columns
        for col in feature_columns:
            m = df[col].mean()
            s = df[col].std()
            if s == 0 or np.isnan(s):
                s = 1.0
            self.mean_dict[col] = float(m)
            self.std_dict[col] = float(s)
            df_std[col] = (df[col] - m) / s
        print(f"\n✓ Standardization completed for {len(feature_columns)} features")
        return df_std

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_destd = df.copy(deep=True)
        for col in self.feature_columns:
            df_destd[col] = (df[col] * self.std_dict[col]) + self.mean_dict[col]
        print(f"\n✓ De-standardization completed")
        return df_destd


# -------------------------
# Multi-channel injector
# -------------------------
class AdaptiveMultiChannelInjector:
    """
    Multi-channel anomaly injector with adaptive delta/window schedule and lazy fitting.
    Option A labeling: a time step is anomalous if ANY channel changed (union mask).
    """

    def __init__(self, cfg: DictConfig, anomaly_registry: dict = None):
        ds = cfg.dataset
        # Core parameters
        self.anomaly_types: List[str] = list(ds.anomalies_type)
        self.anomaly_percentage: float = float(ds.anomaly_percentage)  # percent
        self.window_mean: int = int(ds.window_mean)
        self.window_std: float = float(ds.get("window_std", max(1, 0.1 * self.window_mean)))
        self.delta_mean: float = float(ds.delta_mean)
        self.delta_std: float = float(ds.delta_std)
        self.reset_interval_pct: float = float(ds.get("reset_interval", 100))  # percent
        self.min_channels: int = int(ds.get("min_channels", 1)) if ds.get("min_channels", 1) is not None else None
        self.max_channels: int = int(ds.get("max_channels", 1)) if ds.get("max_channels", 1) is not None else None
        self.channel_prob_decay: float = float(ds.get("channel_prob_decay", 1)) if ds.get("channel_prob_decay", 1) is not None else None
        self.channel_prob_decay = 1 if self.max_channels == 1 else self.channel_prob_decay
        self.channel_prob_decay = None if self.max_channels is None else self.channel_prob_decay
        self.random_seed = ds.get("random_seed", None)

        # registry with anomaly classes (strings -> classes)
        self.anomaly_registry = anomaly_registry if anomaly_registry is not None else ANOMALIES_REGISTRY

        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            random.seed(self.random_seed)

        # fitted cache: fitted_anomalies[channel][anomaly_type] = fitted_obj
        self.fitted_anomalies: Dict[str, Dict[str, object]] = {}

        # current sampled parameters (will be updated at resets)
        self.current_delta = float(self.delta_mean)  # start with mean
        self.current_window = int(self.window_mean)

        print("\nInjector configuration:")
        print(f"  - anomaly_types: {self.anomaly_types}")
        print(f"  - anomaly_percentage: {self.anomaly_percentage}%")
        print(f"  - initial window_mean/std: {self.window_mean}/{self.window_std}")
        print(f"  - delta_mean/std: {self.delta_mean}/{self.delta_std}")
        print(f"  - reset_interval_pct: {self.reset_interval_pct}%")
        print(f"  - channels per window: {self.min_channels}-{self.max_channels}")

    # -------------------------
    # Channel selection policy
    # -------------------------
    def _select_num_channels(self, features) -> int:
        """Return number of channels to perturb following the decay policy."""
        if self.channel_prob_decay is None or self.max_channels is None:
            return len(features)
        if np.random.rand() < self.channel_prob_decay or self.max_channels <= 1:
            return 1
        # else sample 2..max_channels with exponential decay
        possible = np.arange(2, self.max_channels + 1)
        raw = np.exp(-np.arange(len(possible)))  # [1, e^-1, e^-2, ...]
        probs = raw / raw.sum()
        return int(np.random.choice(possible, p=probs))

    def _choose_channels(self, feature_columns: List[str]) -> List[str]:
        n = self._select_num_channels(feature_columns)
        if n == len(feature_columns):
            return list(feature_columns)
        else:
            n = min(n, len(feature_columns))
            return list(np.random.choice(feature_columns, size=n, replace=False))

    # -------------------------
    # Windows extraction helpers
    # -------------------------
    def _extract_nonoverlapping_windows(self, data: np.ndarray, window_len: int) -> np.ndarray:
        """Extract non-overlapping windows (shape: [n_windows, window_len])"""
        n_points = len(data)
        n_windows = n_points // window_len
        if n_windows <= 0:
            return np.empty((0, window_len))
        windows = np.array([data[i * window_len:(i + 1) * window_len] for i in range(n_windows)])
        return windows

    # -------------------------
    # Lazy fitting helpers
    # -------------------------
    def _get_or_fit(self, channel: str, anomaly_type: str, df_standardized: pd.DataFrame) -> object:
        """
        Return a fitted anomaly object for (channel, anomaly_type) using current_delta.
        Fit on-demand if not already in cache.
        """
        if channel in self.fitted_anomalies and anomaly_type in self.fitted_anomalies[channel]:
            return self.fitted_anomalies[channel][anomaly_type]

        # prepare container
        if channel not in self.fitted_anomalies:
            self.fitted_anomalies[channel] = {}

        # Extract windows for this channel using current_window (global non-overlapping)
        data = df_standardized[channel].values
        windows = self._extract_nonoverlapping_windows(data, self.current_window)
        # Fit anomaly object
        anomaly_cls = self.anomaly_registry[anomaly_type]
        obj = anomaly_cls(self.current_delta)
        # Some WOMBAT classes expect shape (M, L) where M = number of windows; pass windows
        # If windows empty, create a single repeated window from data start (fallback)
        if windows.size == 0:
            fallback = np.repeat(data[:self.current_window][np.newaxis, :], 1, axis=0)
            try:
                obj.fit(fallback)
            except Exception:
                # best-effort: try fitting on single sample reshaped
                try:
                    obj.fit(fallback.reshape(1, -1))
                except Exception as e:
                    raise RuntimeError(f"Fit failed for {anomaly_type} on {channel}: {e}")
        else:
            try:
                obj.fit(windows)
            except Exception as e:
                # try flattened fit
                try:
                    obj.fit(windows.reshape(-1, self.current_window))
                except Exception as e2:
                    raise RuntimeError(f"Fit failed for {anomaly_type} on {channel}: {e2}")

        self.fitted_anomalies[channel][anomaly_type] = obj
        return obj

    # -------------------------
    # Reset (clear fitted cache, sample new delta/window)
    # -------------------------
    def reset_schedule(self):
        """Clear cache and sample new delta & window length."""
        # sample delta
        new_delta = float(np.random.normal(self.delta_mean, self.delta_std))
        # clamp delta to positive values (domain-specific)
        new_delta = max(0.0, new_delta)
        # sample window length
        new_window = int(round(np.random.normal(self.window_mean, self.window_std)))
        new_window = max(5, new_window)
        # clear cache
        self.fitted_anomalies.clear()
        # set
        self.current_delta = new_delta
        self.current_window = new_window
        print(f"\n♻️  Schedule reset → new delta={self.current_delta:.4f}, new window={self.current_window}")

    # -------------------------
    # Injection core (UPDATED)
    # -------------------------
    def inject(self, df_standardized: pd.DataFrame, feature_columns: List[str]) -> Tuple[pd.DataFrame, List[dict], pd.DataFrame]:
        """
        Inject anomalies into df_standardized and return:
         - df_standardized with 'is_anomaly' and metadata columns,
         - anomalies_log (list of dicts),
         - stats_log DataFrame describing schedule resets.

        LABELING STRATEGY (Option A): A time step is anomalous if ANY channel changed.
        """
        n_points = len(df_standardized)
        n_target_points = int(round(n_points * (self.anomaly_percentage / 100.0)))
        if n_target_points <= 0:
            print("No anomalies requested (target 0). Returning original")
            return df_standardized.copy(deep=True), [], pd.DataFrame([])

        # reset schedule parameters: initial values
        self.current_delta = float(self.delta_mean)  # start with mean
        self.current_window = int(self.window_mean)

        # create output copy
        df_out = df_standardized.copy(deep=True)
        df_out['is_anomaly'] = 0
        df_out['anomaly_type'] = ''
        df_out['affected_channels'] = ''

        # prepare window indices (we operate on non-overlapping windows at high-level for selection)
        anomalies_log: List[dict] = []
        stats_log: List[dict] = []

        injected_points = 0
        next_reset_points = int(round((self.reset_interval_pct / 100.0) * n_target_points)) if self.reset_interval_pct > 0 else n_target_points + 1
        reset_count = 0

        print(f"\nStarting injection: target {n_target_points} anomalous points")
        print(f"Initial delta={self.current_delta:.4f}, window={self.current_window}")
        # initially the cache is empty (lazy fitting)
        self.fitted_anomalies.clear()

        # loop until we injected enough points
        pbar = tqdm(total=n_target_points, desc="Injecting anomalies")
        max_attempts = n_target_points * 10  # safety break
        attempts = 0

        while injected_points < n_target_points and attempts < max_attempts:
            attempts += 1
            # choose start index ensuring it fits current window
            if self.current_window >= n_points:
                # fallback: use whole array
                start_idx = 0
                end_idx = n_points
            else:
                start_idx = np.random.randint(0, n_points - self.current_window + 1)
                end_idx = start_idx + self.current_window

            # skip if the selected interval already contains anomalies (to avoid overlap)
            if df_out['is_anomaly'].iloc[start_idx:end_idx].any():
                continue

            # choose how many channels and which ones
            selected_channels = self._choose_channels(feature_columns)

            # pick anomaly type uniformly
            anomaly_type = random.choice(self.anomaly_types)

            # We'll collect per-channel "before" and "after" to compute exact changed points
            before_windows = {}
            after_windows = {}
            distorted_channels = []

            # First, for each channel, capture 'before', obtain the anomaly object and compute distorted 'after'
            for ch in selected_channels:
                try:
                    # capture before from df_out (standardized space)
                    before = df_out[ch].values[start_idx:end_idx].copy()

                    # get or fit anomaly object for this channel+type using current delta & current window
                    anomaly_obj = self._get_or_fit(ch, anomaly_type, df_standardized)

                    # generate distorted window (WOMBAT expects shape (M, L))
                    distorted = anomaly_obj.distort(before.reshape(1, -1))[0]

                    # store
                    before_windows[ch] = before
                    after_windows[ch] = distorted
                    distorted_channels.append(ch)
                except Exception as e:
                    print(f"  ⚠ Error preparing/applying {anomaly_type} to {ch} at [{start_idx}:{end_idx}]: {e}")
                    # skip this channel only and continue with others
                    continue

            # if no channel produced a distortion, skip
            if len(distorted_channels) == 0:
                continue

            # Compute precise per-point anomaly mask (union across channels)
            window_len = end_idx - start_idx
            changed_mask = np.zeros(window_len, dtype=bool)
            for ch in distorted_channels:
                before = before_windows[ch]
                after = after_windows[ch]
                # element-wise comparison (float) -> use != which is fine in standardized space
                changed_mask |= (before != after)
                if np.sum(changed_mask) == window_len:
                    # all points changed, no need to check further
                    break

            # If at least one point changed -> update df_out for those indices only
            if changed_mask.any():
                changed_idx_rel = np.where(changed_mask)[0]  # relative positions within window
                changed_idx_abs = (start_idx + changed_idx_rel).tolist()

                # apply after-values only for distorted channels, but only at changed positions
                for ch in distorted_channels:
                    after = after_windows[ch]
                    # only set values where after differs from before (changed_mask)
                    # use iloc since we have integer positions
                    df_out.iloc[start_idx:end_idx, df_out.columns.get_loc(ch)].values[changed_mask] = after[changed_mask]

                # mark anomaly metadata only at changed timestamps
                df_out.iloc[changed_idx_abs, df_out.columns.get_loc('is_anomaly')] = 1
                df_out.iloc[changed_idx_abs, df_out.columns.get_loc('anomaly_type')] = anomaly_type
                # affected_channels column will list all channels that were part of the distortion attempt
                df_out.iloc[changed_idx_abs, df_out.columns.get_loc('affected_channels')] = ','.join(distorted_channels)

                # log anomaly (store full window start/end and exact changed indices)
                anomalies_log.append({
                    "anomaly_id": len(anomalies_log),
                    "start_idx": int(start_idx),
                    "end_idx": int(end_idx),
                    "changed_indices": changed_idx_abs,
                    "anomaly_type": anomaly_type,
                    "affected_channels": distorted_channels,
                    "n_changed_points": int(changed_mask.sum()),
                    "window_length": int(self.current_window),
                    "delta": float(self.current_delta)
                })

                # update counters
                injected_points += int(changed_mask.sum())
                pbar.update(int(changed_mask.sum()))

            # check reset condition
            if injected_points >= next_reset_points and injected_points < n_target_points:
                reset_count += 1
                progress_pct = injected_points / n_target_points * 100.0
                print(f"\n♻️  Reset #{reset_count} triggered at {progress_pct:.2f}% injected.")
                # sample new delta and window
                new_delta = float(np.random.normal(self.delta_mean, self.delta_std))
                new_delta = max(0.0, new_delta)
                new_window = int(round(np.random.normal(self.window_mean, self.window_std)))
                new_window = max(5, new_window)
                # clear cache and set new params
                self.fitted_anomalies.clear()
                self.current_delta = new_delta
                self.current_window = new_window
                stats_log.append({
                    "reset_id": reset_count,
                    "progress_pct": round(progress_pct, 3),
                    "delta": round(self.current_delta, 6),
                    "window_length": int(self.current_window)
                })
                print(f"    New delta={self.current_delta:.4f}, new window={self.current_window}")
                # compute next reset threshold
                next_reset_points += int(round((self.reset_interval_pct / 100.0) * n_target_points))

        pbar.close()
        if attempts >= max_attempts:
            print("⚠ Reached maximum attempts while injecting anomalies (stopping).")

        # final stats log DF
        stats_df = pd.DataFrame(stats_log)

        # compute actual injected points
        actual_injected_points = int((df_out['is_anomaly'] == 1).sum())
        print(f"\nInjection finished: target={n_target_points}, actual_injected={actual_injected_points}")

        return df_out, anomalies_log, stats_df


# -------------------------
# Utility: robust JSON serialize
# -------------------------
def to_json_serializable(obj):
    """Recursively convert OmegaConf, numpy and other non-json types to plain python."""
    if isinstance(obj, (DictConfig, ListConfig)):
        obj = OmegaConf.to_container(obj, resolve=True)
    if isinstance(obj, dict):
        return {k: to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_json_serializable(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_json_serializable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# -------------------------
# Final statistics printing
# -------------------------
def print_final_statistics(df_original: pd.DataFrame, df_final: pd.DataFrame, features: List[str]):
    """
    Print channel-wise stats table including anomalous point counts per feature.
    """
    print("\n" + "=" * 100)
    print("STATISTICAL COMPARISON: Original vs Final (with anomalies)")
    print("=" * 100)

    header = (f"{ 'Channel':<20} {'Original Mean':>14} {'Final Mean':>14} {'Δ Mean (%)':>12} "
              f"{'Original Std':>14} {'Final Std':>14} {'Δ Std (%)':>12}")
    print(header)
    print("-" * len(header))

    max_mean_change = 0.0
    max_std_change = 0.0

    for col in features:
        mean_orig = df_original[col].mean()
        mean_final = df_final[col].mean()
        std_orig = df_original[col].std()
        std_final = df_final[col].std()

        # percentage changes (handle zero denom)
        mean_change_pct = (((mean_final - mean_orig) / (abs(mean_orig) + 1e-12)) * 100.0) if abs(mean_orig) > 1e-12 else (mean_final - mean_orig)
        std_change_pct = (((std_final - std_orig) / (std_orig + 1e-12)) * 100.0) if std_orig != 0 else (std_final - std_orig)

        max_mean_change = max(max_mean_change, abs(mean_change_pct))
        max_std_change = max(max_std_change, abs(std_change_pct))

        print(f"{col:<20} {mean_orig:14.4f} {mean_final:14.4f} {mean_change_pct:12.2f}% "
              f"{std_orig:14.4f} {std_final:14.4f} {std_change_pct:12.2f}%")

    print("-" * len(header))
    print(f"Maximum changes: Mean={max_mean_change:.2f}%, Std={max_std_change:.2f}%")
    return


# -------------------------
# Main pipeline
# -------------------------
def main(args):
    cfg = OmegaConf.load(args.conf_file)
    data_path = cfg.dataset.data_path
    features = list(cfg.dataset.feats)

    print("\n" + "=" * 70)
    print("WOMBAT MULTI-CHANNEL ADAPTIVE INJECTION")
    print("=" * 70)
    print(f"Loading data from: {data_path}")

    # Load data (the helper should return a pandas.DataFrame)
    df_original = load_data(cfg)
    print(f"Loaded {len(df_original)} rows × {df_original.shape[1]} cols")

    df_backup = df_original.copy(deep=True)

    # Standardize
    handler = StandardizationHandler()
    df_std = handler.fit_transform(df_original, feature_columns=features)

    # Create injector
    injector = AdaptiveMultiChannelInjector(cfg, anomaly_registry=ANOMALIES_REGISTRY)

    # Run injection (lazy fitting + adaptive schedule)
    df_with_anom_std, anomalies_log, schedule_df = injector.inject(df_std, features)

    # De-standardize
    df_with_anom = handler.inverse_transform(df_with_anom_std)

    # Restore original values for non-anomalous points
    anomaly_mask = df_with_anom_std['is_anomaly'].astype(bool)
    for col in features:
        df_with_anom.loc[~anomaly_mask, col] = df_backup.loc[~anomaly_mask, col]

    # Save dataset in same format (pkl or csv)
    dir_path = os.path.dirname(data_path)
    base_name, ext = os.path.splitext(os.path.basename(data_path))
    if ext == '':
        ext = '.pkl'  # fallback
    output_path = os.path.join(dir_path, f"{base_name}_with_anomalies{ext}")
    if ext == '.pkl':
        df_with_anom.to_pickle(output_path)
    else:
        df_with_anom.to_csv(output_path, index=False)
    print(f"\n✓ Saved dataset: {output_path}")

    # Build metadata
    n_points = len(df_with_anom)
    n_anom_points = int((df_with_anom_std['is_anomaly'] == 1).sum())
    anomalies_metadata = {
        "metadata": {
            "creation_date": datetime.now().isoformat(),
            "original_data_path": data_path,
            "output_data_path": output_path,
            "total_points": int(n_points),
            "features": features,
            "n_features": len(features),
            "random_seed": int(cfg.dataset.random_seed) if cfg.dataset.random_seed is not None else None
        },
        "injection_config": {
            "anomaly_types": list(cfg.dataset.anomalies_type),
            "anomaly_percentage_target": float(cfg.dataset.anomaly_percentage),
            "delta_mean": float(cfg.dataset.delta_mean),
            "delta_std": float(cfg.dataset.delta_std),
            "window_mean": int(cfg.dataset.window_mean),
            "window_std": float(cfg.dataset.get("window_std", cfg.dataset.window_mean * 0.1)),
            "reset_interval_pct": float(cfg.dataset.get("reset_interval", 100)),
            "min_channels": int(cfg.dataset.get("min_channels", 1)) if cfg.dataset.get("min_channels", 1) is not None else None,
            "max_channels": int(cfg.dataset.get("max_channels", 1)) if cfg.dataset.get("max_channels", 1) is not None else None,
            "channel_prob_decay": float(cfg.dataset.get("channel_prob_decay", 0)) if cfg.dataset.get("channel_prob_decay", 0)is not None else None,
        },
        "summary": {
            "target_anomalous_points": int(round(n_points * (float(cfg.dataset.anomaly_percentage) / 100.0))),
            "actual_anomalous_points": int(n_anom_points),
            "schedule_resets": schedule_df.to_dict(orient="records") if not schedule_df.empty else []
        },
        "anomalies": anomalies_log
    }

    # JSON-safe conversion and save
    anomalies_metadata_clean = to_json_serializable(anomalies_metadata)
    json_path = os.path.join(dir_path, f"{base_name}_anomalies_info.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(anomalies_metadata_clean, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved metadata JSON: {json_path}")

    # Final statistics print (including anomalous points per feature)
    print_final_statistics(df_backup, df_with_anom, features)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)

    # Campiona e salva un sottoinsieme di sequenze anomale per ispezione visiva
    sample_dir = os.path.join(dir_path, "anomaly_samples")

    # --- clean existing output dir ---
    if os.path.exists(sample_dir):
        shutil.rmtree(sample_dir)
    os.makedirs(sample_dir, exist_ok=True)

    sample_and_plot_anomalies_with_labels(
        df_original=df_backup,  # denormalized
        df_with_anom=df_with_anom,  # denormalized
        df_labels=df_with_anom_std["is_anomaly"],  # 0/1 labels
        anomalies_log=anomalies_metadata["anomalies"],
        output_dir=os.path.join(dir_path, "anomaly_samples"),
        sample_pct=5.0,
        extend_window_plot_factor=0.5,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adaptive multi-channel anomaly injection (WOMBAT-style)")
    parser.add_argument("--conf_file", "-c", type=str, default="./dataset_configuration/fiorire_1.yaml",
                        help="Path to config YAML")
    args = parser.parse_args()
    main(args)
