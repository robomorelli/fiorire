#!/usr/bin/env python3
"""
Adaptive WOMBAT-style multi-channel anomaly injector
- Fixed window length and delta (no runtime schedule)
- Optimized selection of start indices (shuffle & subset)
- No overlapping anomalies
- Saves .pkl or .csv and generates metadata + plots
"""

import os
import json
import random
import argparse
from datetime import datetime
from typing import Dict, List, Tuple
import shutil

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from omegaconf import OmegaConf, DictConfig, ListConfig

from anomalies_injection.utils import (
    load_data, ANOMALIES_REGISTRY, make_json_safe,
    sample_and_plot_anomalies_with_labels
)


# -------------------------
# Standardization Handler
# -------------------------
class StandardizationHandler:
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
# Multi-channel injector (fixed windows)
# -------------------------
class AdaptiveMultiChannelInjector:
    def __init__(self, cfg: DictConfig, anomaly_registry: dict = None):
        ds = cfg.dataset

        # Core parameters (fixed)
        self.anomaly_types: List[str] = list(ds.anomalies_type)
        self.anomaly_percentage: float = float(ds.anomaly_percentage)
        self.window_length: int = int(ds.get("window_mean", 16))
        self.delta: float = float(ds.get("delta_mean", 0.5))
        self.min_channels: int = int(ds.get("min_channels", 1)) if ds.get("min_channels", 1) else 1
        self.max_channels: int = int(ds.get("max_channels", 1)) if ds.get("max_channels", 1) else 1
        self.channel_prob_decay: float = float(ds.get("channel_prob_decay", 1)) if ds.get("channel_prob_decay", 1) else 1
        self.channel_prob_decay = 1 if self.max_channels == 1 else self.channel_prob_decay
        self.channel_prob_decay = None if self.max_channels is None else self.channel_prob_decay
        self.random_seed = ds.get("random_seed", None)

        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            random.seed(self.random_seed)

        self.anomaly_registry = anomaly_registry if anomaly_registry is not None else ANOMALIES_REGISTRY

        # fitted cache
        self.fitted_anomalies: Dict[str, Dict[str, object]] = {}

        print("\nInjector configuration (fixed parameters):")
        print(f"  - anomaly_types: {self.anomaly_types}")
        print(f"  - anomaly_percentage: {self.anomaly_percentage}%")
        print(f"  - window_length: {self.window_length}")
        print(f"  - delta: {self.delta}")
        print(f"  - channels per window: {self.min_channels}-{self.max_channels}")

    # -------------------------
    # Channel selection
    # -------------------------
    def _choose_channels(self, feature_columns: List[str]) -> List[str]:
        n_channels = self.min_channels if self.min_channels == self.max_channels else np.random.randint(self.min_channels, self.max_channels + 1)
        return list(np.random.choice(feature_columns, size=min(n_channels, len(feature_columns)), replace=False))

    # -------------------------
    # Lazy fitting
    # -------------------------
    def _get_or_fit(self, channel: str, anomaly_type: str, df_standardized: pd.DataFrame):
        if channel in self.fitted_anomalies and anomaly_type in self.fitted_anomalies[channel]:
            return self.fitted_anomalies[channel][anomaly_type]

        if channel not in self.fitted_anomalies:
            self.fitted_anomalies[channel] = {}

        data = df_standardized[channel].values
        anomaly_cls = self.anomaly_registry[anomaly_type]
        obj = anomaly_cls(self.delta)
        try:
            obj.fit(data.reshape(-1, self.window_length))
        except Exception:
            # fallback single window
            obj.fit(data[:self.window_length].reshape(1, -1))

        self.fitted_anomalies[channel][anomaly_type] = obj
        return obj

    # -------------------------
    # Injection (optimized fixed windows)
    # -------------------------
    def inject(self, df_standardized: pd.DataFrame, feature_columns: List[str]) -> Tuple[pd.DataFrame, List[dict], pd.DataFrame]:
        n_points = len(df_standardized)
        n_windows = n_points // self.window_length
        if n_windows == 0:
            raise ValueError("Dataset too short for the given window length.")

        # valid start indices for windows
        start_indices = np.arange(n_windows) * self.window_length
        np.random.shuffle(start_indices)

        n_windows_to_perturb = max(1, int(round(n_windows * (self.anomaly_percentage / 100.0))))
        selected_starts = start_indices[:n_windows_to_perturb]

        df_out = df_standardized.copy(deep=True)
        df_out['is_anomaly'] = 0
        df_out['anomaly_type'] = ''
        df_out['affected_channels'] = ''

        anomalies_log = []

        print(f"\nInjecting anomalies into {n_windows_to_perturb} windows (out of {n_windows})")
        pbar = tqdm(total=n_windows_to_perturb, desc="Injecting anomalies")

        for start_idx in selected_starts:
            end_idx = min(start_idx + self.window_length, n_points)
            selected_channels = self._choose_channels(feature_columns)
            anomaly_type = random.choice(self.anomaly_types)

            distorted_channels = []
            changed_mask = np.zeros(end_idx - start_idx, dtype=bool)

            for ch in selected_channels:
                before = df_out[ch].values[start_idx:end_idx].copy()
                try:
                    anomaly_obj = self._get_or_fit(ch, anomaly_type, df_standardized)
                    distorted = anomaly_obj.distort(before.reshape(1, -1))[0]
                except Exception:
                    distorted = before.copy()  # fallback, no change
                # compute changed mask
                changed_mask |= (before != distorted)
                # apply changes
                df_out.iloc[start_idx:end_idx, df_out.columns.get_loc(ch)].values[changed_mask] = distorted[changed_mask]
                distorted_channels.append(ch)

            if changed_mask.any():
                changed_idx_abs = (start_idx + np.where(changed_mask)[0]).tolist()
                df_out.iloc[changed_idx_abs, df_out.columns.get_loc('is_anomaly')] = 1
                df_out.iloc[changed_idx_abs, df_out.columns.get_loc('anomaly_type')] = anomaly_type
                df_out.iloc[changed_idx_abs, df_out.columns.get_loc('affected_channels')] = ','.join(distorted_channels)

                anomalies_log.append({
                    "anomaly_id": len(anomalies_log),
                    "start_idx": int(start_idx),
                    "end_idx": int(end_idx),
                    "changed_indices": changed_idx_abs,
                    "anomaly_type": anomaly_type,
                    "affected_channels": distorted_channels,
                    "n_changed_points": int(changed_mask.sum()),
                    "window_length": int(self.window_length),
                    "delta": float(self.delta)
                })

            pbar.update(1)

        pbar.close()
        actual_points = int((df_out['is_anomaly'] == 1).sum())
        target_points = int(round(len(df_standardized) * (self.anomaly_percentage / 100)))

        print(f"\nInjection finished: total anomalous points={actual_points}")

        # ---------------------------------------
        # WARNING PER DISCREPANZE NEI PUNTI
        # ---------------------------------------
        if abs(actual_points - target_points) > target_points * 0.01:
            print("\n" + "!" * 80)
            print("⚠️  WARNING: Mismatch between target and actual anomalous points")
            print(f"    Target:    {target_points}")
            print(f"    Actual:    {actual_points}")
            print("\nPossible causes:")
            print("  • Some anomalies do NOT alter all 16 points of the window")
            print("  • Some anomalies (e.g., Impulse, Step, PSA) modify only certain feature dimensions")
            print("  • The window may be valid, but the applied anomaly does not affect every value")
            print("\nThis discrepancy is expected and depends on the intrinsic nature of each anomaly type.")
            print("!" * 80 + "\n")
        return df_out, anomalies_log, pd.DataFrame([])  # no schedule resets


# -------------------------
# JSON-safe utility
# -------------------------
def to_json_serializable(obj):
    if isinstance(obj, (DictConfig, ListConfig)):
        obj = OmegaConf.to_container(obj, resolve=True)
    if isinstance(obj, dict):
        return {k: to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list) or isinstance(obj, tuple):
        return [to_json_serializable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# -------------------------
# Final statistics
# -------------------------
def print_final_statistics(df_original: pd.DataFrame, df_final: pd.DataFrame, features: List[str]):
    print("\n" + "=" * 100)
    print("STATISTICAL COMPARISON: Original vs Final (with anomalies)")
    print("=" * 100)
    header = (f"{ 'Channel':<20} {'Original Mean':>14} {'Final Mean':>14} {'Δ Mean (%)':>12} "
              f"{'Original Std':>14} {'Final Std':>14} {'Δ Std (%)':>12}")
    print(header)
    print("-" * len(header))

    for col in features:
        mean_orig = df_original[col].mean()
        mean_final = df_final[col].mean()
        std_orig = df_original[col].std()
        std_final = df_final[col].std()
        mean_change_pct = (((mean_final - mean_orig) / (abs(mean_orig) + 1e-12)) * 100.0) if abs(mean_orig) > 1e-12 else (mean_final - mean_orig)
        std_change_pct = (((std_final - std_orig) / (std_orig + 1e-12)) * 100.0) if std_orig != 0 else (std_final - std_orig)
        print(f"{col:<20} {mean_orig:14.4f} {mean_final:14.4f} {mean_change_pct:12.2f}% "
              f"{std_orig:14.4f} {std_final:14.4f} {std_change_pct:12.2f}%")


# -------------------------
# Main pipeline
# -------------------------
def main(args):
    cfg = OmegaConf.load(args.conf_file)
    features = list(cfg.dataset.feats)
    target_channels = cfg.dataset.get("target_channels", features)
    anomalies_types = cfg.dataset.anomalies_type
    data_path = cfg.dataset.data_path
    if target_channels is None or len(target_channels) == 0:
        target_channels = features

    print(f"Channels selected for anomaly injection: {target_channels}")
    df_original = load_data(cfg)
    print(f"Loaded {len(df_original)} rows × {df_original.shape[1]} cols")
    df_backup = df_original.copy(deep=True)

    exp_name = '_'.join(('delta_' + str(cfg.dataset.delta_mean).split('.')[1], 'window_mean_' + str(cfg.dataset.window_mean),
                         "perc_" + str(cfg.dataset.anomaly_percentage), '_'.join([x for x in anomalies_types])
                         , 'num_target_channels_' + str(len(target_channels))))

    # Standardize
    handler = StandardizationHandler()
    df_std = handler.fit_transform(df_original, feature_columns=features)

    # Injector
    injector = AdaptiveMultiChannelInjector(cfg, anomaly_registry=ANOMALIES_REGISTRY)
    df_with_anom_std, anomalies_log, _ = injector.inject(df_std, target_channels)

    # De-standardize
    df_with_anom = handler.inverse_transform(df_with_anom_std)
    anomaly_mask = df_with_anom_std['is_anomaly'].astype(bool)
    for col in features:
        df_with_anom.loc[~anomaly_mask, col] = df_backup.loc[~anomaly_mask, col]


    dir_path = os.path.dirname(data_path)
    base_name, ext = os.path.splitext(os.path.basename(data_path))
    if ext == '':
        ext = '.pkl'  # fallback
    output_path = os.path.join(dir_path, f"{base_name}_{exp_name}_with_anomalies{ext}")
    if ext == '.pkl':
        df_with_anom.to_pickle(output_path)
    else:
        df_with_anom.to_csv(output_path, index=False)
    print(f"\n✓ Saved dataset: {output_path}")

    # Metadata
    anomalies_metadata = {
        "metadata": {
            "creation_date": datetime.now().isoformat(),
            "original_data_path": cfg.dataset.data_path,
            "output_data_path": output_path,
            "total_points": int(len(df_with_anom)),
            "features": features,
            "n_features": len(features),
            "target_channels": target_channels,
            "n_target_channels": len(target_channels),
            "random_seed": int(cfg.dataset.random_seed) if cfg.dataset.random_seed is not None else None
        },
        "injection_config": {
            "anomaly_types": list(cfg.dataset.anomalies_type),
            "anomaly_percentage_target": float(cfg.dataset.anomaly_percentage),
            "window_length": int(cfg.dataset.get("window_mean",16)),
            "delta": float(cfg.dataset.get("delta_mean",0.5)),
            "min_channels": int(cfg.dataset.get("min_channels", 1)) if cfg.dataset.get("min_channels", 1) is not None else None,
            "max_channels": int(cfg.dataset.get("max_channels", 1)) if cfg.dataset.get("max_channels", 1) is not None else None,
            "channel_prob_decay": float(cfg.dataset.get("channel_prob_decay", 0)) if cfg.dataset.get("channel_prob_decay", 0)is not None else None,
            "injection_channels": target_channels
        },
        "summary": {
            "target_anomalous_points": int(round(len(df_with_anom)*cfg.dataset.anomaly_percentage/100)),
            "actual_anomalous_points": int(anomaly_mask.sum())
        },
        "anomalies": anomalies_log
    }

    summary= {
        "target_anomalous_points": int(round(len(df_with_anom) * cfg.dataset.anomaly_percentage / 100)),
        "actual_anomalous_points": int(anomaly_mask.sum())
    }

    print(summary)

    json_path = os.path.join(dir_path, f"{base_name}_{exp_name}_info.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(to_json_serializable(anomalies_metadata), f, indent=2, ensure_ascii=False)
    print(f"✓ Saved metadata JSON: {json_path}")

    # Statistics
    print_final_statistics(df_backup, df_with_anom, features)

    # Plot sample anomalies
    # Campiona e salva un sottoinsieme di sequenze anomale per ispezione visiva
    sample_dir = os.path.join(dir_path, f"anomaly_samples_{exp_name}")

    # --- clean existing output dir ---
    if os.path.exists(sample_dir):
        shutil.rmtree(sample_dir)
    os.makedirs(sample_dir, exist_ok=True)

    sample_and_plot_anomalies_with_labels(
        df_original=df_backup,
        df_with_anom=df_with_anom,
        df_labels=df_with_anom_std["is_anomaly"],
        anomalies_log=anomalies_metadata["anomalies"],
        output_dir=sample_dir,
        sample_pct=5.0,
        extend_window_plot_factor=1.5
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adaptive multi-channel anomaly injection (WOMBAT-style)")
    parser.add_argument("--conf_file", "-c", type=str, default="./dataset_configuration/fiorire_1.yaml",
                        help="Path to config YAML")
    args = parser.parse_args()
    main(args)

