import pandas as pd
from typing import List, Dict, Tuple
import csv
import random
import matplotlib.pyplot as plt
from omegaconf import DictConfig, ListConfig, OmegaConf
from sympy.codegen import Print
import shutil

from config import *
from wombats.anomalies.increasing import *
from wombats.anomalies.invariant import *
from wombats.anomalies.decreasing import *

ANOMALIES_REGISTRY = {
    'GWN':GWN,
    'Constant':Constant,
    'Step':Step,
    'Impulse':Impulse,
    'GNN':GNN,
    'PrincipalSubspaceAlteration': PrincipalSubspaceAlteration,
    }

def to_json_serializable(obj):
    """Recursively convert OmegaConf and numpy types to pure Python"""
    if isinstance(obj, (DictConfig, ListConfig)):
        obj = OmegaConf.to_container(obj, resolve=True)
    if isinstance(obj, dict):
        return {k: to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_json_serializable(v) for v in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def make_json_safe(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (np.ndarray, list, tuple)):
        return [make_json_safe(o) for o in obj]
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    return obj

def load_dataframe(cfg):
    """
    Load a pandas DataFrame from CSV, TSV, Excel, Parquet, Pickle, or text-like files.
    Automatically detects file type and delimiter for text files. Validates expected columns if provided.

    Parameters:
        file_path (str): Path to the data file.
        expected_cols (list of str, optional): List of expected column names. Used to validate delimiter detection.

    Returns:
        pd.DataFrame: Loaded DataFrame.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the file format is unsupported or cannot be parsed.
    """
    file_path = cfg.dataset.data_path
    expected_cols = cfg.dataset.feats

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()

    def detect_delimiter(path):
        """Auto-detect delimiter from first line of a text file."""
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            sample = f.read(2048)
        try:
            return csv.Sniffer().sniff(sample).delimiter
        except csv.Error:
            for d in [",", "\t", ";", "|"]:
                if d in sample:
                    return d
            return ","  # default

    def try_read_csv(path):
        """Try multiple delimiters and validate expected columns."""
        delimiters = [",", "\t", ";", "|"]
        for d in delimiters:
            try:
                df = pd.read_csv(path, delimiter=d)
                if expected_cols is None or all(col in df.columns for col in expected_cols):
                    return df  # valid delimiter
            except Exception:
                continue
        # last resort: read with auto-detected delimiter
        d = detect_delimiter(path)
        df = pd.read_csv(path, delimiter=d)
        return df

    try:
        # CSV / text-like files
        if ext in [".csv", ".txt", ".dat"] or ext not in [".xlsx", ".xlsm", ".xls", ".xlsb", ".odf", ".ods", ".odt", ".pkl", ".pickle", ".parquet"]:
            return try_read_csv(file_path)

        # Pickle
        elif ext in [".pkl", ".pickle"]:
            return pd.read_pickle(file_path)

        # Parquet
        elif ext == ".parquet":
            return pd.read_parquet(file_path)

        # Excel modern
        elif ext in [".xlsx", ".xlsm"]:
            try:
                df = pd.read_excel(file_path, engine="openpyxl")
            except Exception:
                df = try_read_csv(file_path)
            return df

        # Excel old
        elif ext == ".xls":
            try:
                df = pd.read_excel(file_path, engine="xlrd")
            except Exception:
                try:
                    df = pd.read_excel(file_path, engine="openpyxl")
                except Exception:
                    df = try_read_csv(file_path)
            return df

        # Excel binary
        elif ext == ".xlsb":
            return pd.read_excel(file_path, engine="pyxlsb")

        # OpenDocument formats
        elif ext in [".odf", ".ods", ".odt"]:
            return pd.read_excel(file_path, engine="odf")

        else:
            # Last resort: treat as text
            return try_read_csv(file_path)

    except Exception as e:
        raise ValueError(f"Failed to load file '{file_path}': {e}")

def load_data(cfg):

    df = load_dataframe(cfg)

    cfg.dataset.target = (
        cfg.dataset.target
        if isinstance(cfg.dataset.target, (list, ListConfig))
        else [cfg.dataset.target] if cfg.dataset.target
        else None
    )

    columns = [x for x in cfg.dataset.feats if x not in cfg.dataset.target] if cfg.dataset.target else cfg.dataset.feats

    df = df[columns].dropna()
    #if cfg.dataset.dataset_subset:
    #p    df = df.iloc[:cfg.dataset.dataset_subset, :]

    flag_col = getattr(cfg.dataset, "flag_col", None)
    align_data = getattr(cfg.dataset, "align_data", False)
    detect_flag = getattr(cfg.dataset, "detect_flag", False)

    # Optional alignment and flag trimming
    if align_data or detect_flag:
        col_to_rem = [c for c in df.columns if c.startswith("ANT47") or c.startswith("Frame")]
        if col_to_rem:
            print(f"🔧 Preprocessing: removing {len(col_to_rem)} ANT47 columns")
            df = df.drop(columns=[c for c in df.columns if c.startswith("ANT47") or c.startswith("Frame")])
        print("🔧 Preprocessing: align_data =", align_data, ", detect_flag =", detect_flag)
        # --- 1️⃣ Align columns if required ---
        if align_data:
            series_dict = {col: df[col] for col in df.columns}
            min_len = min(len(s) for s in series_dict.values())
            aligned_data = {col: s.iloc[:min_len].reset_index(drop=True) for col, s in series_dict.items()}
            df = pd.DataFrame(aligned_data)
            print(f"✅ Data aligned to {min_len} samples")

        # --- 2️⃣ Detect flag and trim if required ---
        if detect_flag:
            if flag_col and flag_col in df.columns:
                print(f"⚙️ Detecting first change in flag column: '{flag_col}'")
                changes = df[flag_col].diff().fillna(0)
                change_idxs = changes[changes != 0].index
                if len(change_idxs) > 0:
                    first_change_idx = change_idxs[0]
                    df = df.loc[first_change_idx:].reset_index(drop=True)
                    print(f"✅ Trimmed dataset from first flag change at index {first_change_idx}")
                else:
                    print("⚠️ No flag change detected — dataset not trimmed.")
            else:
                print("⚠️ No valid flag column found in cfg.dataset.flag_column.")

    return df

def sample_and_plot_anomalies_with_labels(
    df_original: pd.DataFrame,      # denormalized original
    df_with_anom: pd.DataFrame,     # denormalized with anomalies
    df_labels: pd.Series,           # 0/1 labels aligned with df_with_anom
    anomalies_log: list,
    output_dir: str,
    sample_pct: float = 5.0,
    extend_window_plot_factor: float = 0.5,
):
    """
    Sample a subset of anomalies and plot:

        - original channel (denormalized)
        - injected channel (denormalized)
        - 0/1 labels on a secondary y-axis
        - a context window around the anomaly region

    extend_window_plot_factor defines how much extra context is shown.
    """

    if not anomalies_log:
        print("⚠ No anomalies found — no plots generated.")
        return

    # Clean output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    n_anoms = len(anomalies_log)
    n_sample = max(1, int(round(n_anoms * sample_pct / 100.0)))
    sampled = random.sample(anomalies_log, n_sample)

    print(f"\n📊 Generating {n_sample} anomaly plots in: {output_dir}")

    total_len = len(df_original)

    for i, anom in enumerate(sampled, 1):
        start = anom["start_idx"]
        end = anom["end_idx"]
        anomaly_type = anom["anomaly_type"]
        delta = anom["delta"]
        affected_channels = anom["affected_channels"]

        # -------------------------
        # EXTENDED WINDOW CALCULATION
        # -------------------------
        window_len = end - start
        extra = int(window_len * extend_window_plot_factor)

        plot_start = max(0, start - extra)
        plot_end   = min(total_len, end + extra)

        # slice labels
        labels_window = df_labels.iloc[plot_start:plot_end].values
        x_axis = np.arange(plot_end - plot_start)

        for ch in affected_channels:

            orig = df_original[ch].iloc[plot_start:plot_end].values   # ← denormalized
            anomv = df_with_anom[ch].iloc[plot_start:plot_end].values # ← denormalized

            if len(orig) == 0:
                continue

            # ---- Plot ----
            fig, ax1 = plt.subplots(figsize=(12, 5))

            ax1.plot(x_axis, orig, label="Original", alpha=0.6)
            ax1.plot(x_axis, anomv, label="Injected", alpha=0.8)

            ax1.set_xlabel("Index (relative)")
            ax1.set_ylabel("Signal value")
            ax1.legend(loc="upper left")

            # SECOND AXIS FOR LABELS
            ax2 = ax1.twinx()
            ax2.plot(
                x_axis,
                labels_window,
                drawstyle="steps-post",
                linewidth=2,
                alpha=0.7,
                color="red"
            )
            ax2.set_ylabel("Label (0/1)")
            ax2.set_ylim(-0.1, 1.2)

            # highlight the true anomaly window
            anomaly_rel_start = start - plot_start
            anomaly_rel_end   = end - plot_start

            ax1.axvspan(
                anomaly_rel_start,
                anomaly_rel_end,
                color='red',
                alpha=0.12,
                label="Anomalous interval"
            )

            plt.title(
                f"{ch} — {anomaly_type} (Δ={delta:.3f})\n"
                f"Anomaly [{start}:{end}] | Plot [{plot_start}:{plot_end}]"
            )
            plt.tight_layout()

            fname = f"sample_{i:03d}_{ch}_{anomaly_type}_Δ{delta:.2f}.png"
            plt.savefig(os.path.join(output_dir, fname), dpi=150)
            plt.close()

    print(f"✓ Saved {n_sample} anomaly examples\n")
