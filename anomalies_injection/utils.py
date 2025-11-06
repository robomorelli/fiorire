import pandas as pd
from typing import List, Dict, Tuple
import csv
import random
import matplotlib.pyplot as plt
from omegaconf import DictConfig, ListConfig, OmegaConf
from config import *
from wombats.anomalies.increasing import *
from wombats.anomalies.invariant import *
from wombats.anomalies.decreasing import *

ANOMALIES_REGISTRY = {
    'GWN':GWN,
    'Constant':Constant,
    'Step':Step,
    'Impulse':Impulse,
    'GNN':GNN
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

def sample_and_plot_anomalies(
    df_original: pd.DataFrame,
    df_with_anom: pd.DataFrame,
    anomalies_log: List[dict],
    output_dir: str,
    sample_pct: float = 5.0,
):
    """
    Campiona una percentuale di sequenze anomale e salva i grafici di confronto.
    Ogni grafico mostra la serie originale e quella con anomalia per un singolo canale.

    Args:
        df_original: DataFrame originale (non standardizzato)
        df_with_anom: DataFrame finale (de-standardizzato)
        anomalies_log: lista dei dizionari di anomalie iniettate
        output_dir: cartella di destinazione per i plot
        sample_pct: percentuale di anomalie da campionare (es. 5.0 = 5%)
    """
    if not anomalies_log:
        print("⚠ Nessuna anomalia registrata, nessun plot generato.")
        return

    os.makedirs(output_dir, exist_ok=True)

    n_anoms = len(anomalies_log)
    n_sample = max(1, int(round(n_anoms * sample_pct / 100.0)))
    sampled = random.sample(anomalies_log, n_sample)

    print(f"\n📊 Genero {n_sample} grafici di anomalie campionate in '{output_dir}'")

    for i, anom in enumerate(sampled, 1):
        start = anom["start_idx"]
        end = anom["end_idx"]
        anomaly_type = anom["anomaly_type"]
        delta = anom["delta"]
        affected_channels = anom["affected_channels"]

        for ch in affected_channels:
            orig = df_original[ch].iloc[start:end].values
            anomv = df_with_anom[ch].iloc[start:end].values
            if len(orig) == 0:
                continue

            plt.figure(figsize=(8, 4))
            plt.plot(orig, label="Originale", alpha=0.7)
            plt.plot(anomv, label="Anomalia", alpha=0.8)
            plt.title(f"{ch} — {anomaly_type} (Δ={delta:.3f}) [{start}:{end}]")
            plt.xlabel("Indice campione")
            plt.ylabel("Valore")
            plt.legend()
            plt.tight_layout()

            fname = f"sample_{i:03d}_{ch}_{anomaly_type}_Δ{delta:.2f}.png"
            fpath = os.path.join(output_dir, fname)
            plt.savefig(fpath, dpi=150)
            plt.close()

    print(f"✓ Salvati {n_sample} esempi di anomalie ({sample_pct:.1f}% del totale)")