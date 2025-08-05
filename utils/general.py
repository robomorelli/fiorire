# utils/general.py
from pathlib import Path
from omegaconf import ListConfig
from omegaconf import OmegaConf
import numpy as np
from config import *

def find_project_root(current: Path, markers=('config', 'main.py')) -> Path:
    """
    Ascend from the current directory to find the project root based on common markers.
    """
    for parent in [current] + list(current.parents):
        if any((parent / marker).exists() for marker in markers):
            return parent
    return current  # fallback: return current if no marker found

def make_paths_absolute(cfg):
    dataset_path = Path(cfg.dataset.data_path)

    if not dataset_path.is_absolute():
        current_file = Path(__file__).resolve()
        project_root = find_project_root(current_file)
        absolute_path = project_root / dataset_path
        cfg.dataset.data_path = str(absolute_path.resolve())

def extract_config_bkp(cfg_path):
    cfg = OmegaConf.load(cfg_path)
    config = {}
    for k, v in cfg.tune_config.items():
        try:
            config[k] = ray_mapper[v.split('(')[0]]([float(s) if '.' in s else int(s) for s in v.split(v.split('(')[0])[1].\
                                                strip("()").strip("[]").split(',')])
            print([float(s) if '.' in s else int(s) for s in v.split(v.split('(')[0])[1].\
                                                strip("()").strip("[]").split(',')])
        except:
            config[k] = ray_mapper[v.split('(')[0]]([s.strip(' ').strip("''") for s in v.split(v.split('(')[0])[1]\
                                                    .strip("()").strip("[]").split(',')])
            print([s.strip(' ').strip("''") for s in v.split(v.split('(')[0])[1].strip("()")\
                  .strip("[]").split(',')])
    return config, cfg

def extract_config(cfg_path):
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(cfg_path)
    config = {}

    for k, v in cfg.tune_config.items():
        try:
            # Handle numeric values
            config[k] = ray_mapper[v.split('(')[0]](
                [float(s) if '.' in s else int(s) for s in v.split(v.split('(')[0])[1].strip("()[]").split(',')]
            )
        except:
            # Handle string or categorical values
            config[k] = ray_mapper[v.split('(')[0]](
                [s.strip().strip("''").strip('"') for s in v.split(v.split('(')[0])[1].strip("()[]").split(',')]
            )
    return config, cfg


def extract_fixed_config(cfg_path):
    cfg = OmegaConf.load(cfg_path)
    config = {}
    for k, v in cfg.tune_config.items():
        if isinstance(v, (list, ListConfig)):  # e.g. [0.001, 0.003]
            config[k] = v[0]  # Use first value for testing
        elif isinstance(v, str) and v.startswith("tune.choice"):
            # Handle cases where string parsing is needed
            values = v[v.find("[")+1 : v.find("]")].split(",")
            config[k] = eval(values[0].strip())
        else:
            config[k] = v  # Fallback
    return config, cfg

def extract_fixed_config_bkp(cfg_path):
    cfg = OmegaConf.load(cfg_path)
    config = {}

    for k, v in cfg.tune_config.items():
        if isinstance(v, (list, ListConfig)):
            config[k] = v[0]  # Just use the first item in the list
        elif isinstance(v, str) and v.startswith("tune.choice"):
            try:
                # Try parsing as float/int
                values = [float(s) if '.' in s else int(s)
                          for s in v[v.find("[")+1 : v.find("]")].split(",")]
                config[k] = values[0]
            except ValueError:
                # Parse as strings
                values = [s.strip().strip('"').strip("'")
                          for s in v[v.find("[")+1 : v.find("]")].split(",")]
                config[k] = values[0]
        else:
            config[k] = v  # Fallback
    return config, cfg


def inject_binary_anomalies(length=100_000, anomaly_ratio=0.1, min_seq_len=100, max_seq_len=5000, seed=42):
    np.random.seed(seed)
    total_anomalies = int(length * anomaly_ratio)  # e.g., 10_000
    binary_column = np.zeros(length, dtype=int)

    used = 0
    starts = []

    while used < total_anomalies:
        # Sample random sequence length
        seq_len = np.random.randint(min_seq_len, max_seq_len + 1)
        seq_len = min(seq_len, total_anomalies - used)  # don't overshoot total

        # Find valid starting point
        max_start = length - seq_len
        attempts = 0
        while True:
            start = np.random.randint(0, max_start)
            end = start + seq_len
            # Avoid overlap with previous sequences
            if binary_column[start:end].sum() == 0:
                break
            attempts += 1
            if attempts > 1000:
                raise RuntimeError("Too many attempts to find non-overlapping region.")

        binary_column[start:end] = 1
        starts.append((start, end))
        used += seq_len

    return binary_column, starts