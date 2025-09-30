# utils/general.py
from pathlib import Path
from omegaconf import OmegaConf, DictConfig, ListConfig
from typing import Tuple
import numpy as np
import torch
import ray
from ray.tune.syncer import SyncConfig

from config import *

def find_project_root(current: Path, markers=('config', 'main.py')) -> Path:
    """
    Ascend from the current directory to find the project root based on common markers.
    """
    for parent in [current] + list(current.parents):
        if any((parent / marker).exists() for marker in markers):
            return parent
    return current  # fallback: return current if no marker found

def resolve_paths(cfg: DictConfig, root_dir: str=root) -> DictConfig:
    """
    Recursively find keys containing 'path' and resolve relative paths
    against root_dir.
    """
    def _resolve(d):
        for k, v in d.items():
            if isinstance(v, DictConfig) or isinstance(v, dict):
                _resolve(v)
            elif isinstance(v, str) and "path" in k.lower():
                if not os.path.isabs(v):
                    abs_path = os.path.abspath(os.path.join(root_dir, v))
                    d[k] = abs_path

    _resolve(cfg)
    return cfg


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


def infer_model_type(model: torch.nn.Module) -> Tuple[str, str]:
    """
    Infer model type from the modules of the model.
    Returns:
        - model_type: One of ['cnn', 'lstm', 'mixed', 'unknown']
        - defining_layer: The name of the last relevant layer class (e.g., 'Conv1d' or 'LSTM')
    """
    last_relevant_layer = None

    for mod in reversed(list(model.modules())):
        if isinstance(mod, torch.nn.LSTM):
            return "lstm", "LSTM"
        elif isinstance(mod, torch.nn.Conv1d):
            return "cnn", "Conv1d"
        elif isinstance(mod, torch.nn.Conv2d):
            return "cnn", "Conv2d"

    # Fallbacks
    for mod in model.modules():
        if isinstance(mod, torch.nn.LSTM):
            last_relevant_layer = "LSTM"
        elif isinstance(mod, torch.nn.Conv1d):
            last_relevant_layer = "Conv1d"

    if last_relevant_layer:
        return "mixed", last_relevant_layer
    else:
        return "unknown", None


def reduce_anomaly_mask(all_errors, thresholds, model_type):
    """
    Reduces anomaly mask to [N, 1, L] based on model type for F1 score computation.

    Args:
        all_errors (Tensor): [N, C, L] for CNN or [N, L, C] for LSTM
        thresholds (array-like): per-channel threshold [C]
        model_type (str): 'cnn' or 'lstm'

    Returns:
        reduced_mask (Tensor): [N, 1, L]
    """
    if model_type == 'cnn':
        thresholds_tensor = torch.tensor(thresholds).view(1, -1, 1)  # [1, C, 1]
        mask = (all_errors > thresholds_tensor).int()               # [N, C, L]
        reduced = mask.any(dim=1, keepdim=True).int()               # [N, 1, L]

    elif model_type == 'lstm':
        thresholds_tensor = torch.tensor(thresholds).view(1, 1, -1)  # [1, 1, C]
        mask = (all_errors > thresholds_tensor).int()               # [N, L, C]
        reduced = mask.any(dim=2, keepdim=True).permute(0, 2, 1).int()  # [N, 1, L]

    else:
        raise ValueError(f"[❌ Error] Unknown model type: {model_type}")

    return reduced


def get_sync_config():
    cluster_resources = ray.cluster_resources()
    # Count how many nodes are in the cluster by checking "node" resource count
    num_nodes = cluster_resources.get("node", 0)

    if num_nodes <= 1:
        print("Single node detected - disabling syncer.")
        return SyncConfig(syncer=None)
    else:
        print(f"Multiple nodes detected ({num_nodes}) - enabling default syncer.")
        return SyncConfig()  # Default sync config, enables syncing