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


def extract_config(cfg_path=None, cfg=None, fine_tuning=False):

    assert cfg_path is not None or cfg is not None

    if cfg is None:
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

    if fine_tuning:
        # Extract checkpoint_path and fine_tuning from tune_config
        # These should have single values: tune.choice([value])
        checkpoint_path = config.get('opt.checkpoint_path')
        fine_tuning = config.get('opt.fine_tuning')

        cfg['model']['checkpoint_path'] = '/'.join(checkpoint_path.categories)
        cfg['opt']['fine_tuning'] = fine_tuning.categories[0]

    return config, cfg


def extract_fixed_config(cfg_path=None, cfg=None):

    assert cfg_path is not None or cfg is not None

    if cfg is None:
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


def merge_pretraining_finetuning_configs(pretraining_cfg, finetuning_cfg, output_path=None):
    """
    Merge pre-training and fine-tuning configs:
    1. If parameter is in BOTH tune_configs: keep fine-tuning multi-choice
    2. If parameter is ONLY in pre-training tune_config: add to fine-tuning but collapsed
       to the specific value from pre-training's model/opt/dataset sections
    3. Preserve fine-tuning specific keys (fine_tuning, checkpoint_path, exp_name, etc.)

    Args:
        pretraining_cfg: Pre-training config dictionary
        finetuning_cfg: Fine-tuning config dictionary
        output_path: Optional path to save the merged config

    Returns:
        Merged fine-tuning config
    """
    import copy

    # Make a deep copy to avoid modifying original
    finetuning_cfg = copy.deepcopy(finetuning_cfg)

    # Ensure tune_config exists
    if 'tune_config' not in finetuning_cfg:
        finetuning_cfg['tune_config'] = {}

    if 'tune_config' not in pretraining_cfg:
        print("Warning: No tune_config in pre-training config")
        return finetuning_cfg

    print("=" * 60)
    print("MERGING CONFIGS")
    print("=" * 60)

    # Define keys that should be preserved from fine-tuning config
    # These are fine-tuning specific and should NOT be overwritten by pre-training
    PRESERVE_FINETUNING_KEYS = {
        'model': ['checkpoint_path'],  # Where to load pretrained weights from
        'opt': ['fine_tuning', 'exp_name'],  # Fine-tuning flag and experiment name
        'dataset': ['name', 'data_path', 'feats', 'flag_col', 'align_data', 'detect_flag'],  # New dataset info
    }

    # Get all keys from pre-training tune_config
    pretraining_keys = set(pretraining_cfg['tune_config'].keys())
    finetuning_keys = set(finetuning_cfg['tune_config'].keys())

    # Keys in both configs
    common_keys = pretraining_keys & finetuning_keys
    # Keys only in pre-training
    only_pretraining_keys = pretraining_keys - finetuning_keys

    print(f"\n1. Parameters in BOTH configs (keeping fine-tuning multi-choice):")
    if common_keys:
        for key in sorted(common_keys):
            print(f"   ✓ {key}: {finetuning_cfg['tune_config'][key]}")
    else:
        print("   (none)")

    print(f"\n2. Parameters ONLY in pre-training (adding collapsed to fine-tuning):")

    # Process keys only in pre-training
    added_count = 0
    for key in sorted(only_pretraining_keys):
        # Parse the key to get section and parameter name
        # e.g., "model.num_layers" -> section="model", param="num_layers"
        parts = key.split('.', 1)
        if len(parts) != 2:
            print(f"   ⚠ Skipping malformed key: {key}")
            continue

        section, param_name = parts

        # Get the collapsed value from the corresponding section in pre-training config
        if section in pretraining_cfg and param_name in pretraining_cfg[section]:
            collapsed_value = pretraining_cfg[section][param_name]

            # Skip non-hyperparameter fields
            if param_name in ['name', 'aux_channels', 'output_size', 'parameter_count',
                              'checkpoint_path', 'opt_metric', 'metrics_to_report',
                              'other_reports', 'order_by', 'num_workers', 'evaluate_metrics',
                              'n_std', 'metrics_dataset_path', 'detect_anomaly_epoch_freq',
                              'exp_name', 'max_epochs', 'fine_tuning']:
                continue

            # Format the tune.choice with the collapsed value
            if isinstance(collapsed_value, str):
                tune_value = f"tune.choice(['{collapsed_value}'])"
            elif isinstance(collapsed_value, (int, float)):
                tune_value = f"tune.choice([{collapsed_value}])"
            else:
                # For other types, convert to string representation
                tune_value = f"tune.choice([{collapsed_value}])"

            # Add to fine-tuning config
            finetuning_cfg['tune_config'][key] = tune_value
            print(f"   + Added {key}: {tune_value}")
            added_count += 1
        else:
            print(f"   ⚠ Warning: {key} not found in pre-training '{section}' section")

    if added_count == 0:
        print("   (none)")

    # ==========================================
    # PRESERVE FINE-TUNING SPECIFIC KEYS
    # ==========================================
    print(f"\n3. Preserving fine-tuning specific keys:")
    preserved_count = 0
    for section, keys_to_preserve in PRESERVE_FINETUNING_KEYS.items():
        if section in finetuning_cfg:
            for key in keys_to_preserve:
                if key in finetuning_cfg[section]:
                    # Ensure this key is NOT overwritten by pre-training config
                    value = finetuning_cfg[section][key]
                    print(f"   🔒 {section}.{key}: {value}")
                    preserved_count += 1

    if preserved_count == 0:
        print("   (none)")

    # Copy non-tune_config sections from fine-tuning
    # This ensures dataset, model.checkpoint_path, opt.fine_tuning, etc. are preserved
    for section in ['dataset', 'model', 'opt', 'resources']:
        if section in finetuning_cfg:
            # Start with fine-tuning section
            merged_section = copy.deepcopy(finetuning_cfg[section])

            # For sections that need merging (like 'opt'), add missing keys from pretraining
            if section == 'opt' and section in pretraining_cfg:
                for key, value in pretraining_cfg[section].items():
                    # Only add if not already in fine-tuning AND not in preserve list
                    if key not in merged_section and key not in PRESERVE_FINETUNING_KEYS.get(section, []):
                        # Skip adding keys that are typically experiment-specific
                        if key not in ['exp_name', 'fine_tuning', 'max_epochs']:
                            merged_section[key] = value

            finetuning_cfg[section] = merged_section

    # Display final configuration organized by sections
    print("\n" + "=" * 60)
    print("FINAL FINE-TUNING tune_config:")
    print("=" * 60)

    # Sort keys by section
    all_keys = sorted(finetuning_cfg['tune_config'].keys())

    for section_name in ['model', 'opt', 'dataset']:
        section_keys = [k for k in all_keys if k.startswith(f'{section_name}.')]
        if section_keys:
            print(f"\n{section_name.upper()} parameters:")
            for key in section_keys:
                value = finetuning_cfg['tune_config'][key]
                # Check if it's multi-choice or collapsed
                # Multi-choice has comma outside quotes
                is_multi = value.count(',') > 0 and (
                        value.count(',') > value.count("','") or
                        ',' in value.split('[')[1].split(']')[0].replace("'", "").replace('"', '')
                )
                marker = "🔀" if is_multi else "📌"
                print(f"  {marker} {key}: {value}")

    # Display preserved keys from main config sections
    print("\n" + "=" * 60)
    print("PRESERVED FINE-TUNING SPECIFIC CONFIG:")
    print("=" * 60)

    if 'model' in finetuning_cfg and 'checkpoint_path' in finetuning_cfg['model']:
        print(f"\nMODEL:")
        print(f"  🔒 checkpoint_path: {finetuning_cfg['model']['checkpoint_path']}")

    if 'opt' in finetuning_cfg:
        print(f"\nOPT:")
        if 'fine_tuning' in finetuning_cfg['opt']:
            print(f"  🔒 fine_tuning: {finetuning_cfg['opt']['fine_tuning']}")
        if 'exp_name' in finetuning_cfg['opt']:
            print(f"  🔒 exp_name: {finetuning_cfg['opt']['exp_name']}")

    if 'dataset' in finetuning_cfg:
        print(f"\nDATASET:")
        print(f"  🔒 name: {finetuning_cfg['dataset']['name']}")
        print(f"  🔒 data_path: {finetuning_cfg['dataset']['data_path']}")
        print(f"  🔒 num_features: {len(finetuning_cfg['dataset'].get('feats', []))}")

    # Save to file if requested
    if output_path:
        import yaml
        with open(output_path, 'w') as f:
            yaml.dump(finetuning_cfg, f, default_flow_style=False, sort_keys=False)
        print(f"\n✅ Saved merged config to: {output_path}")

    return finetuning_cfg




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