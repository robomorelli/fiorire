# utils/general.py
from pathlib import Path
import ast
import re
from omegaconf import OmegaConf, DictConfig, ListConfig
from typing import Tuple
import numpy as np
import torch
import ray
from ray.tune.syncer import SyncConfig

from config import *


import os
import shutil
from ray.tune import Callback

class SaveBestMetricCheckpoint(Callback):
    """
    Save a checkpoint only when a given metric improves.
    Works with Ray Tune AIR (session.report).
    """

    def __init__(self, metric="val_loss", mode="min", checkpoint_dir="best_checkpoints"):
        self.metric = metric
        self.mode = mode
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_value = None

    def _is_better(self, current, best):
        if best is None:
            return True
        if self.mode == "min":
            return current < best
        else:  # "max"
            return current > best

    def on_trial_result(self, iteration, trials, trial, result, **info):
        # Skip if metric not present
        if self.metric not in result:
            return

        metric_value = result[self.metric]

        # Check improvement
        if not self._is_better(metric_value, self.best_value):
            return

        self.best_value = metric_value

        # The trial will have a latest_checkpoint, created in session.report
        ckpt = trial.checkpoint
        if ckpt is None:
            return

        # Save the improved checkpoint to a custom directory
        save_path = os.path.join(self.checkpoint_dir, f"{trial.trial_id}")
        os.makedirs(save_path, exist_ok=True)
        dst = os.path.join(save_path, "checkpoint")

        # Copy checkpoint directory
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(ckpt.path, dst)

        print(f"[Callback] Improved {self.metric}: {metric_value}. Saved checkpoint at: {dst}")


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

''' 
def resolve_paths(cfg, root):
    """
    Recursively resolve relative paths in config to absolute paths.
    """
    from pathlib import Path

    # ✅ Handle None root
    if root is None:
        root = Path.cwd()  # Use current working directory
    else:
        root = Path(root)

    root_dir = str(root.resolve())

    def _resolve(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if isinstance(v, str) and ('path' in k.lower() or 'dir' in k.lower()):
                    # Skip if already absolute
                    if not os.path.isabs(v):
                        abs_path = os.path.abspath(os.path.join(root_dir, v))
                        node[k] = abs_path
                elif isinstance(v, (dict, DictConfig)):
                    _resolve(v)
                elif isinstance(v, (list, ListConfig)):
                    _resolve(v)
        elif isinstance(node, (list, ListConfig)):
            for item in node:
                if isinstance(item, (dict, DictConfig, list, ListConfig)):
                    _resolve(item)

    _resolve(cfg)
    return cfg


def extract_config(cfg_path=None, cfg=None, fine_tuning=False):
    """
    Estrae configurazione da YAML e mappa a Ray Tune functions.
    Gestisce correttamente path multipli e stringhe quotate.
    """
    assert cfg_path is not None or cfg is not None

    if cfg is None:
        cfg = OmegaConf.load(cfg_path)

    config = {}

    for k, v in cfg.tune_config.items():
        # Extract function name (es. 'tune.choice')
        match = re.match(r'(\w+\.\w+)\((.*)\)', v, re.DOTALL)

        if not match:
            raise ValueError(f"Cannot parse tune config: {k} = {v}")

        func_name = match.group(1)
        args_str = match.group(2).strip()

        # Parse arguments using ast.literal_eval (gestisce stringhe quotate!)
        try:
            # ast.literal_eval converte stringa Python → oggetto Python
            # "[1, 2, 3]" → [1, 2, 3]
            # "['a', 'b']" → ['a', 'b']
            parsed_args = ast.literal_eval(args_str)

            # Se è già una lista, usala direttamente
            if isinstance(parsed_args, list):
                config[k] = ray_mapper[func_name](parsed_args)
            else:
                # Se è singolo valore, wrappalo in lista
                config[k] = ray_mapper[func_name]([parsed_args])

        except (ValueError, SyntaxError) as e:
            print(f"⚠️ Error parsing {k} = {v}")
            print(f"   Args string: {args_str}")
            raise e

    if fine_tuning:
        # Extract checkpoint_path and fine_tuning from tune_config
        checkpoint_path = config.get('opt.checkpoint_path')
        fine_tuning_flag = config.get('opt.fine_tuning')

        if checkpoint_path:
            # Se è un tune.choice con lista di path
            if hasattr(checkpoint_path, 'categories'):
                cfg['opt']['checkpoint_path'] = checkpoint_path.categories
            else:
                cfg['opt']['checkpoint_path'] = checkpoint_path

        if fine_tuning_flag:
            if hasattr(fine_tuning_flag, 'categories'):
                cfg['opt']['fine_tuning'] = fine_tuning_flag.categories[0]
            else:
                cfg['opt']['fine_tuning'] = fine_tuning_flag

    return config, cfg
'''

import re
from omegaconf import OmegaConf
import ray.tune as tune


def extract_config(cfg_path=None, cfg=None, fine_tuning=False):
    """
    Extract Ray Tune config from OmegaConf configuration.

    Args:
        cfg_path: Path to YAML config file (optional if cfg provided)
        cfg: OmegaConf object (optional if cfg_path provided)
        fine_tuning: Whether this is a fine-tuning config

    Returns:
        config: Dict with Ray Tune search spaces
        cfg: OmegaConf configuration
    """
    assert cfg_path is not None or cfg is not None, "Either cfg_path or cfg must be provided"

    if cfg is None:
        cfg = OmegaConf.load(cfg_path)

    # Ray tune function mapper
    ray_mapper = {
        'tune.choice': tune.choice,
        'tune.uniform': tune.uniform,
        'tune.loguniform': tune.loguniform,
        'tune.randint': tune.randint,
        'tune.quniform': tune.quniform,
        'tune.qloguniform': tune.qloguniform,
        'tune.randn': tune.randn,
        'tune.qrandn': tune.qrandn,
    }

    config = {}

    for k, v in cfg.tune_config.items():
        # ✅ FIX: Use regex to extract tune function name robustly
        # Matches: tune.choice, tune.uniform, etc. at the START of the string
        match = re.match(r'^(tune\.\w+)\s*\((.+)\)$', v.strip())

        if not match:
            raise ValueError(f"Invalid tune config format for '{k}': {v}")

        tune_func_name = match.group(1)  # e.g., "tune.choice"
        params_str = match.group(2)  # e.g., '["path/to/file.pt"]'

        # Check if tune function is valid
        if tune_func_name not in ray_mapper:
            raise ValueError(f"Unknown tune function '{tune_func_name}' for key '{k}'")

        tune_func = ray_mapper[tune_func_name]

        # ✅ Parse parameters based on tune function type
        if tune_func_name == 'tune.choice':
            # Parse list: ["item1", "item2", ...] or [1, 2, 3]
            params = parse_choice_params(params_str)
            config[k] = tune_func(params)

        elif tune_func_name in ['tune.uniform', 'tune.loguniform', 'tune.quniform', 'tune.qloguniform']:
            # Parse range: (min, max) or (min, max, q)
            params = parse_range_params(params_str)
            config[k] = tune_func(*params)

        elif tune_func_name in ['tune.randint', 'tune.randn', 'tune.qrandn']:
            # Parse simple params
            params = parse_simple_params(params_str)
            config[k] = tune_func(*params)

        else:
            # Fallback: try to eval (dangerous but backward compatible)
            try:
                config[k] = eval(v)
            except Exception as e:
                raise ValueError(f"Failed to parse '{k}': {v}\nError: {e}")

    # ✅ Handle fine-tuning specific extraction
    if fine_tuning:
        # Extract checkpoint_path and fine_tuning from tune_config
        checkpoint_path = config.get('opt.checkpoint_path')
        fine_tuning_flag = config.get('opt.fine_tuning')

        if checkpoint_path:
            # Extract single value from tune.choice
            if hasattr(checkpoint_path, 'categories'):
                cfg['model']['checkpoint_path'] = checkpoint_path.categories[0]
            else:
                cfg['model']['checkpoint_path'] = checkpoint_path

        if fine_tuning_flag:
            if hasattr(fine_tuning_flag, 'categories'):
                cfg['opt']['fine_tuning'] = fine_tuning_flag.categories[0]
            else:
                cfg['opt']['fine_tuning'] = fine_tuning_flag

    return config, cfg


def parse_choice_params(params_str):
    """
    Parse tune.choice parameters: ["item1", "item2"] or [1, 2, 3]

    Args:
        params_str: String like '["item1", "item2"]' or '[1, 2, 3]'

    Returns:
        List of parsed values
    """
    # Remove outer brackets
    params_str = params_str.strip()
    if params_str.startswith('[') and params_str.endswith(']'):
        params_str = params_str[1:-1]

    # Split by comma (but respect quoted strings)
    items = []
    current = []
    in_quotes = False
    quote_char = None

    for char in params_str + ',':  # Add comma at end to trigger last item
        if char in ['"', "'"]:
            if not in_quotes:
                in_quotes = True
                quote_char = char
            elif char == quote_char:
                in_quotes = False
                quote_char = None
        elif char == ',' and not in_quotes:
            item = ''.join(current).strip()
            if item:
                items.append(parse_value(item))
            current = []
            continue
        current.append(char)

    return items


def parse_range_params(params_str):
    """
    Parse range parameters: (min, max) or (min, max, q)

    Args:
        params_str: String like '0.0001, 0.001' or '0.0001, 0.001, 0.0001'

    Returns:
        Tuple of float values
    """
    # Remove parentheses if present
    params_str = params_str.strip().strip('()')

    # Split by comma
    parts = [p.strip() for p in params_str.split(',')]

    # Convert to float
    return tuple(float(p) for p in parts)


def parse_simple_params(params_str):
    """
    Parse simple parameters: single value or two values

    Args:
        params_str: String like '10' or '0, 10'

    Returns:
        Tuple of values
    """
    # Remove parentheses if present
    params_str = params_str.strip().strip('()')

    # Split by comma
    parts = [p.strip() for p in params_str.split(',')]

    # Try to convert to int, then float
    result = []
    for p in parts:
        try:
            result.append(int(p))
        except ValueError:
            result.append(float(p))

    return tuple(result) if len(result) > 1 else result[0]


def parse_value(value_str):
    """
    Parse a single value (string, int, or float).

    Args:
        value_str: String representation of value

    Returns:
        Parsed value (str, int, or float)
    """
    value_str = value_str.strip()

    # Remove quotes if present
    if (value_str.startswith('"') and value_str.endswith('"')) or \
            (value_str.startswith("'") and value_str.endswith("'")):
        return value_str[1:-1]

    # Try to parse as number
    try:
        # Try int first
        if '.' not in value_str:
            return int(value_str)
        else:
            return float(value_str)
    except ValueError:
        # Return as string
        return value_str


def extract_fixed_config(cfg_path=None, cfg=None):
    """
    Extract fixed config for testing/debugging (takes first value from each tune parameter).

    Args:
        cfg_path: Path to YAML config file (optional if cfg provided)
        cfg: OmegaConf object (optional if cfg_path provided)

    Returns:
        config: Dict with fixed values (first from each tune parameter)
        cfg: OmegaConf configuration
    """
    assert cfg_path is not None or cfg is not None, "Either cfg_path or cfg must be provided"

    if cfg is None:
        cfg = OmegaConf.load(cfg_path)

    config = {}

    for k, v in cfg.tune_config.items():
        # ✅ Handle list/ListConfig directly
        if isinstance(v, (list, ListConfig)):
            config[k] = v[0]  # Use first value

        # ✅ Handle string with tune.* function
        elif isinstance(v, str) and v.strip().startswith("tune."):
            # Use regex to extract function and parameters
            match = re.match(r'^tune\.\w+\s*\((.+)\)$', v.strip())

            if not match:
                raise ValueError(f"Invalid tune config format for '{k}': {v}")

            params_str = match.group(1)

            # Extract first value from the parameter list
            first_value = extract_first_value(params_str)
            config[k] = first_value

        else:
            # ✅ Fallback: use value as-is
            config[k] = v

    return config, cfg


def extract_first_value(params_str):
    """
    Extract the first value from tune parameter string.

    Handles:
    - tune.choice(["item1", "item2"]) → "item1"
    - tune.choice([1, 2, 3]) → 1
    - tune.uniform(0.001, 0.01) → 0.001
    - tune.randint(1, 10) → 1

    Args:
        params_str: Parameter string (e.g., '["item1", "item2"]' or '0.001, 0.01')

    Returns:
        First value parsed appropriately
    """
    params_str = params_str.strip()

    # ✅ Handle list parameters: ["item1", "item2"] or [1, 2, 3]
    if params_str.startswith('[') and params_str.endswith(']'):
        # Remove outer brackets
        inner = params_str[1:-1].strip()

        # Parse first item (respecting quotes)
        first_item = extract_first_list_item(inner)
        return parse_value(first_item)

    # ✅ Handle range parameters: (min, max) or min, max
    else:
        # Remove parentheses if present
        inner = params_str.strip('()')

        # Split by comma and take first
        parts = split_respecting_quotes(inner)
        if parts:
            return parse_value(parts[0])

        raise ValueError(f"Could not extract first value from: {params_str}")


def extract_first_list_item(list_content):
    """
    Extract first item from list content, respecting quoted strings.

    Args:
        list_content: String like '"item1", "item2"' or '1, 2, 3'

    Returns:
        First item as string
    """
    current = []
    in_quotes = False
    quote_char = None
    depth = 0  # Track nested brackets/parentheses

    for char in list_content:
        # Track quotes
        if char in ['"', "'"]:
            if not in_quotes:
                in_quotes = True
                quote_char = char
            elif char == quote_char:
                in_quotes = False
                quote_char = None

        # Track nesting
        elif not in_quotes:
            if char in ['[', '(']:
                depth += 1
            elif char in [']', ')']:
                depth -= 1
            elif char == ',' and depth == 0:
                # Found separator at top level - return current item
                return ''.join(current).strip()

        current.append(char)

    # Return accumulated item
    return ''.join(current).strip()


def split_respecting_quotes(text):
    """
    Split by comma but respect quoted strings.

    Args:
        text: String to split

    Returns:
        List of parts
    """
    parts = []
    current = []
    in_quotes = False
    quote_char = None

    for char in text + ',':  # Add comma to trigger last part
        if char in ['"', "'"]:
            if not in_quotes:
                in_quotes = True
                quote_char = char
            elif char == quote_char:
                in_quotes = False
                quote_char = None
        elif char == ',' and not in_quotes:
            part = ''.join(current).strip()
            if part:
                parts.append(part)
            current = []
            continue

        current.append(char)

    return parts


def parse_value(value_str):
    """
    Parse a single value (string, int, float, or path).

    Args:
        value_str: String representation of value

    Returns:
        Parsed value (str, int, or float)
    """
    value_str = value_str.strip()

    # Remove quotes if present
    if (value_str.startswith('"') and value_str.endswith('"')) or \
            (value_str.startswith("'") and value_str.endswith("'")):
        return value_str[1:-1]

    # Try to parse as number
    try:
        # Try int first
        if '.' not in value_str and 'e' not in value_str.lower():
            return int(value_str)
        else:
            return float(value_str)
    except ValueError:
        # Return as string (could be a path or identifier)
        return value_str


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
        try:
            print(f"  🔒 num_features: {len(finetuning_cfg['dataset'].get('feats', []))}")
        except:
            print("  🔒 num_features: (unable to determine) - gettig features automatically from data")

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
        mask = (all_errors.squeeze() > thresholds_tensor).int()               # [N, C, L]
        reduced = mask.any(dim=1, keepdim=True).int()               # [N, 1, L]

    elif model_type == 'lstm':
        thresholds_tensor = torch.tensor(thresholds).view(1, 1, -1)  # [1, 1, C]
        mask = (all_errors.squeeze() > thresholds_tensor).int()               # [N, L, C]
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
        return SyncConfig()
    else:
        print(f"Multiple nodes detected ({num_nodes}) - enabling default syncer.")
        return SyncConfig()  # Default sync config, enables syncing


def trial_dirname_creator(trial, max_params=5):
    """
    Custom trial directory name creator that uses underscores instead of commas

    For fine-tuning: creates trial directories under the pretrained model's directory
    For pre-training: creates trial directories normally

    Args:
        trial: Ray Tune trial object
        max_params: Maximum number of parameters to include in directory name (default: 5)
    """
    config = trial.config

    # Short names mapping for common parameters
    short_names = {
        'opt.lr': 'lr',
        'opt.batch_size': 'bs',
        'opt.epochs': 'ep',
        'opt.lr_patience': 'lrp',
        'opt.es_patience': 'esp',
        'dataset.perc_overlap': 'overlap',
        'dataset.scaler': 'scaler',
        'dataset.seq_in_length': 'seqlen',
        'model.activation': 'act',
        'model.bottleneck_activation': 'btl_act',
        'model.num_layers': 'layers',
        'model.base_filters': 'filters',
        'model.kernel_size': 'kernel',
        'model.compression_factor': 'comp',
        'model.double_deconv': 'dbl_deconv',
        'model.halve_time': 'halve_t',
        'model.halve_features': 'halve_f',
        'model.flattened': 'flat',
        'model.increasing': 'incr',
        'model.pool': 'pool',
        'model.stride': 'stride',
        'model.dilation': 'dilation',
    }

    # Check if fine-tuning mode
    is_fine_tuning = config.get('opt.fine_tuning', 0) == 1
    checkpoint_path = config.get('opt.checkpoint_path', '')

    # Build parameter string
    params = []
    for key, value in sorted(config.items()):
        # Skip Ray internal keys and non-varying parameters
        if key.startswith('_') or key in ['opt.checkpoint_path', 'opt.fine_tuning']:
            continue

        # Use short name if available, otherwise replace dots with underscores
        key_short = short_names.get(key, key.replace('.', '_'))

        # Format value based on type
        if isinstance(value, float):
            value_str = f"{value:.4f}"
        elif isinstance(value, str):
            value_str = value[:10]  # Truncate strings to 10 chars
        else:
            value_str = str(value)

        params.append(f"{key_short}={value_str}")

        # Stop once we reach max_params
        if len(params) >= max_params:
            break

    # Create base trial name
    param_string = "_".join(params) if params else "trial"
    base_name = f"{trial.trial_id}_{param_string}"

    # If fine-tuning, prepend the pretrained checkpoint directory
    if is_fine_tuning and checkpoint_path:
        import os
        checkpoint_dir = os.path.dirname(checkpoint_path)
        pretrained_trial_dir = os.path.dirname(checkpoint_dir)
        pretrained_trial_name = os.path.basename(pretrained_trial_dir)

        # Extract just the trial ID from pretrained name (last part after last underscore)
        pretrained_id = pretrained_trial_name.split('_')[-1]

        # Shorter prefix with just the pretrained trial ID
        return f"FT_{pretrained_id}__{base_name}"

    return base_name


def get_finetuning_local_dir(checkpoint_path, date_str):
    """
    Extract the local directory for fine-tuning based on pretrained checkpoint path

    Args:
        checkpoint_path: Path to pretrained checkpoint
                        Example: .../8efbc_00000_params/checkpoint_000001/model.pt
        date_str: Date string for unique directory naming

    Returns:
        local_dir: Path where fine-tuning experiments will be saved
                   Example: .../8efbc_00000_params/fine_tuning_from_8efbc_00000_2025-10-31
        pretrained_trial_id: ID of the pretrained trial
    """
    import os

    # Extract directory structure from checkpoint path
    checkpoint_dir = os.path.dirname(checkpoint_path)  # Remove model.pt
    pretrained_trial_dir = os.path.dirname(checkpoint_dir)  # Remove checkpoint_000001
    pretrained_trial_name = os.path.basename(pretrained_trial_dir)  # Get trial name

    # Extract trial ID (first two parts: hash_id)
    pretrained_trial_id = "_".join(pretrained_trial_name.split('_')[:2])

    # Create fine-tuning directory INSIDE the pretrained trial directory
    local_dir = os.path.join(pretrained_trial_dir, f'fine_tuning_from_{pretrained_trial_id}_{date_str}')

    return local_dir, pretrained_trial_id


def clean_loaded_cfg(loaded_cfg):

    # ✅ FIX: Remove opt.config_file_path from loaded_cfg
    # This field is not needed in fine-tuning and causes parsing issues
    if 'tune_config' in loaded_cfg and 'opt.config_file_path' in loaded_cfg['tune_config']:
        print("⚠️  Removing 'opt.config_file_path' from pretrained config (not needed for fine-tuning)")
        del loaded_cfg['tune_config']['opt.config_file_path']

    # Also remove from opt section if present
    if 'opt' in loaded_cfg and 'config_file_path' in loaded_cfg['opt']:
        del loaded_cfg['opt']['config_file_path']

    # Now safe to extract config
    _, cfg_pre = extract_config(cfg_path=None, cfg=loaded_cfg)

    return loaded_cfg

'''

## Struttura risultante:
```
ray_results/
└── conv_ae2D_fiorire_2025-10-30_15-41-45/
    └── conv_ae2D_fiorire/
        └── 8efbc_00000_overlap=0.5000_scaler=StandardSc_seqlen=16_act=Relu_filters=32/
            ├── checkpoint_000001/
            │   └── model.pt
            ├── checkpoint_000002/
            │   └── model.pt
            └── fine_tuning_from_8efbc_00000_2025-10-30_17-00-56/  # 👈 Dentro il trial!
                └── conv_ae2D_fiorire_fine_tuning/
                    ├── 5f309_00000_lr=0.0009_lrp=5_overlap=0.2/
                    ├── 5f309_00001_lr=0.0001_lrp=10_overlap=0.5/
                    └── 5f309_00002_lr=0.0009_lrp=10_overlap=0.2/
'''