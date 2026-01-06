from sklearn.metrics import roc_curve, auc, f1_score
from utils.general import resolve_paths, infer_model_type, reduce_anomaly_mask
from omegaconf import OmegaConf, ListConfig
import torch
import numpy as np
from tqdm import tqdm
from typing import List, Optional
import types
import torch.nn.functional as F
import traceback

from config import *

class EarlyStopping:
    """
    Early stopping to stop the training when a monitored metric does not improve after
    a given patience. Supports both 'min' (loss) and 'max' (accuracy/F1).
    """
    def __init__(self, patience=5, min_delta=0, opt_metric_dict=None):
        self.patience = patience
        self.min_delta = min_delta

        self.metric_key = opt_metric_dict.get("metric_key", "val_loss")
        self.mode = opt_metric_dict.get("mode", "min")
        self.best_metric = opt_metric_dict.get(
            "best_metric",
            float("inf") if self.mode == "min" else -float("inf")
        )

        if self.mode not in ["min", "max"]:
            raise ValueError("mode must be 'min' or 'max'")

        self.counter = 0
        self.early_stop = False

    def __call__(self, current_metric):
        """
        Update state with current metric.
        Returns:
            improved (bool): True if metric improved, False otherwise
            best_metric (float): Updated best metric so far
        """
        improved = False

        if self.mode == "min":
            improved = (self.best_metric - current_metric) > self.min_delta
        else:  # mode == "max"
            improved = (current_metric - self.best_metric) > self.min_delta

        if improved:
            self.best_metric = current_metric
            self.counter = 0
        else:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print("INFO: Early stopping")
                self.early_stop = True

        return improved, self.best_metric


# Global cache for base configs (loaded once per file)
_BASE_CONFIG_CACHE = {}


def model_setup(config_file_name, config, root=None):
    """
    Setup model configuration with caching to prevent reloading from disk.

    ⚠️ NOTE: 'shared_config' must be extracted BEFORE OmegaConf merge!
    """
    import os
    from omegaconf import OmegaConf
    global _BASE_CONFIG_CACHE

    cache_key = config_file_name

    # ✅ Extract shared_config FIRST (don't pass to OmegaConf)
    shared_config = config.pop('shared_config', None) if config else None

    # Load base config ONCE (or use cached)
    if cache_key not in _BASE_CONFIG_CACHE:
        print(f"📂 [PID {os.getpid()}] Loading and caching base config: {config_file_name}")
        base_cfg = OmegaConf.load(config_path + config_file_name)
        _BASE_CONFIG_CACHE[cache_key] = OmegaConf.to_container(base_cfg, resolve=True)
    else:
        print(f"📌 [PID {os.getpid()}] Using cached base config: {config_file_name}")

    # Clone from cache
    cfg = OmegaConf.create(_BASE_CONFIG_CACHE[cache_key])
    OmegaConf.set_struct(cfg, False)

    # Merge Ray Tune parameters (WITHOUT shared_config)
    if config and len(config) > 0:
        print(f"🔀 [PID {os.getpid()}] Merging frozen Ray Tune parameters")

        for k, v in config.items():
            OmegaConf.update(cfg, k, v, merge=True)

        try:
            print(f"   ✓ Final opt.lr = {cfg.opt.get('lr', 'N/A')}")
        except:
            pass

    # Resolve paths
    cfg = resolve_paths(cfg, root)
    shared_config = resolve_paths(shared_config, root)

    # ✅ Return both cfg and shared_config separately
    return cfg, shared_config


def update_input_output(cfg):
    """
    Infer input and output dimensions from the configuration.
    """
    if isinstance(cfg.dataset.feats, (list, ListConfig)):
        feats = cfg.dataset.feats
    else:
        feats = [cfg.dataset.feats]

    # Handle 'target'
    if isinstance(cfg.dataset.target, (list, ListConfig)):
        target = cfg.dataset.target
    elif isinstance(cfg.dataset.target, str):
        target = [cfg.dataset.target]
    else:
        target = None

    # Handle remove_columns
    if cfg.opt.get("remove_columns", False):
        remove_columns_val = cfg.opt.get("remove_columns")
        if isinstance(remove_columns_val, (list, ListConfig)):
            remove_columns = remove_columns_val
        elif isinstance(remove_columns_val, str):
            remove_columns = [remove_columns_val]
        else:
            remove_columns = []
    else:
        remove_columns = []

    # Update cfg
    cfg.dataset.feats = feats
    cfg.dataset.target = target
    cfg.opt.remove_columns = remove_columns
    seq_out_length = cfg.dataset.get('seq_out_length', None)
    if seq_out_length  is None:
        cfg.dataset.seq_out_length  = cfg.dataset.seq_in_length

    return cfg, feats, target


def compute_indices_with_overlap(base_indices, overlap, seq_len):
    """
    Compute final indices with overlap, respecting chunk boundaries.

    Args:
        base_indices: Base indices (may have gaps/chunks)
        overlap: Percentage overlap (0.0 to 1.0)
        seq_len: Sequence length

    Returns:
        final_indices: Indices with overlap applied
    """
    import numpy as np

    # Ensure numpy array
    if not isinstance(base_indices, np.ndarray):
        base_indices = np.array(base_indices)

    # Calculate step
    step = max(1, int(seq_len * (1 - overlap)))

    # ✅ Detect chunk boundaries (where diff > 1)
    diffs = np.diff(base_indices)
    chunk_breaks = np.where(diffs > 1)[0] + 1

    # Split into chunks
    chunks = np.split(base_indices, chunk_breaks)

    final_indices = []

    print(f"\n{'=' * 60}")
    print(f"DEBUG: compute_indices_with_overlap")
    print(f"{'=' * 60}")
    print(f"Total base indices: {len(base_indices):,}")
    print(f"Step: {step}")
    print(f"Chunks detected: {len(chunks)}")
    print(f"{'=' * 60}\n")

    total_obtained = 0

    for i, chunk in enumerate(chunks):
        if len(chunk) == 0:
            continue

        # ✅ subsample chunk positions
        chunk_final = chunk[::step]

        expected = len(chunk) // step
        obtained = len(chunk_final)
        total_obtained += obtained

        print(
            f"Chunk {i:2d}: len={len(chunk):7,} | expected={expected:5,} | obtained={obtained:5,} | {'✓' if obtained >= expected else '✗'}")

        final_indices.append(chunk_final)

    print(f"\n{'=' * 60}")
    print(f"Total expected:  {len(base_indices) // step:,}")
    print(f"Total obtained:  {total_obtained:,}")
    print(f"Difference:      {abs(len(base_indices) // step - total_obtained):,}")
    print(f"{'=' * 60}\n")

    # Concatenate all chunks
    if len(final_indices) > 0:
        return np.concatenate(final_indices)
    else:
        return np.array([], dtype=np.int64)



def infer_metric_mode(metric_name: str) -> str:
    name = metric_name.lower()

    # Abbiniamo pattern → mode
    for key, mode in DEFAULT_METRIC_MODES.items():
        if key in name:
            return mode

    # fallback: loss non matchato sopra
    if "loss" in name:
        return "min"

    return "max"  # default sicuro

def get_opt_metric(cfg, metrics_loader=None, available_metrics=None):
    """
    Determines the optimization metric for early stopping or tuning.
    Returns dict: {"metric_key":..., "mode":..., "best_metric":...}
    """

    if available_metrics is None:
        available_metrics = PREFERRED_METRICS + ["val_loss"]

    metric_key = None
    mode = None

    # -------------------------------
    # Case 1 → opt_metric specified
    # -------------------------------
    if cfg.opt.opt_metric:
        opt_metric_dict = cfg.opt.opt_metric
        if not isinstance(vars(opt_metric_dict), dict):
            raise ValueError(f"opt_metric must be a dict, got {type(opt_metric_dict)}")

        # Take first key:mode pair
        metric_key = list(opt_metric_dict.keys())[0]
        mode_from_cfg = list(opt_metric_dict.values())[0] if list(opt_metric_dict.values())[0] else None

        # Validate metric_key
        if metric_key not in available_metrics:
            print(f"[WARN] Metric '{metric_key}' not valid → fallback to 'val_loss'")
            metric_key = "val_loss"
            mode = "min"
        else:
            # Infer mode if missing
            default_mode = DEFAULT_METRIC_MODES.get(metric_key.replace("val_", ""), "min")
            if mode_from_cfg is None:
                mode = default_mode
            else:
                mode = mode_from_cfg
                # Check consistency with default
                if mode != default_mode:
                    print(f"[WARNING] Provided mode '{mode}' for metric '{metric_key}' differs from standard '{default_mode}'. Using '{default_mode}' instead.")
                    mode = default_mode

        # Check if metrics_loader required but missing
        if metric_key != "val_loss" and metrics_loader is None:
            print(f"[WARN] Metric '{metric_key}' requires metrics_loader → fallback 'val_loss'")
            metric_key = "val_loss"
            mode = "min"

    # -------------------------------
    # Case 2 → opt_metric not specified
    # -------------------------------
    else:
        # Try preferred metrics
        for m in PREFERRED_METRICS:
            if m in cfg.opt.metrics_to_report:
                metric_key = m
                break
        if metric_key is None:
            metric_key = "val_loss"

        # Infer mode from DEFAULT_METRIC_MODES
        mode = DEFAULT_METRIC_MODES.get(metric_key.replace("val_", ""), "min")

        # If metrics_loader necessary but missing
        if metric_key != "val_loss" and metrics_loader is None:
            print(f"[WARN] Metric '{metric_key}' requires metrics_loader → fallback 'val_loss'")
            metric_key = "val_loss"
            mode = "min"

    # -------------------------------
    # Initialize best_metric
    # -------------------------------
    best_metric = float("inf") if mode == "min" else -float("inf")

    return {
        "metric_key": metric_key,
        "mode": mode,
        "best_metric": best_metric
    }

def get_opt_metric_kp(cfg, metrics_loader=None):
    opt_metric_dict = {}
    metric_key, mode = list(cfg.opt.opt_metric.items())[0]

    if metric_key in available_metrics and mode in available_modes:
        pass
    else:
        print(f"Warning: opt_metric is set to {metric_key} but is not in available_metrics. Setting opt_metric to 'loss'.")
        metric_key = 'val_loss'
        mode = 'min'

    if metric_key != 'val_loss' and metrics_loader is None:
        print(f"Warning: opt_metric is set to {metric_key} but no metrics_loader is provided. Setting opt_metric to 'loss'.")
        metric_key = 'val_loss'
        mode = 'min'

    best_metric = float("inf") if mode == "min" else -float("inf")
    opt_metric_dict["metric_key"] = metric_key
    opt_metric_dict["mode"] = mode
    opt_metric_dict["best_metric"] = best_metric

    return  opt_metric_dict

def compute_errors(outputs: torch.Tensor, targets: torch.Tensor, error_type: str = "abs"):
    """
    Computes the requested reconstruction error.

    Args:
        outputs: model outputs with shape [N, C, L]
        targets: ground-truth targets with shape [N, C, L]
        error_type: "abs" for mean absolute error (MAE),
                    "se" for squared error

    Returns:
        errors: tensor of element-wise errors with shape [N, C, L]
    """
    if error_type == "abs":
        # Absolute error (L1)
        return torch.abs(outputs.detach() - targets)
    elif error_type == "se":
        # Squared error (L2)
        return (outputs.detach() - targets) ** 2
    else:
        raise ValueError(
            f"Unknown error_type='{error_type}', must be 'abs' or 'se'"
        )


def mean_std_per_channel(errors: torch.Tensor, model_type: str, last_layer:str = None):
    if model_type == "cnn":
        if last_layer == "Conv2d":
            if errors.dim() == 4:
                if errors.shape[1] == 1:
                    errors = errors.squeeze(1)
                    mean = errors.mean(dim=(0, 2)).numpy()
                    std = errors.std(dim=(0, 2)).numpy()
                else:
                    mean = errors.mean(dim=(0, 3)).numpy()
                    std = errors.std(dim=(0, 3)).numpy()
        else:
            mean = errors.mean(dim=(0, 2)).numpy()
            std = errors.std(dim=(0, 2)).numpy()
    elif model_type == "lstm":
        mean = errors.mean(dim=(0, 1)).numpy()
        std = errors.std(dim=(0, 1)).numpy()
    else:
        raise ValueError(f"Unknown model type {model_type}")
    return mean, std

def train_one_epoch(model, dataloader, criterion, optimizer, device, desc="Train"):
    model.train()
    epoch_loss = 0
    all_errors = []

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=desc, leave=False)
    for i, (inputs, targets, is_anomaly) in pbar:

        inputs, targets = inputs.to(device), targets.to(device)
        # conv ae 1d torch.Size([100, 16, 8]), torch.Size([100, 16, 8]), torch.Size([100, 1, 8])
        # conv ae 2d torch.Size([100, 1, 16, 8]), torch.Size([100, 1, 16, 8]), torch.Size([100, 1, 8])
        optimizer.zero_grad()
        outputs = model(inputs).to(device)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # Compute absolute error (reconstruction error)
        errors = compute_errors(outputs, targets)
        all_errors.append(errors.cpu())

        epoch_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    # Stack all errors: [N, C, L]
    all_errors = torch.cat(all_errors, dim=0)
    model_type, last_layer = infer_model_type(model)
    channel_mean_errors, channel_std_errors = mean_std_per_channel(all_errors, model_type, last_layer)

    return {
        "train_loss": epoch_loss / len(dataloader),
        "anomaly_threshold": {
            "channel_means": channel_mean_errors,  # shape: [C]
            "channel_stds": channel_std_errors  # shape: [C]
        },
    }

@torch.no_grad()
def validate_one_epoch(
    cfg,
    model,
    dataloader: Optional[torch.utils.data.DataLoader],
    metric_loader: Optional[torch.utils.data.DataLoader],
    criterion,
    device,
    desc: str = "Validation",
    evaluate_metrics: bool = True,
    normal_anomalous_ratio: int = 1,
    num_thresholds=50,
    use_error:str = "abs" ):

    model.eval()
    epoch_loss = 0
    all_errors = []

    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=desc, leave=False)
        for i, (inputs, targets, is_anomaly) in pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs).to(device)
            loss = criterion(outputs, targets)

            errors = compute_errors(outputs, targets, error_type=use_error)
            all_errors.append(errors.cpu())

            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

    all_errors = torch.cat(all_errors, dim=0)  # [N, C, L]

    # Core validation results
    results = {
        "val_loss": epoch_loss / len(dataloader),
    }

    if cfg.opt.get('use_val_normal_errors', 0) and cfg.dataset.get('val_overlap', 0) == 0 \
        and cfg.opt.get('anomaly_strategy', None) == "corrupt_validation" and cfg.opt.corruption_config.get('corruption_ratio', None) == 1 \
        and cfg.opt.get('metric_seq_overlap', 0):
        print('USING NORMAL ERRORS FROM VAL')
    else:
        print('Computing normal errors from metric loader')
        all_errors = None

    # Optionally evaluate anomaly detection metrics
    if evaluate_metrics:
        print("\n[INFO] Evaluating anomaly detection metrics...with error =", use_error)
        print("\n[INFO] F1 score is computed approximatively don'0t trust")
        test_results, indices = test_anomaly_step(
            model=model,
            metric_dataloader=metric_loader,
            device=device,
            external_normal_errors=all_errors,
            num_thresh=num_thresholds,
            epsilon=1e-5,
            desc=f"Testing anomalies ({use_error})",
            normal_anomalies_ratio=normal_anomalous_ratio,
            seed=123,
            use_error=use_error
        )

        # Flatten only necessary fields for logging
        if "metrics_results" in test_results:
            metrics = test_results["metrics_results"]
            results.update({
                "val_f1_score": metrics["val_f1_score"],   #best among all the possible threhsold
                "val_roc_auc": metrics["val_roc_auc"],
                "val_fpr": metrics["val_fpr"],
                "val_tpr": metrics["val_tpr"],
                #"val_best_thresh_youden": metrics["val_best_thresh_youden"],
                "val_best_thresh_f1": metrics["val_best_thresh_f1"],
                #"best_n_std": metrics["best_n_std"],
                #"channel_means": metrics["channel_means"],
                #"channel_stds": metrics["channel_stds"],
                # optionally more if needed for post-analysis
                # "channel_thresholds": metrics["channel_thresholds"]
            })

    else:
        indices = None

    return results, indices

@torch.no_grad()
def test_anomaly_step(
        model,
        metric_dataloader,
        device="cuda",
        external_normal_errors=None,
        num_thresh=100,
        epsilon=1e-3,
        desc="Testing anomalies",
        normal_anomalies_ratio=1,
        seed=123,
        use_error="abs"
):
    """
    Compute anomaly detection metrics with comprehensive NaN/Inf checking.

    Args:
        model: The trained model
        metric_dataloader: DataLoader with metric dataset (normal + anomalies)
        device: Device to run on
        external_normal_errors: Pre-computed normal errors (optional)
        num_thresh: Number of thresholds (deprecated, using Youden's index)
        epsilon: Small constant for normalization stability
        desc: Progress bar description
        normal_anomalies_ratio: Ratio of normal to anomaly sequences
        seed: Random seed for sampling
        use_error: "abs" or "se" for error computation

    Returns:
        metrics_dict: Dictionary with all computed metrics
    """
    model.eval()
    anomaly_errors_list = []
    anomaly_masks_list = []

    should_compute_normal_errors = (external_normal_errors is None)
    normal_errors_list = [] if should_compute_normal_errors else None

    normal_running_sum = 0.0
    normal_running_count = 0
    anom_running_sum = 0.0
    anom_running_count = 0

    # ✅ DIAGNOSTIC counters
    nan_count_recon = 0
    nan_count_errors = 0

    normal_bar = tqdm(total=0, position=0, leave=True, desc="Normals")
    anom_bar = tqdm(total=0, position=1, leave=True, desc="Anomalies")

    model_type, last_layer = infer_model_type(model)

    print(f"\n[TEST_ANOMALY] Starting - use_error: {use_error}, epsilon: {epsilon}")

    # ==============================
    # 1) PASS THROUGH DATALOADER
    # ==============================
    for batch_idx, batch in enumerate(tqdm(metric_dataloader, desc=desc, position=2)):
        x, target, mask, *_ = batch
        x, target = x.to(device), target.to(device)

        is_anom = mask.view(mask.size(0), -1).sum(dim=1) > 0
        is_norm = ~is_anom

        # ==================
        # Anomalies
        # ==================
        if is_anom.any():
            x_anom, target_anom, mask_anom = x[is_anom], target[is_anom], mask[is_anom]
            recon = model(x_anom)

            # ✅ CHECK: Reconstruction (anomalies)
            if torch.isnan(recon).any() or torch.isinf(recon).any():
                nan_count_recon += 1
                if nan_count_recon == 1:
                    print(f"\n⚠️  [TEST_ANOMALY] NaN/Inf in RECONSTRUCTION at batch {batch_idx}!")
                    print(f"   - Type: Anomalies")
                    print(f"   - NaN: {torch.isnan(recon).sum().item()}")
                    print(f"   - Inf: {torch.isinf(recon).sum().item()}")

            err_anom = compute_errors(target_anom, recon, error_type=use_error).cpu()

            # ✅ CHECK: Errors (anomalies)
            if torch.isnan(err_anom).any() or torch.isinf(err_anom).any():
                nan_count_errors += 1
                if nan_count_errors == 1:
                    print(f"\n⚠️  [TEST_ANOMALY] NaN/Inf in ERRORS at batch {batch_idx}!")
                    print(f"   - Type: Anomalies")
                    print(f"   - Error type: {use_error}")
                    print(f"   - NaN: {torch.isnan(err_anom).sum().item()}")
                    print(f"   - Inf: {torch.isinf(err_anom).sum().item()}")

            if last_layer == "Conv2d":
                err_anom = torch.squeeze(err_anom, (1))

            anomaly_errors_list.append(err_anom)
            anomaly_masks_list.append(mask_anom.cpu())

            # Update progress
            batch_mean = err_anom.mean().item()
            anom_running_sum += err_anom.sum().item()
            anom_running_count += err_anom.numel()
            global_mean = anom_running_sum / anom_running_count
            anom_bar.set_postfix({"batch_mean": f"{batch_mean:.6f}",
                                  "global_mean": f"{global_mean:.6f}"})
            anom_bar.update(1)

        # ==================
        # Normals
        # ==================
        if is_norm.any() and should_compute_normal_errors:
            x_norm, target_norm = x[is_norm], target[is_norm]
            recon = model(x_norm)

            # ✅ CHECK: Reconstruction (normals)
            if torch.isnan(recon).any() or torch.isinf(recon).any():
                nan_count_recon += 1
                if nan_count_recon == 1:
                    print(f"\n⚠️  [TEST_ANOMALY] NaN/Inf in RECONSTRUCTION at batch {batch_idx}!")
                    print(f"   - Type: Normals")
                    print(f"   - NaN: {torch.isnan(recon).sum().item()}")
                    print(f"   - Inf: {torch.isinf(recon).sum().item()}")

            err_norm = compute_errors(target_norm, recon, error_type=use_error).cpu()

            # ✅ CHECK: Errors (normals)
            if torch.isnan(err_norm).any() or torch.isinf(err_norm).any():
                nan_count_errors += 1
                if nan_count_errors == 1:
                    print(f"\n⚠️  [TEST_ANOMALY] NaN/Inf in ERRORS at batch {batch_idx}!")
                    print(f"   - Type: Normals")
                    print(f"   - Error type: {use_error}")
                    print(f"   - NaN: {torch.isnan(err_norm).sum().item()}")
                    print(f"   - Inf: {torch.isinf(err_norm).sum().item()}")

            if last_layer == "Conv2d":
                err_norm = torch.squeeze(err_norm, (1))

            normal_errors_list.append(err_norm)

            # Update progress
            batch_mean = err_norm.mean().item()
            normal_running_sum += err_norm.sum().item()
            normal_running_count += err_norm.numel()
            global_mean = normal_running_sum / normal_running_count
            normal_bar.set_postfix({"batch_mean": f"{batch_mean:.6f}",
                                    "global_mean": f"{global_mean:.6f}"})
            normal_bar.update(1)

    normal_bar.close()
    anom_bar.close()

    print(f"\n[TEST_ANOMALY] Forward pass complete:")
    print(f"   - Batches with NaN/Inf reconstruction: {nan_count_recon}")
    print(f"   - Batches with NaN/Inf errors: {nan_count_errors}")

    anomaly_errors = torch.cat(anomaly_errors_list, dim=0)  # [N_anom, C, L]
    anomaly_masks = torch.cat(anomaly_masks_list, dim=0)  # [N_anom, 1, L]

    # ==============================
    # 2) NORMAL ERROR SOURCE
    # ==============================
    if should_compute_normal_errors:
        # From loader
        normal_errors_all = torch.cat(normal_errors_list, dim=0)
        normal_error_source = "loader"
    else:
        # From external
        if last_layer == "Conv2d":
            external_normal_errors = torch.squeeze(external_normal_errors, (1))
        normal_errors_all = external_normal_errors.cpu()
        normal_error_source = "external"

    # ==============================
    # 3) SAMPLE NORMAL SEQUENCES
    # ==============================
    N_anom = anomaly_errors.shape[0]
    N_norm_needed = int(N_anom * normal_anomalies_ratio)
    N_norm_needed = min(N_norm_needed, normal_errors_all.shape[0])

    np.random.seed(seed)
    idx_main = np.random.permutation(normal_errors_all.shape[0])[:N_norm_needed]
    normal_errors = normal_errors_all[idx_main]

    print(f"\n[INFO] Selected anomalous sequences: {N_anom}")
    print(f"[INFO] Selected normal sequences:    {normal_errors.shape[0]}")
    print(f"[INFO] Normal error source:          {normal_error_source}")

    # ==============================
    # 4) DISTRIBUTIONS
    # ==============================
    def compute_error_stats(errors_tensor):
        flat = errors_tensor.flatten().numpy()
        return {
            "mean": flat.mean(),
            "std": flat.std(),
            "min": flat.min(),
            "max": flat.max(),
            "q1": np.quantile(flat, 0.01),
            "q5": np.quantile(flat, 0.05),
            "q25": np.quantile(flat, 0.25),
            "median": np.quantile(flat, 0.50),
            "q75": np.quantile(flat, 0.75),
            "q95": np.quantile(flat, 0.95)
        }

    normal_stats = compute_error_stats(normal_errors)
    anomaly_stats = compute_error_stats(anomaly_errors)

    # ==============================
    # 5) NORMALIZATION
    # ==============================
    normal_perm = normal_errors.permute(0, 2, 1)  # [N, L, C]
    anomaly_perm = anomaly_errors.permute(0, 2, 1)

    C = normal_perm.shape[2]

    flat_norm = normal_perm.reshape(-1, C).float()
    normalization_factor = torch.quantile(flat_norm, 0.5, dim=0)

    # ✅ CHECK: Normalization factor
    if torch.isnan(normalization_factor).any() or torch.isinf(normalization_factor).any():
        print(f"\n⚠️  [NORMALIZATION] NaN/Inf in NORMALIZATION_FACTOR!")
        print(f"   - NaN: {torch.isnan(normalization_factor).sum().item()}")
        print(f"   - Inf: {torch.isinf(normalization_factor).sum().item()}")
        print(f"   - Values: {normalization_factor}")

    norm = normalization_factor.view(1, 1, C) + epsilon

    normal_norm = normal_perm / norm
    anomaly_norm = anomaly_perm / norm

    assert not torch.isnan(normal_norm).any() and not torch.isinf(normal_norm).any(), "NaN/Inf in normal_norm!"
    assert not torch.isnan(anomaly_norm).any() and not torch.isinf(anomaly_norm).any(), "NaN/Inf in anomaly_norm!"

    # ✅ CHECK 1: Post-normalization NaN/Inf
    nan_normal = torch.isnan(normal_norm).sum().item()
    inf_normal = torch.isinf(normal_norm).sum().item()
    nan_anomaly = torch.isnan(anomaly_norm).sum().item()
    inf_anomaly = torch.isinf(anomaly_norm).sum().item()

    if nan_normal > 0 or inf_normal > 0 or nan_anomaly > 0 or inf_anomaly > 0:
        print(f"\n⚠️  [NORMALIZATION] NaN/Inf detected after normalization!")
        print(f"   - Normal errors:  NaN={nan_normal}, Inf={inf_normal}")
        print(f"   - Anomaly errors: NaN={nan_anomaly}, Inf={inf_anomaly}")
        print(
            f"   - Normalization factor: min={normalization_factor.min().item():.6f}, max={normalization_factor.max().item():.6f}")
        print(f"   - Epsilon: {epsilon}")

    masks_norm = torch.zeros((normal_norm.shape[0], anomaly_masks.shape[1], anomaly_masks.shape[2]), dtype=torch.int)

    all_errors = torch.cat([normal_norm, anomaly_norm], dim=0)  # [N, L, C]
    all_masks = torch.cat([masks_norm, anomaly_masks], dim=0)  # [N, L, 1]

    N, T, C = all_errors.shape

    # ==============================
    # 6) STEP-WISE SCORE
    # ==============================
    if use_error == "abs":
        val_anomaly_scores = all_errors.mean(dim=2)  # mean over features
    else:  # "se"
        val_anomaly_scores = all_errors.mean(dim=2)

    val_labels = (all_masks.view(N, T, -1).sum(dim=2) > 0).int()  # [N, T]

    flat_scores = val_anomaly_scores.flatten().numpy()
    flat_labels = val_labels.flatten().numpy()

    # ✅ CHECK 2: Flat scores NaN/Inf (before ROC)
    nan_scores = np.isnan(flat_scores).sum()
    inf_scores = np.isinf(flat_scores).sum()
    total_scores = len(flat_scores)

    if nan_scores > 0 or inf_scores > 0:
        print(f"\n⚠️  [SCORES] NaN/Inf detected in anomaly scores!")
        print(f"   - NaN: {nan_scores} / {total_scores} ({100 * nan_scores / total_scores:.2f}%)")
        print(f"   - Inf: {inf_scores} / {total_scores} ({100 * inf_scores / total_scores:.2f}%)")
        print(f"   - This will corrupt ROC/AUC calculation!")

    nan_scores = np.isnan(flat_labels).sum()
    inf_scores = np.isinf(flat_labels).sum()
    total_scores = len(flat_labels)

    if nan_scores > 0 or inf_scores > 0:
        print(f"\n⚠️  [SCORES] NaN/Inf detected in anomaly scores!")
        print(f"   - NaN: {nan_scores} / {total_scores} ({100 * nan_scores / total_scores:.2f}%)")
        print(f"   - Inf: {inf_scores} / {total_scores} ({100 * inf_scores / total_scores:.2f}%)")
        print(f"   - This will corrupt ROC/AUC calculation!")


    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(flat_labels, flat_scores)
    roc_auc = auc(fpr, tpr)

    # ==============================
    # F1 ESTIMATION FROM TPR/FPR
    # ==============================
    n_pos = int(flat_labels.sum())
    n_neg = len(flat_labels) - n_pos

    # Use Youden's index
    ix_youden = np.argmax(tpr - fpr)
    rec_est = tpr[ix_youden]
    fpr_val = fpr[ix_youden]

    prec_est = (rec_est * n_pos) / (rec_est * n_pos + fpr_val * n_neg + 1e-12)
    f1_est = 2 * (prec_est * rec_est) / (prec_est + rec_est + 1e-12)

    print(f"\n[TEST_ANOMALY] Metrics computed successfully:")
    print(f"   - ROC AUC: {roc_auc:.4f}")
    print(f"   - F1: {f1_est:.4f}")
    print(f"   - TPR: {rec_est:.4f}")
    print(f"   - FPR: {fpr_val:.4f}")

    # ==============================
    # 7) BUILD RETURN OBJECT
    # ==============================
    metrics_dict = {
        "metrics_results": {
            "val_anomaly_scores": val_anomaly_scores,  # [N, T]
            "val_labels": val_labels,  # [N, T]

            "val_roc_auc": roc_auc,
            "val_fpr": fpr_val,
            "val_tpr": rec_est,
            "val_best_thresh_youden": thresholds[ix_youden],
            "val_best_thresh_f1": None,  # deprecated
            "val_f1_score": f1_est,
            "val_precision": prec_est,
            "val_recall": rec_est,

            "val_normal_stats": normal_stats,
            "val_anomaly_stats": anomaly_stats,
            "val_normalization_factor": normalization_factor
        },
        "sampled_normal_indices": idx_main,
        "normal_error_source": normal_error_source
    }

    return metrics_dict, idx_main


    '''
    # Flatten per timestep
    scores = val_anomaly_scores.flatten().numpy()
    labels = val_labels.flatten().numpy()
    
    # Separiamo normali e anomalie
    scores_norm = scores[labels == 0]
    scores_anom = scores[labels == 1]
    
    # Statistiche rapide
    print("Normali:")
    print(f"  mean={scores_norm.mean():.4f}, std={scores_norm.std():.4f}, min={scores_norm.min():.4f}, max={scores_norm.max():.4f}")
    print("Anomalie:")
    print(f"  mean={scores_anom.mean():.4f}, std={scores_anom.std():.4f}, min={scores_anom.min():.4f}, max={scores_anom.max():.4f}")
    
    # Istogramma
    plt.figure(figsize=(8,5))
    plt.hist(scores_norm, bins=50, alpha=0.6, label="Normali")
    plt.hist(scores_anom, bins=50, alpha=0.6, label="Anomalie")
    plt.xlabel("Score di ricostruzione (L2 o SE)")
    plt.ylabel("Conteggio")
    plt.title("Distribuzione dei punteggi di ricostruzione per timestep")
    plt.legend()
    plt.show()
    
    # Boxplot
    plt.figure(figsize=(6,4))
    plt.boxplot([scores_norm, scores_anom], labels=["Normali", "Anomalie"])
    plt.ylabel("Score di ricostruzione")
    plt.title("Boxplot score normali vs anomalie")
    plt.show()
    '''

    return metrics_dict, idx_main


@torch.no_grad()
def test_anomaly_step_kp(
        model,
        metric_dataloader,
        device="cuda",
        external_normal_errors=None,
        compare_external_with_loader=False,
        num_thresh=10,
        epsilon=1e-5,
        desc="Testing anomalies",
        normal_anomalies_ratio=1,
        seed=123,
        shuffle=True,
):

    model.eval()
    anomaly_errors_list = []
    anomaly_masks_list = []
    normal_errors_list = [] if external_normal_errors is None or compare_external_with_loader else None

    normal_running_sum = 0.0
    normal_running_count = 0
    anom_running_sum = 0.0
    anom_running_count = 0

    normal_bar = tqdm(total=0, position=0, leave=True, desc="Normals")
    anom_bar   = tqdm(total=0, position=1, leave=True, desc="Anomalies")

    model_type, last_layer = infer_model_type(model)

    # ==============================
    # 1) PASS THROUGH DATALOADER
    # ==============================
    for batch in tqdm(metric_dataloader, desc=desc, position=2):
        x, target, mask, *_ = batch
        x, target = x.to(device), target.to(device)

        is_anom = mask.view(mask.size(0), -1).sum(dim=1) > 0
        is_norm = ~is_anom

        # Anomalies
        if is_anom.any():
            x_anom, target_anom, mask_anom = x[is_anom], target[is_anom], mask[is_anom]
            recon = model(x_anom)
            err_anom = torch.abs(recon - target_anom).cpu()

            if last_layer == "Conv2d":
                err_anom = torch.squeeze(err_anom)

            anomaly_errors_list.append(err_anom)
            anomaly_masks_list.append(mask_anom.cpu())

            # update progress
            batch_mean = err_anom.mean().item()
            anom_running_sum += err_anom.sum().item()
            anom_running_count += err_anom.numel()
            global_mean = anom_running_sum / anom_running_count

            anom_bar.set_postfix({"batch_mean": f"{batch_mean:.6f}",
                                  "global_mean": f"{global_mean:.6f}"})
            anom_bar.update(1)

        # Normals
        if is_norm.any() and normal_errors_list is not None:
            x_norm, target_norm = x[is_norm], target[is_norm]
            recon = model(x_norm)
            err_norm = torch.abs(recon - target_norm).cpu()

            if last_layer == "Conv2d":
                err_norm = torch.squeeze(err_norm)

            normal_errors_list.append(err_norm)

            batch_mean = err_norm.mean().item()
            normal_running_sum += err_norm.sum().item()
            normal_running_count += err_norm.numel()
            global_mean = normal_running_sum / normal_running_count

            normal_bar.set_postfix({"batch_mean": f"{batch_mean:.6f}",
                                    "global_mean": f"{global_mean:.6f}"})
            normal_bar.update(1)

    normal_bar.close()
    anom_bar.close()

    anomaly_errors = torch.cat(anomaly_errors_list, dim=0)
    anomaly_masks  = torch.cat(anomaly_masks_list, dim=0)

    # ============================================
    # 2) NORMAL ERROR SOURCE DETERMINATION
    # ============================================
    if external_normal_errors is not None:
        if last_layer == "Conv2d":
            external_normal_errors = torch.squeeze(external_normal_errors)

        normal_errors_all = external_normal_errors.cpu()
        normal_error_source = "external"
        normal_loader_all = torch.cat(normal_errors_list, dim=0) if compare_external_with_loader else None
    else:
        normal_errors_all = torch.cat(normal_errors_list, dim=0)
        normal_loader_all = None
        normal_error_source = "loader"

    # ============================================
    # 3) SAMPLE MAIN NORMAL SEQUENCES
    # ============================================
    N_anom = anomaly_errors.shape[0]
    N_norm_needed = int(N_anom * normal_anomalies_ratio)
    N_norm_needed = min(N_norm_needed, normal_errors_all.shape[0])

    np.random.seed(seed)
    idx_main = np.random.permutation(normal_errors_all.shape[0])[:N_norm_needed]
    normal_errors = normal_errors_all[idx_main]

    # PRINTS
    print(f"\n[INFO] Selected anomalous sequences: {N_anom}")
    print(f"[INFO] Selected normal sequences:    {normal_errors.shape[0]}")
    print(f"[INFO] Normal error source:          {normal_error_source}")

    # ============================================
    # 4) PRINT ERROR DISTRIBUTIONS
    # ============================================
    def print_error_stats(name, tensor):
        flat = tensor.flatten().numpy()
        print(f"\n[ERROR DISTRIBUTION] {name}")
        print(f"  mean:      {flat.mean():.6f}")
        print(f"  std:       {flat.std():.6f}")
        print(f"  min/max:   {flat.min():.6f} / {flat.max():.6f}")
        print(f"  q1,q5,q25: {np.quantile(flat, [0.01, 0.05, 0.25])}")
        print(f"  median:    {np.quantile(flat, 0.50):.6f}")
        print(f"  q75,q95:   {np.quantile(flat, [0.75, 0.95])}")

    print_error_stats("NORMAL sequences", normal_errors)
    print_error_stats("ANOMALOUS sequences", anomaly_errors)

    # ============================================
    # 5) NORMALIZATION
    # ============================================
    C = normal_errors.shape[1]

    normal_perm  = normal_errors.permute(0, 2, 1)
    anomaly_perm = anomaly_errors.permute(0, 2, 1)

    flat_norm = normal_perm.reshape(-1, C).float()
    normalization_factor = torch.quantile(flat_norm, 0.5, dim=0)
    norm = normalization_factor.view(1, 1, C) + epsilon

    normal_norm  = normal_perm  / norm
    anomaly_norm = anomaly_perm / norm

    masks_norm = torch.zeros((normal_norm.shape[0], anomaly_masks.shape[1], anomaly_masks.shape[2]),
                             dtype=torch.int)

    all_errors = torch.cat([normal_norm, anomaly_norm], dim=0)
    all_masks  = torch.cat([masks_norm, anomaly_masks], dim=0)

    # ============================================
    # 6) METRIC COMPUTATION
    # ============================================
    anomaly_scores = all_errors.mean(dim=1)
    seq_true   = (all_masks.view(all_masks.size(0), -1).sum(dim=1) > 0).numpy().astype(int)
    seq_scores = anomaly_scores.mean(dim=1).numpy()

    fpr, tpr, thresholds = roc_curve(seq_true, seq_scores)
    roc_auc = auc(fpr, tpr)

    candidate_thresholds = np.quantile(seq_scores, np.linspace(0, 1, num_thresh))
    f1s = [f1_score(seq_true, (seq_scores >= t).astype(int)) for t in candidate_thresholds]

    ix_f1 = np.argmax(f1s)
    ix_youden = np.argmax(tpr - fpr)

    # ============================================
    # FINAL SUMMARY PRINTS
    # ============================================
    print("\n[INFO] FINAL CHECK --- sequences used:")
    print(f"   - anomalous: {anomaly_errors.shape[0]}")
    print(f"   - normal:    {normal_errors.shape[0]}")

    # ============================================
    # 7) BUILD RETURN OBJECT
    # ============================================
    metrics_dict = {
        "metrics_results": {
            "val_anomaly_scores": anomaly_scores,
            "val_roc_auc": roc_auc,
            "val_fpr": fpr[ix_youden],
            "val_tpr": tpr[ix_youden],
            "val_best_thresh_youden": thresholds[ix_youden],
            "val_best_thresh_f1": candidate_thresholds[ix_f1],
            "val_f1_score": f1s[ix_f1],
            "val_normalization_factor": normalization_factor,
        },

        # new fields requested
        "sampled_normal_indices": idx_main,         # <=== SAVED INDICES
        "normal_error_source": normal_error_source  # <=== external / loader
    }

    return metrics_dict, idx_main



def get_module_type(name):
    """
    Classifica moduli in modo robusto per supportare:
    - Bottleneck nested (encoder.bottleneck.*) o non nested (bottleneck.*)
    - Conv1D e Conv2D
    - Latent layers nested o separati

    Priority order:
      1. latent (Linear to/from latent space)
      2. bottleneck_conv (Conv part of bottleneck)
      3. bottleneck (Generic bottleneck)
      4. encoder (Pure encoder conv layers)
      5. decoder (Pure decoder conv/deconv layers)
      6. other

    Returns:
        str: Module type classification
    """
    lname = name.lower()

    # =====================================================
    # Priority 1: LINEAR LATENT LAYERS
    # =====================================================
    latent_keywords = [
        "to_latent",  # encoder.bottleneck.to_latent (nested)
        "encoder_layer",  # encoder.encoder_layer (separate)
        "latent_to_flatten",  # decoder.*.latent_to_flatten
        "from_latent",
        "reshape"  # decoder.reshape (Linear from latent)
    ]

    if any(keyword in lname for keyword in latent_keywords):
        return "latent"

    # =====================================================
    # Priority 2: BOTTLENECK (Conv/BN/Activation part)
    # =====================================================
    if "bottleneck" in lname:
        conv_indicators = [
            "conv", "batch_norm", "bn", "activation", "act",
            "flatten", "unflatten"
        ]
        if any(indicator in lname for indicator in conv_indicators):
            return "bottleneck_conv"
        return "bottleneck"

    # =====================================================
    # Priority 2.5: FLATTEN separato (non in bottleneck)
    # =====================================================
    # encoder.flatten (Modello 2) → "other" per non freezarlo con "encoder"
    if "flatten" in lname and "encoder" in lname and "bottleneck" not in lname:
        return "other"

    # =====================================================
    # Priority 3: ENCODER (Pure convolutional encoder)
    # =====================================================
    if "encoder" in lname:
        encoder_indicators = [
            "enc_lay", "conv", "pool", "maxpool", "avgpool"
        ]
        if any(indicator in lname for indicator in encoder_indicators):
            return "encoder"
        return "encoder"

    # =====================================================
    # Priority 4: DECODER (Convolutional decoder)
    # =====================================================
    if "decoder" in lname:
        decoder_indicators = [
            "dec_lay", "deconv", "convtranspose", "upsample", "decoder_out"
        ]
        if any(indicator in lname for indicator in decoder_indicators):
            return "decoder"
        return "decoder"

    return "other"


def adjust_model_for_finetuning(
        fine_tuning_cfg,
        model,
        checkpoint,
        pre_feats,
        fine_feats,
        pre_seq_len,
        fine_seq_len,
        conv_type="conv_ae2d",
        device='cuda:0'
):
    """
    Adjust the model for fine-tuning when input dimensions change.

    Supports multiple fine-tuning strategies:
    - adaptive_layer: Learnable Conv2D adapters with spatial context
    - linear_proj: Linear/non-linear feature projection
    - latent_space: Reinitialize latent layers (random init)
    - soft_latent_space: Preserve latent dimensions for weight restoration (pre-trained)

    Args:
        fine_tuning_cfg: Fine-tuning configuration
        model: Model to adjust
        checkpoint: Pre-trained checkpoint
        pre_feats: Pre-training features
        fine_feats: Fine-tuning features
        pre_seq_len: Pre-training sequence length
        fine_seq_len: Fine-tuning sequence length
        conv_type: 'conv_ae1d' or 'conv_ae2d'
        device: Device to use

    Returns:
        Adjusted model
    """

    features_changed = pre_feats != fine_feats
    seq_changed = pre_seq_len != fine_seq_len

    print(f"🔍 Pre-training: feats={pre_feats}, seq={pre_seq_len}")
    print(f"🔍 Fine-tuning: feats={fine_feats}, seq={fine_seq_len}")

    # =========================================================================
    # HELPER FUNCTIONS
    # =========================================================================

    def update_latent(model, new_flattened, new_latent_dim):
        """
        Update latent space dimension by replacing linear layers.
        Creates new layers with Kaiming normal initialization.

        Args:
            model: Autoencoder model (already on device)
            new_flattened: New flattened size after conv layers
            new_latent_dim: New latent dimension

        Returns:
            model: Updated model (on same device)
        """

        def init_kaiming_linear(layer, mode='fan_in'):
            """Initialize linear layer with Kaiming normal."""
            nn.init.kaiming_normal_(layer.weight, mode=mode, nonlinearity='relu')
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

        encoder = model.encoder
        decoder = model.decoder

        # Auto-detect device
        try:
            device = next(model.parameters()).device
            print(f"\n🔧 Updating Latent Space:")
            print(f"   - Detected device:  {device}")
        except StopIteration:
            device = torch.device('cpu')
            print(f"\n⚠️  WARNING: Could not detect device, using CPU")

        print(f"   - New flatten size: {new_flattened}")
        print(f"   - New latent dim:   {new_latent_dim}")
        print(f"   - Compression On Feaures (not on Inputs!!!):      {new_flattened / new_latent_dim:.1f}:1")

        # Update encoder: to_latent
        encoder_updated = False
        for name, module in encoder.named_modules():
            if isinstance(module, nn.Linear) and "to_latent" in name.lower():
                old_in_features = module.in_features
                old_out_features = module.out_features

                new_layer = nn.Linear(new_flattened, new_latent_dim).to(device)
                init_kaiming_linear(new_layer)

                # Navigate to parent and replace
                parts = name.split('.')
                parent = encoder
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, parts[-1], new_layer)

                print(f"   ✅ Encoder: {name}")
                print(f"      Old: Linear({old_in_features} → {old_out_features})")
                print(f"      New: Linear({new_flattened} → {new_latent_dim})")
                encoder_updated = True

        if not encoder_updated:
            raise RuntimeError("❌ ERROR: No 'to_latent' layer found in encoder!")

        encoder.flattened_size = new_flattened
        encoder.latent_dim = new_latent_dim

        # Update decoder: latent_to_flatten
        decoder_updated = False
        for name, module in decoder.named_modules():
            if isinstance(module, nn.Linear) and "latent_to_flatten" in name.lower():
                old_in_features = module.in_features
                old_out_features = module.out_features

                new_layer = nn.Linear(new_latent_dim, new_flattened).to(device)
                init_kaiming_linear(new_layer)

                parts = name.split('.')
                parent = decoder
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, parts[-1], new_layer)

                print(f"   ✅ Decoder: {name}")
                print(f"      Old: Linear({old_in_features} → {old_out_features})")
                print(f"      New: Linear({new_latent_dim} → {new_flattened})")
                decoder_updated = True

        if not decoder_updated:
            raise RuntimeError("❌ ERROR: No 'latent_to_flatten' layer found in decoder!")

        decoder.flattened_size = new_flattened
        decoder.latent_dim = new_latent_dim

        print(f"\n✔  Latent space update complete!")
        return model

    def validate_freeze_policy(mode, freeze_layers, is_fallback=False):
        """
        Validate and inform about freeze policy compatibility with fine-tuning mode.

        This function only issues WARNINGS, never blocks execution.
        The actual freeze protection is handled by freeze_layers_with_logging.

        Key distinction:
        - latent_space: Random init latent → Will be PROTECTED automatically
        - soft_latent_space: Pre-trained latent → Can be frozen safely

        Args:
            mode: Fine-tuning mode
            freeze_layers: Freeze specification
            is_fallback: If True, indicates automatic fallback occurred
        """

        if freeze_layers is None or freeze_layers == '0':
            return  # No freezing, all modes compatible

        if isinstance(freeze_layers, int):
            freeze_layers = str(freeze_layers)

        # Convert to list
        if isinstance(freeze_layers, str):
            freeze_list = [freeze_layers]
        else:
            freeze_list = list(freeze_layers)

        warnings = []

        # =========================================================================
        # Inform about mode-specific behavior
        # =========================================================================

        if mode == 'latent_space':
            # latent_space = RANDOM INIT → Will be protected automatically
            if any(freeze in freeze_list for freeze in
                   ['encoder-bottleneck', 'encoder_bottleneck', 'bottleneck', 'all']):

                if is_fallback:
                    warnings.append(
                        f"⚠️  AUTOMATIC FALLBACK to 'latent_space' mode:\n"
                        f"   → Latent layers have RANDOM INIT → Will be PROTECTED automatically\n"
                        f"   → freeze_layers='{freeze_layers}' will freeze backbone only\n"
                        f"   → Latent layers will remain TRAINABLE (mode has precedence)\n"
                        f"   → This is the expected behavior"
                    )
                else:
                    warnings.append(
                        f"ℹ️  Mode 'latent_space' with freeze_layers='{freeze_layers}':\n"
                        f"   → Latent layers have RANDOM INIT → Will be PROTECTED automatically\n"
                        f"   → Requested freeze will apply to backbone layers only\n"
                        f"   → Latent layers will remain TRAINABLE (mode protection has precedence)\n"
                        f"   → This is the expected behavior - no action needed"
                    )

        elif mode == 'soft_latent_space':
            # soft_latent_space = PRE-TRAINED WEIGHTS → Can freeze bottleneck
            if any(freeze in freeze_list for freeze in ['encoder-bottleneck', 'encoder_bottleneck', 'bottleneck']):
                warnings.append(
                    f"✅ Mode 'soft_latent_space' with freeze bottleneck:\n"
                    f"   → Latent weights will be RESTORED from checkpoint\n"
                    f"   → Bottleneck will be FROZEN (pre-trained weights preserved)\n"
                    f"   → This is a VALID configuration for transfer learning"
                )

            if 'all' in freeze_list:
                warnings.append(
                    f"ℹ️  Mode 'soft_latent_space' with freeze_layers='all':\n"
                    f"   → This will freeze the ENTIRE model including latent layers\n"
                    f"   → Only adapters (if present) will be trainable\n"
                    f"   → Make sure this is intentional"
                )

        # Adapter-based modes: adapters always protected
        if mode in ['adaptive_layer', 'linear_proj', 'conv2d_adapter']:
            if 'all' in freeze_list:
                warnings.append(
                    f"ℹ️  Mode '{mode}' with freeze_layers='all':\n"
                    f"   → Adapters will remain TRAINABLE (always protected)\n"
                    f"   → Backbone will be frozen"
                )

        # Display warnings (informative only)
        if warnings:
            print("\n" + "=" * 80)
            print("📋 FREEZE POLICY INFORMATION:")
            print("=" * 80)
            for warning in warnings:
                print(f"\n{warning}")
            print("\n" + "=" * 80)
        else:
            print(f"✅ Freeze policy: '{freeze_layers}' with mode '{mode}'")

    def freeze_layers_with_logging(model, freeze_layers, fine_tuning_mode=None):
        """
        Freeze model layers selectively based on freeze_layers specification.
        Uses get_module_type() for robust classification.

        Special handling for latent layers:
        - latent_space mode: Latent layers are PROTECTED (random init, must train)
        - soft_latent_space mode: Latent layers CAN be frozen (pre-trained weights)

        Args:
            model: Model to freeze
            freeze_layers: Freeze specification
            fine_tuning_mode: Fine-tuning mode for special handling
        """

        print("\n" + "=" * 80)
        print("🧊 FREEZE-LAYERS: STARTING PROCEDURE")
        print("=" * 80)

        if freeze_layers is None:
            freeze_layers = []
        elif isinstance(freeze_layers, int):
            freeze_layers = [str(freeze_layers)]
        elif isinstance(freeze_layers, str):
            freeze_layers = [freeze_layers]

        # Special case: '0' means no freezing
        if "0" in freeze_layers:
            print("\nℹ️  Freeze layers = '0' → no freezing applied")
            freeze_layers = []

        # =====================================================================
        # Identify protected layers based on mode
        # =====================================================================
        always_protected = ["adapter", "adaptive"]
        mode_protected = []

        # CRITICAL: Only latent_space (random init) protects latent layers
        # soft_latent_space (pre-trained) allows freezing latent layers
        is_random_init_latent = (fine_tuning_mode == "latent_space")
        is_pretrained_latent = (fine_tuning_mode == "soft_latent_space")

        if is_random_init_latent:
            mode_protected = [
                "to_latent",
                "latent_to_flatten",
                "from_latent",
                "encoder_layer",
                "reshape"
            ]
            print(f"\n⚙️  Fine-tuning mode = 'latent_space'")
            print(f"   → Latent layers have RANDOM INIT → MUST be trainable")
            print(f"   → Protected keywords: {mode_protected}")

        elif is_pretrained_latent:
            print(f"\n⚙️  Fine-tuning mode = 'soft_latent_space'")
            print(f"   → Latent layers have PRE-TRAINED weights → CAN be frozen")
            print(f"   → No special protection for latent layers")

        def is_always_protected(name):
            return any(k in name.lower() for k in always_protected)

        def is_mode_protected(name):
            return any(k in name.lower() for k in mode_protected)

        def freeze_module(name, module):
            for p in module.parameters(recurse=True):
                p.requires_grad = False
            print(f"❄️  FREEZE → {name}")

        def keep_module(name, module, reason=""):
            print(f"🔥 KEEP   → {name} {reason}")
            for p in module.parameters(recurse=True):
                p.requires_grad = True

        # =====================================================================
        # Apply freeze logic
        # =====================================================================
        for name, module in model.named_modules():
            if not name:  # Skip root
                continue

            module_type = get_module_type(name)

            # Check 1: Always protected (adapter, adaptive)
            if is_always_protected(name):
                keep_module(name, module, "(always protected)")
                continue

            # Check 2: Mode protected (only for latent_space with random init)
            if is_mode_protected(name):
                keep_module(name, module, f"(protected by mode={fine_tuning_mode} - random init)")
                continue

            # Check 3: Freeze 'all'
            if "all" in freeze_layers:
                freeze_module(name, module)
                continue

            # Check 4: Freeze 'encoder-bottleneck'
            if "encoder-bottleneck" in freeze_layers or "encoder_bottleneck" in freeze_layers:
                if module_type in ["encoder", "bottleneck_conv", "bottleneck", "latent"]:
                    freeze_module(name, module)
                    continue
                else:
                    keep_module(name, module, "(not in encoder-bottleneck)")
                    continue

            # Check 5: Freeze 'encoder-decoder'
            if "encoder-decoder" in freeze_layers or "encoder_decoder" in freeze_layers:
                if module_type in ["encoder", "decoder"]:
                    freeze_module(name, module)
                    continue
                else:
                    keep_module(name, module, "(not in encoder-decoder)")
                    continue

            # Check 6: Freeze individual components
            if "encoder" in freeze_layers and module_type == "encoder":
                freeze_module(name, module)
                continue

            if "decoder" in freeze_layers and module_type == "decoder":
                freeze_module(name, module)
                continue

            if "bottleneck" in freeze_layers and module_type in ["bottleneck_conv", "bottleneck", "latent"]:
                freeze_module(name, module)
                continue

            # Check 7: Freeze numeric (first N layers)
            if any(item.isdigit() for item in freeze_layers):
                numeric_layers = [int(item) for item in freeze_layers if item.isdigit()]
                max_n = max(numeric_layers)
                freeze_count = getattr(model, "_freeze_count", 0)
                if freeze_count < max_n:
                    freeze_module(name, module)
                    model._freeze_count = freeze_count + 1
                    continue

            # Default: TRAINABLE
            keep_module(name, module, "(default: trainable)")

        print("\n✅ FREEZE COMPLETE\n" + "=" * 80 + "\n")

    # =========================================================================
    # FINE-TUNING MODE IMPLEMENTATIONS
    # =========================================================================

    def apply_adaptive_layer_mode(model, cfg, pre_feats, fine_feats, pre_seq_len, fine_seq_len, device):
        """
        Apply adaptive_layer fine-tuning mode.
        Creates learnable Conv2D adapters for input/output transformation.

        Args:
            model: Model to modify
            cfg: Configuration
            pre_feats, fine_feats: Feature dimensions
            pre_seq_len, fine_seq_len: Sequence lengths
            device: Device

        Returns:
            Modified model
        """

        print("⚙️  Fine-tuning mode: 'adaptive_layer' (learnable Conv2D adapters)")

        # Get adapter configuration
        adapter_config = cfg.opt.get('adapter', {})
        hidden_dim = adapter_config.get('hidden_dim', 32)
        use_residual = adapter_config.get('use_residual', True)
        num_layers = adapter_config.get('num_layers', 2)

        print(f"   Configuration:")
        print(f"      - Hidden dim: {hidden_dim}")
        print(f"      - Residual: {use_residual}")
        print(f"      - Num layers: {num_layers}")

        # Create INPUT adapter
        adapter_in = create_feature_adapter(
            pre_feats=pre_feats,
            pre_seq_len=pre_seq_len,
            fine_feats=fine_feats,
            fine_seq_len=fine_seq_len,
            cfg=cfg,
            adapter_type='input',
            hidden_dim=hidden_dim,
            use_residual=use_residual,
            num_layers=num_layers
        )
        model.input_adapter = adapter_in.to(device)

        # Create OUTPUT adapter
        adapter_out = create_feature_adapter(
            pre_feats=pre_feats,
            pre_seq_len=pre_seq_len,
            fine_feats=fine_feats,
            fine_seq_len=fine_seq_len,
            cfg=cfg,
            adapter_type='output',
            hidden_dim=hidden_dim,
            use_residual=use_residual,
            num_layers=num_layers
        )
        model.output_adapter = adapter_out.to(device)

        print(f"\n   ✅ Adaptive layer adapters created!")
        return model

    def apply_linear_proj_mode(model, cfg, pre_feats, fine_feats, pre_seq_len, fine_seq_len, device):
        """
        Apply linear_proj fine-tuning mode.
        Creates linear/non-linear projection adapters.

        Args:
            model: Model to modify
            cfg: Configuration
            pre_feats, fine_feats: Feature dimensions
            pre_seq_len, fine_seq_len: Sequence lengths
            device: Device

        Returns:
            Modified model
        """

        print("⚙️  Fine-tuning mode: 'linear_proj' (feature projection)")

        # Get adapter configuration
        adapter_config = cfg.opt.get('adapter', {})
        use_nonlinear = adapter_config.get('use_nonlinear', False)
        hidden_dim = adapter_config.get('hidden_dim', 32)

        print(f"   Configuration:")
        print(f"      - Type: {'Non-linear' if use_nonlinear else 'Linear'}")
        if use_nonlinear:
            print(f"      - Hidden dim: {hidden_dim}")

        # Create INPUT adapter
        adapter_in = create_projection_adapter(
            pre_feats=pre_feats,
            pre_seq_len=pre_seq_len,
            fine_feats=fine_feats,
            fine_seq_len=fine_seq_len,
            cfg=cfg,
            adapter_type='input',
            use_nonlinear=use_nonlinear,
            hidden_dim=hidden_dim
        )
        model.input_adapter = adapter_in.to(device)

        # Create OUTPUT adapter
        adapter_out = create_projection_adapter(
            pre_feats=pre_feats,
            pre_seq_len=pre_seq_len,
            fine_feats=fine_feats,
            fine_seq_len=fine_seq_len,
            cfg=cfg,
            adapter_type='output',
            use_nonlinear=use_nonlinear,
            hidden_dim=hidden_dim
        )
        model.output_adapter = adapter_out.to(device)

        print(f"\n   ✅ Projection adapters created!")
        return model

    def apply_latent_space_mode(model, checkpoint, pre_feats, fine_feats, pre_seq_len, fine_seq_len):
        """
        Apply latent_space fine-tuning mode.
        Reinitializes latent layers ONLY if dimensions don't match required size.

        Args:
            model: Model to modify
            checkpoint: Pre-trained checkpoint
            pre_feats, fine_feats: Feature dimensions
            pre_seq_len, fine_seq_len: Sequence lengths

        Returns:
            Modified model
        """

        print("⚙️  Fine-tuning mode: 'latent_space' (reinitialize latent if needed)")

        # Retrieve checkpoint dimensions
        old_latent_dim = checkpoint.get('cfg', {}).get('model', {}).get("latent_dim")
        old_flattened_feats = checkpoint.get('cfg', {}).get('model', {}).get("flattened_size")
        old_input_size = pre_feats * pre_seq_len

        # Retrieve current model dimensions
        current_flattened = getattr(model.encoder, "flattened_size")
        current_latent = getattr(model.encoder, "latent_dim")
        new_input_size = fine_feats * fine_seq_len

        compression_factor = getattr(model.encoder, "compression_factor")
        compression_type = getattr(model.encoder, "compression_type", 'on_features')

        print(f"📊 Dimension Analysis:")
        print(f"   - Checkpoint:     input={old_input_size}, flatten={old_flattened_feats}, latent={old_latent_dim}")
        print(f"   - Current model:  input={new_input_size}, flatten={current_flattened}, latent={current_latent}")
        print(f"   - Compression:    type={compression_type}, factor={compression_factor}")

        # =========================================================================
        # Calculate REQUIRED latent dimension based on compression strategy
        # =========================================================================

        if compression_type == 'on_features':
            # Latent based on flattened size
            required_latent = int(current_flattened // compression_factor)
            print(f"   - Required latent (from flatten): {required_latent}")

        elif compression_type == 'on_inputs':
            # Latent based on input size
            required_latent = int(new_input_size // compression_factor)
            print(f"   - Required latent (from input): {required_latent}")

        else:
            raise ValueError(f"Unknown compression type: {compression_type}")

        # =========================================================================
        # Determine if update is needed
        # =========================================================================

        # Check if current latent matches required latent
        latent_mismatch = (current_latent != required_latent)
        flatten_mismatch = (current_flattened != old_flattened_feats)

        print(f"\n🔍 Compatibility Check:")
        print(
            f"   - Flatten: checkpoint={old_flattened_feats}, current={current_flattened} → {'CHANGED' if flatten_mismatch else 'UNCHANGED'}")
        print(
            f"   - Latent:  current={current_latent}, required={required_latent} → {'MISMATCH' if latent_mismatch else 'MATCH'}")

        if latent_mismatch:
            # Latent dimensions don't match → MUST update
            print(f"\n🔧 Latent Update REQUIRED:")
            print(f"   → Current latent ({current_latent}) ≠ Required latent ({required_latent})")
            print(f"   → Reinitializing latent layers with correct dimensions...")

            # Validation
            min_latent = 16
            max_latent = new_input_size * 0.8
            if required_latent < min_latent:
                print(f"\n⚠️  WARNING: Latent dim {required_latent} is very small (< {min_latent})")
            if required_latent > max_latent:
                print(f"\n⚠️  WARNING: Latent dim {required_latent} > 80% of input")

            # Update latent space with required dimensions
            model = update_latent(model, current_flattened, required_latent)
            print(f"   ✅ Latent space updated: {current_latent} → {required_latent}")

        else:
            # Latent dimensions already match → NO update needed
            print(f"\n✅ Latent dimensions already CORRECT:")
            print(f"   → Current latent ({current_latent}) = Required latent ({required_latent})")
            print(f"   → No reinitialization needed!")

            if flatten_mismatch:
                print(f"\n   ℹ️  Note: Flatten changed ({old_flattened_feats} → {current_flattened})")
                print(f"      But latent is already correctly sized for new flatten")

        return model

    def apply_soft_latent_space_mode(model, checkpoint, pre_feats, fine_feats, pre_seq_len, fine_seq_len):
        """
        Apply soft_latent_space fine-tuning mode.

        Strategy:
        - If flatten unchanged: Keep original latent dimensions (weights restored later)
        - If flatten changed: Fall back to latent_space mode (reinitialize)

        CRITICAL: Returns the ACTUAL mode used (may differ from requested if fallback occurs)

        Args:
            model: Model to modify
            checkpoint: Pre-trained checkpoint
            pre_feats, fine_feats: Feature dimensions
            pre_seq_len, fine_seq_len: Sequence lengths

        Returns:
            tuple: (model, effective_mode)
                - model: Modified model
                - effective_mode: 'soft_latent_space' or 'latent_space' (if fallback)
        """

        print("⚙️  Fine-tuning mode: 'soft_latent_space' (preserve latent dimensions if possible)")

        # Retrieve dimensions
        old_flattened = checkpoint.get('cfg', {}).get('model', {}).get("flattened_size")
        old_latent = checkpoint.get('cfg', {}).get('model', {}).get("latent_dim")
        new_flattened = getattr(model.encoder, "flattened_size")

        print(f"📊 Dimension Check:")
        print(f"   - Pre-training: flatten={old_flattened}, latent={old_latent}")
        print(f"   - Fine-tuning:  flatten={new_flattened}")

        # Check if flatten dimension matches
        flatten_unchanged = (old_flattened == new_flattened)

        if flatten_unchanged:
            # TRUE SOFT MODE: Dimensions preserved
            print(f"\n✅ Flatten UNCHANGED → TRUE soft_latent_space mode")
            print(f"   Model latent layers will match checkpoint dimensions")
            print(f"   → Latent weights can be restored in load_pretrained_checkpoint")

            # Verify current model latent dimensions
            current_latent = getattr(model.encoder, "latent_dim")

            if current_latent != old_latent:
                print(f"\n⚠️  Current latent ({current_latent}) != Pre-trained latent ({old_latent})")
                print(f"   Adjusting model to match pre-trained dimensions...")

                # Update to match pre-trained dimensions
                model = update_latent(model, old_flattened, old_latent)
                print(f"   ✅ Model adjusted to match pre-trained latent dimensions")
            else:
                print(f"   ✓ Model latent dimensions already match: {current_latent}")

            print(f"\n   ✅ SOFT latent space mode: dimensions preserved for weight restoration")

            # Return TRUE soft mode
            return model, 'soft_latent_space'

        else:
            # FALLBACK TO LATENT_SPACE: Cannot preserve dimensions
            print(f"\n⚠️  Flatten CHANGED: {old_flattened} → {new_flattened}")
            print(f"   ❌ Cannot preserve latent dimensions")
            print(f"   🔄 FALLING BACK to 'latent_space' mode (random init)")
            print(f"\n" + "=" * 80)
            print(f"⚠️  CRITICAL: Effective mode changed to 'latent_space'")
            print(f"   → Latent layers will be RANDOMLY INITIALIZED")
            print(f"   → Freeze policy will be RE-VALIDATED for 'latent_space'")
            print(f"=" * 80)

            # Call latent_space mode
            model = apply_latent_space_mode(
                model, checkpoint, pre_feats, fine_feats, pre_seq_len, fine_seq_len
            )

            # Return EFFECTIVE mode (latent_space, not soft!)
            return model, 'latent_space'

    # =========================================================================
    # MAIN LOGIC
    # =========================================================================

    # Get freeze configuration
    freeze_layers = fine_tuning_cfg.opt.get("freeze_layers", None)
    if freeze_layers:
        print(f"🧊 Requested freeze layers: {freeze_layers}")

    # -------------------------------------------------------------------------
    # Handle Conv1D
    # -------------------------------------------------------------------------
    if conv_type.lower() == "conv_ae1d":
        if features_changed:
            print(f"🔧 Creating Conv1D adapters with refinement layers")

            # Get activation from config
            from config import activation_dict
            activation = activation_dict.get(
                fine_tuning_cfg.model.get('activation', 'ELU'),
                nn.ELU()
            )

            # Get adapter configuration
            adapter_config = fine_tuning_cfg.opt.get('adapter', {})
            hidden_dim = adapter_config.get('hidden_dim', 32)

            print(f"   Configuration:")
            print(f"      - Input:  {fine_feats} → {pre_feats}")
            print(f"      - Output: {pre_feats} → {fine_feats}")
            print(f"      - Hidden dim: {hidden_dim}")
            print(f"      - Activation: {fine_tuning_cfg.model.get('activation', 'ELU')}")

            # Input adapter: fine_feats → pre_feats
            model.input_adapter = nn.Sequential(
                # Initial projection
                nn.Conv1d(fine_feats, hidden_dim, kernel_size=1),
                nn.BatchNorm1d(hidden_dim),
                activation,
                # Refinement layer with context
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_dim),
                activation,
                # Final projection
                nn.Conv1d(hidden_dim, pre_feats, kernel_size=1)
            ).to(device)

            # Initialize weights
            for m in model.input_adapter:
                if isinstance(m, nn.Conv1d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

            # Output adapter: pre_feats → fine_feats
            model.output_adapter = nn.Sequential(
                # Initial projection
                nn.Conv1d(pre_feats, hidden_dim, kernel_size=1),
                nn.BatchNorm1d(hidden_dim),
                activation,
                # Refinement layer with context
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_dim),
                activation,
                # Final projection
                nn.Conv1d(hidden_dim, fine_feats, kernel_size=1)
            ).to(device)

            # Initialize weights
            for m in model.output_adapter:
                if isinstance(m, nn.Conv1d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

            # Count parameters
            input_params = sum(p.numel() for p in model.input_adapter.parameters())
            output_params = sum(p.numel() for p in model.output_adapter.parameters())

            print(f"\n   ✅ Conv1D adapters created!")
            print(f"      - Input adapter:  {input_params:,} parameters")
            print(f"      - Output adapter: {output_params:,} parameters")

        if freeze_layers:
            freeze_layers_with_logging(
                model, freeze_layers,
                fine_tuning_mode=fine_tuning_cfg.opt.get('fine_tuning_mode')
            )

        return model

    # -------------------------------------------------------------------------
    # Handle Conv2D
    # -------------------------------------------------------------------------
    # Handle Conv2D
    elif conv_type.lower() == "conv_ae2d":

        if features_changed or seq_changed:
            print("🔧 Adjusting Conv2D (features or sequence changed)")

            mode = fine_tuning_cfg.opt.get('fine_tuning_mode', None)

            if mode is None:
                raise ValueError("fine_tuning_mode must be specified when dimensions change!")

            # Inform about freeze policy (informative only, never blocks)
            validate_freeze_policy(mode, freeze_layers, is_fallback=False)

            # =================================================================
            # Apply appropriate mode and get EFFECTIVE mode
            # =================================================================
            effective_mode = mode  # Default: effective = requested

            if mode == "adaptive_layer":
                model = apply_adaptive_layer_mode(
                    model, fine_tuning_cfg, pre_feats, fine_feats,
                    pre_seq_len, fine_seq_len, device
                )

            elif mode == "linear_proj":
                model = apply_linear_proj_mode(
                    model, fine_tuning_cfg, pre_feats, fine_feats,
                    pre_seq_len, fine_seq_len, device
                )

            elif mode == "latent_space":
                model = apply_latent_space_mode(
                    model, checkpoint, pre_feats, fine_feats,
                    pre_seq_len, fine_seq_len
                )

            elif mode == "soft_latent_space":
                # CRITICAL: soft_latent_space may fall back to latent_space
                model, effective_mode = apply_soft_latent_space_mode(
                    model, checkpoint, pre_feats, fine_feats,
                    pre_seq_len, fine_seq_len
                )

                # ✅ If mode changed due to AUTOMATIC fallback, inform user
                if effective_mode != mode:
                    mode = effective_mode
                    print(f"\n" + "=" * 80)
                    print(f"🔄 AUTOMATIC MODE FALLBACK")
                    print(f"=" * 80)
                    print(f"   Requested:  soft_latent_space")
                    print(f"   Effective:  latent_space (due to flatten change)")
                    print(f"   Freeze:     {freeze_layers}")
                    print(f"\n⚙️  AUTOMATIC PROTECTION:")
                    print(f"   → Latent layers will be PROTECTED automatically")
                    print(f"   → Freeze will apply to backbone only")
                    print(f"   → Latent space remains TRAINABLE (mode has precedence)")
                    print(f"\n✅ Proceeding with automatic protection...")
                    print(f"=" * 80 + "\n")

                    # Inform about freeze policy with fallback context
                    validate_freeze_policy(effective_mode, freeze_layers, is_fallback=True)

            else:
                raise ValueError(f"Unsupported fine_tuning_mode: '{mode}'")

            # =================================================================
            # Apply freeze policy with EFFECTIVE mode
            # Protection is handled automatically by freeze_layers_with_logging
            # =================================================================
            if freeze_layers:
                print(f"\n🧊 Applying freeze policy with mode: '{effective_mode}'")
                freeze_layers_with_logging(
                    model, freeze_layers,
                    fine_tuning_mode=effective_mode
                )

        else:
            print("✅ Feature and sequence dimensions identical — no adapter needed")

            if freeze_layers:
                freeze_layers_with_logging(
                    model, freeze_layers,
                    fine_tuning_mode=fine_tuning_cfg.opt.get('fine_tuning_mode')
                )

        return model, mode

    else:
        raise ValueError(f"Unsupported conv_type '{conv_type}' (expected 'conv_ae1d' or 'conv_ae2d')")


# -------------------------------
# Helper: patch forward dynamically
# -------------------------------
def patch_forward_if_needed(model):
    """
    Patch forward to handle input/output adapters dynamically
    only if forward is not already compatible.
    """
    # Detect if forward already handles input_adapter
    src = model.forward.__code__.co_names
    if "input_adapter" in src or "output_adapter" in src:
        # Forward already compatible
        return

    original_forward = model.forward
    def new_forward(self, x):
        if hasattr(self, "input_adapter") and self.input_adapter is not None:
            x = self.input_adapter(x)
        out = original_forward(x)
        if hasattr(self, "output_adapter") and self.output_adapter is not None:
            out = self.output_adapter(out)
        return out

    model.forward = types.MethodType(new_forward, model)


def load_compatible_weights(model, checkpoint_state_dict):
    """
    Load matching weights from a checkpoint into a model safely,
    and report exactly which keys are skipped or mismatched.
    """
    model_dict = model.state_dict()
    compatible_dict = {}
    skipped_keys = []

    for k, v in checkpoint_state_dict.items():
        if k in model_dict:
            if isinstance(v, torch.Tensor) and v.shape == model_dict[k].shape:
                compatible_dict[k] = v.detach()
            else:
                print(f"⚠️ Skipping {k} (shape mismatch: checkpoint {v.shape} vs model {model_dict[k].shape})")
                skipped_keys.append(k)
        else:
            print(f"⚠️ Skipping {k} (key not found in model)")
            skipped_keys.append(k)

    # Load only compatible
    msg = model.load_state_dict(compatible_dict, strict=False)

    # Summary
    print(f"✅ Loaded {len(compatible_dict)} compatible tensors")
    print(f"⚠️ Skipped {len(skipped_keys)} incompatible/missing tensors:")
    for k in skipped_keys:
        print(f"   - {k}")

    if msg.missing_keys:
        print(f"ℹ️ Missing keys reported by PyTorch: {msg.missing_keys}")
    if msg.unexpected_keys:
        print(f"ℹ️ Unexpected keys reported by PyTorch: {msg.unexpected_keys}")

    return model


def load_pretrained_checkpoint(model, config, device):
    """
    Load pretrained weights into model for fine-tuning

    Args:
        model: PyTorch model instance
        config: Ray Tune config dictionary containing 'checkpoint_path' and 'fine_tuning'
        device: torch.device

    Returns:
        model: Model with loaded weights
        loaded: Boolean indicating if weights were loaded successfully
    """
    # Check if fine-tuning is enabled and checkpoint path is provided
    mode = None
    if not config.opt.get('fine_tuning', False):
        print("ℹ️ Training from scratch (no fine-tuning)")
        return model, False, mode

    if not config.opt.get('checkpoint_path', False):
        print("⚠️ WARNING: fine_tuning=True but no checkpoint_path provided!")
        raise Exception("Please provide a checkpoint path")

    checkpoint_path = config.opt.get('checkpoint_path')
    print(f"\n{'=' * 60}")
    print(f"{'=' * 60}")
    print(f"Checkpoint: {checkpoint_path}")

    try:
        # Capture initial state (for verification)
        initial_keys = list(model.state_dict().keys())[:3]
        initial_state = {k: model.state_dict()[k].clone() for k in initial_keys}
        initial_sample = list(initial_state.values())[0].flatten()[:5]
        print(f"Initial weights sample (random): {initial_sample}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        pre_trained_number_of_feats = len(checkpoint['cfg'].dataset.feats)
        fine_tuning_number_of_feats = config.dataset.n_features

        pre_training_seq_len = checkpoint['cfg'].dataset.seq_in_length
        fine_tuning_seq_len = config.dataset.seq_in_length

        if (
                pre_trained_number_of_feats != fine_tuning_number_of_feats
                or pre_training_seq_len != fine_tuning_seq_len
        ):
            print(f"⚠️ Dimension mismatch detected between pre-training and fine-tuning datasets!")
            strict = False

            model, mode = adjust_model_for_finetuning(
                config,
                model,
                checkpoint=checkpoint,
                pre_feats=pre_trained_number_of_feats,
                fine_feats=fine_tuning_number_of_feats,
                pre_seq_len=pre_training_seq_len,
                fine_seq_len=fine_tuning_seq_len,
                conv_type=config.model.name,
                device=device,
            )

        else:
            strict = True

        # Extract model state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            pretrained_epoch = checkpoint.get('epoch', '?')
            pretrained_loss = checkpoint.get('loss', '?')
            pretrained_value_loss = checkpoint.get('loss_value', '?')
            pretrained_params = checkpoint.get('parameters_number', None)

            print(f"Checkpoint info:")
            print(f"  - Epoch: {pretrained_epoch}")
            print(f"  - Loss/Metric: {pretrained_loss}")
            print(f"  - Loss/Metric value: {pretrained_value_loss}")
            if pretrained_params:
                print(f"  - Parameters: {pretrained_params:,}")
        else:
            state_dict = checkpoint
            pretrained_epoch = '?'
            pretrained_params = None

        # Load weights into model
        if not strict:
            print("⚠️ Loading with strict=False due to dimension mismatch")
            load_compatible_weights(model, state_dict)

        else:
            print("🔄 Loading with strict=True")
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
            if missing_keys:
                print(f"⚠️ Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"⚠️ Unexpected keys: {unexpected_keys}")

            # Verify weights changed
            loaded_state = model.state_dict()
            loaded_sample = list(loaded_state.values())[0].flatten()[:5]
            print(f"Loaded weights sample: {loaded_sample}")

            weights_changed = not torch.allclose(initial_sample.cpu(), loaded_sample.cpu(), rtol=1e-5)
            if weights_changed:
                print("✅ Weights successfully loaded and different from random initialization")
            else:
                print("⚠️ WARNING: Weights appear unchanged!")

            # Verify parameter count matches (if available)
            current_params = sum(p.numel() for p in model.parameters())
            if pretrained_params and pretrained_params != current_params:
                print(f"⚠️ WARNING: Parameter count mismatch!")
                print(f"   Pretrained: {pretrained_params:,}")
                print(f"   Current: {current_params:,}")

        print(f"✅ Loaded pretrained weights from epoch {pretrained_epoch}")

        print(f"{'=' * 60}\n")

        print('Model to fine tune', model)

        return model, True, mode

    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        traceback.print_exc()
        raise


def verify_pretrained_loading(model, checkpoint_path, device):
    """
    Detailed verification that checkpoint matches current model

    Args:
        model: PyTorch model
        checkpoint_path: Path to checkpoint
        device: torch.device

    Returns:
        dict: Verification results
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    current_state = model.state_dict()

    if 'model_state_dict' in checkpoint:
        pretrained_state = checkpoint['model_state_dict']
    else:
        pretrained_state = checkpoint

    results = {
        'all_keys_match': set(current_state.keys()) == set(pretrained_state.keys()),
        'num_params_current': sum(p.numel() for p in model.parameters()),
        'num_params_pretrained': checkpoint.get('parameters_number', None),
        'matched_layers': 0,
        'total_layers': 0,
    }

    # Check layer-by-layer matching
    for key in list(current_state.keys())[:10]:  # Check first 10 layers
        if key in pretrained_state:
            results['total_layers'] += 1
            if torch.allclose(current_state[key], pretrained_state[key], rtol=1e-5):
                results['matched_layers'] += 1

    results['match_percentage'] = (results['matched_layers'] / results['total_layers'] * 100) if results[
                                                                                                     'total_layers'] > 0 else 0

    return results


# ===========================================
# Learnable Adaptive Resizer 2D
# ===========================================
class AdaptiveLearnableResizer2D(nn.Module):
    def __init__(self, h_in, h_out, channels=1):
        super().__init__()
        self.h_out = h_out
        self.channels = channels
        # learnable 1x1 conv
        self.conv1x1 = nn.Conv2d(channels, channels, kernel_size=1)
        nn.init.kaiming_normal_(self.conv1x1.weight, nonlinearity='relu')

    def forward(self, x):
        # deterministic resize
        x = F.interpolate(x, size=(self.h_out, x.shape[-1]), mode='bilinear', align_corners=False)
        # learnable conv
        #x = self.conv1x1(x)
        return x
# models/utils/adapters.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import activation_dict


class FeatureAdapter2D(nn.Module):
    """
    Learnable adapter for changing feature dimensions in Conv2D time series.

    Automatically handles:
    - Feature dimension change (height)
    - Optional temporal dimension change (width)
    - Configurable activation function
    - Hidden channels for capacity
    - Spatial context awareness
    - Optional residual connection

    Args:
        h_in: Input height (number of features in fine-tuning)
        h_out: Output height (number of features in pre-training)
        w_in: Input width (sequence length in fine-tuning), optional
        w_out: Output width (sequence length in pre-training), optional
        channels: Number of input/output channels (default: 1)
        hidden_dim: Hidden dimension for capacity (default: 32)
        activation: Activation function (nn.Module)
        use_residual: Add residual connection (default: True)
        kernel_size_h: Kernel size along feature dimension (default: 3)
        num_layers: Number of transformation layers (default: 2)

    Example:
        >>> # INPUT adapter: 19×16 → 16×16
        >>> adapter_in = FeatureAdapter2D(
        ...     h_in=19, h_out=16, w_in=16, w_out=16,
        ...     channels=1, hidden_dim=32,
        ...     activation=nn.ELU()
        ... )
        >>> x = torch.randn(4, 1, 19, 16)
        >>> y = adapter_in(x)
        >>> print(y.shape)  # torch.Size([4, 1, 16, 16])
    """

    def __init__(
        self,
        h_in,
        h_out,
        w_in=None,
        w_out=None,
        channels=1,
        hidden_dim=32,
        activation=None,
        use_residual=True,
        kernel_size_h=3,
        num_layers=2
    ):
        super().__init__()

        self.h_in = h_in
        self.h_out = h_out
        self.w_in = w_in
        self.w_out = w_out
        self.channels = channels
        self.hidden_dim = hidden_dim
        self.use_residual = use_residual

        # Default activation if not provided
        if activation is None:
            activation = nn.ELU()
        self.activation = activation

        # Compute padding for 'same' convolution
        padding_h = kernel_size_h // 2

        # Build transformation layers
        layers = []

        # First layer: expand channels + spatial context
        layers.extend([
            nn.Conv2d(
                channels, hidden_dim,
                kernel_size=(kernel_size_h, 1),
                padding=(padding_h, 0),
                bias=True
            ),
            nn.BatchNorm2d(hidden_dim),
            activation
        ])

        # Middle layers: context refinement
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Conv2d(
                    hidden_dim, hidden_dim,
                    kernel_size=(3, 1),
                    padding=(1, 0),
                    bias=True
                ),
                nn.BatchNorm2d(hidden_dim),
                activation
            ])

        # Final layer: project back to channels
        layers.append(
            nn.Conv2d(hidden_dim, channels, kernel_size=1, bias=True)
        )

        self.transform = nn.Sequential(*layers)

        self._init_weights()

        # Print configuration
        total_params = sum(p.numel() for p in self.parameters())
        print(f"\n   🔧 FeatureAdapter2D initialized:")
        print(f"      Input:  [{channels}, {h_in}, {w_in if w_in else 'W'}]")
        print(f"      Output: [{channels}, {h_out}, {w_out if w_out else 'W'}]")
        print(f"      Hidden dim: {hidden_dim}")
        print(f"      Layers: {num_layers}")
        print(f"      Residual: {use_residual}")
        print(f"      Params: {total_params:,}")

    def _init_weights(self):
        """Initialize weights with Kaiming normal."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Forward pass with interpolation + learned transformation.

        Args:
            x: Input tensor [B, C, H_in, W_in]

        Returns:
            Output tensor [B, C, H_out, W_out]
        """
        # Determine target size
        if self.w_out is not None:
            target_size = (self.h_out, self.w_out)
        else:
            target_size = (self.h_out, x.shape[-1])

        # Interpolate to target dimensions
        x_interp = F.interpolate(
            x, size=target_size,
            mode='bilinear',
            align_corners=False
        )

        # Apply learnable transformation
        x_transformed = self.transform(x_interp)

        # Add residual connection if enabled
        if self.use_residual:
            output = x_interp + x_transformed
        else:
            output = x_transformed

        return output


# =============================================================================
# FACTORY FUNCTION - Easy Creation from Config
# =============================================================================

def create_feature_adapter(
    pre_feats,
    pre_seq_len,
    fine_feats,
    fine_seq_len,
    cfg,
    adapter_type='input',
    hidden_dim=32,
    use_residual=True,
    num_layers=2
):
    """
    Factory function to create adapter from config.

    Args:
        pre_feats: Number of features in pre-training model
        pre_seq_len: Sequence length in pre-training model
        fine_feats: Number of features in fine-tuning data
        fine_seq_len: Sequence length in fine-tuning data
        cfg: Config object with model.activation
        adapter_type: 'input' or 'output'
        hidden_dim: Hidden dimension (default: 32)
        use_residual: Use residual connection (default: True)
        num_layers: Number of transformation layers (default: 2)

    Returns:
        FeatureAdapter2D instance

    Example:
        >>> # INPUT adapter: fine → pre (19×16 → 16×16)
        >>> adapter_in = create_feature_adapter(
        ...     pre_feats=16, pre_seq_len=16,
        ...     fine_feats=19, fine_seq_len=16,
        ...     cfg=cfg, adapter_type='input'
        ... )

        >>> # OUTPUT adapter: pre → fine (16×16 → 19×16)
        >>> adapter_out = create_feature_adapter(
        ...     pre_feats=16, pre_seq_len=16,
        ...     fine_feats=19, fine_seq_len=16,
        ...     cfg=cfg, adapter_type='output'
        ... )
    """

    # Get activation from config
    activation_name = cfg.model.get('activation', 'ELU')
    activation = activation_dict.get(activation_name, nn.ELU())

    # Determine input/output dimensions based on adapter type
    if adapter_type == 'input':
        # INPUT: fine-tuning → pre-training
        h_in = fine_feats
        h_out = pre_feats
        w_in = fine_seq_len
        w_out = pre_seq_len
        print(f"\n🔧 Creating INPUT adapter:")
        print(f"   Fine-tuning [{fine_feats}×{fine_seq_len}] → Pre-training [{pre_feats}×{pre_seq_len}]")

    elif adapter_type == 'output':
        # OUTPUT: pre-training → fine-tuning
        h_in = pre_feats
        h_out = fine_feats
        w_in = pre_seq_len
        w_out = fine_seq_len
        print(f"\n🔧 Creating OUTPUT adapter:")
        print(f"   Pre-training [{pre_feats}×{pre_seq_len}] → Fine-tuning [{fine_feats}×{fine_seq_len}]")

    else:
        raise ValueError(f"adapter_type must be 'input' or 'output', got '{adapter_type}'")

    # Create adapter
    adapter = FeatureAdapter2D(
        h_in=h_in,
        h_out=h_out,
        w_in=w_in if w_in != w_out else None,  # Only specify if different
        w_out=w_out if w_in != w_out else None,
        channels=1,
        hidden_dim=hidden_dim,
        activation=activation,
        use_residual=use_residual,
        num_layers=num_layers
    )

    return adapter


# =============================================================================
# LIGHTWEIGHT VERSION - If you want minimal parameters
# =============================================================================

class LightweightFeatureAdapter2D(nn.Module):
    """
    Lightweight version with fewer parameters.
    Good for quick experiments or limited GPU memory.
    """

    def __init__(self, h_in, h_out, w_in=None, w_out=None,
                 channels=1, hidden_dim=16, activation=None):
        super().__init__()

        self.h_out = h_out
        self.w_out = w_out

        if activation is None:
            activation = nn.ELU()

        # Single transformation layer
        self.transform = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(hidden_dim),
            activation,
            nn.Conv2d(hidden_dim, channels, kernel_size=1)
        )

        self._init_weights()

        total_params = sum(p.numel() for p in self.parameters())
        print(f"\n   🔧 LightweightFeatureAdapter2D: {total_params:,} params")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        target_size = (self.h_out, self.w_out if self.w_out else x.shape[-1])
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        x = self.transform(x)
        return x


# =============================================================================
# ADAPTER 1: Feature Projection (Linear/Non-linear)
# =============================================================================

class FeatureProjectionAdapter(nn.Module):
    """
    Linear/Non-linear projection adapter for feature dimension transformation.

    All parameters have 'adapter' in their names for proper LR assignment.

    Args:
        n_in: Input number of features
        n_out: Output number of features
        use_nonlinear: If True, use non-linear projection (default: False)
        hidden_dim: Hidden dimension for non-linear (default: 32)
        activation: Activation function (default: GELU)
    """

    def __init__(
            self,
            n_in,
            n_out,
            use_nonlinear=False,
            hidden_dim=32,
            activation=None
    ):
        super().__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.use_nonlinear = use_nonlinear

        if activation is None:
            activation = nn.GELU()

        if not use_nonlinear:
            # ✅ Nome include "adapter"
            self.adapter_projection = nn.Linear(n_in, n_out)
        else:
            # ✅ Tutti i nomi includono "adapter"
            self.adapter_projection = nn.Sequential(
                nn.Linear(n_in, hidden_dim),
                nn.LayerNorm(hidden_dim),
                activation,
                nn.Linear(hidden_dim, n_out)
            )

        self._init_weights()

        total_params = sum(p.numel() for p in self.parameters())
        print(f"\n   🔧 FeatureProjectionAdapter initialized:")
        print(f"      Features: {n_in} → {n_out}")
        print(f"      Type: {'Non-linear (hidden={})'.format(hidden_dim) if use_nonlinear else 'Linear'}")
        print(f"      Parameters: {total_params:,}")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x: [B, C, H_in, W]
        Returns:
            [B, C, H_out, W]
        """
        B, C, H_in, W = x.shape
        assert H_in == self.n_in, f"Expected {self.n_in} input features, got {H_in}"

        # Reshape: treat each timestep independently
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(B * W, C * H_in)

        # Apply projection
        x_proj = self.adapter_projection(x)

        # Reshape back
        x_proj = x_proj.view(B, W, C, self.n_out)
        x_proj = x_proj.permute(0, 2, 3, 1).contiguous()

        return x_proj


# =============================================================================
# ADAPTER 2: Conv2D Feature Mapper (with context)
# =============================================================================

class Conv2DFeatureAdapter(nn.Module):
    """
    Conv2D-based feature adapter with spatial context.

    All parameters have 'adapter' in their names for proper LR assignment.

    Args:
        h_in: Input features
        h_out: Output features
        channels: Number of channels (default: 1)
        hidden_dim: Hidden channels (default: 64)
        kernel_size: Kernel size along features (default: 5)
        num_layers: Number of conv layers (default: 3)
        activation: Activation function
        use_residual: Use residual connection (default: True)
    """

    def __init__(
            self,
            h_in,
            h_out,
            channels=1,
            hidden_dim=64,
            kernel_size=5,
            num_layers=3,
            activation=None,
            use_residual=True
    ):
        super().__init__()

        self.h_in = h_in
        self.h_out = h_out
        self.channels = channels
        self.use_residual = use_residual

        if activation is None:
            activation = nn.GELU()

        kernel_h = min(kernel_size, h_in, h_out)
        padding_h = kernel_h // 2

        layers = []

        # First layer
        layers.extend([
            nn.Conv2d(
                channels, hidden_dim,
                kernel_size=(kernel_h, 1),
                padding=(padding_h, 0),
                bias=True
            ),
            nn.BatchNorm2d(hidden_dim),
            activation
        ])

        # Middle layers
        for _ in range(num_layers - 2):
            layers.extend([
                nn.Conv2d(
                    hidden_dim, hidden_dim,
                    kernel_size=(3, 1),
                    padding=(1, 0),
                    bias=True
                ),
                nn.BatchNorm2d(hidden_dim),
                activation
            ])

        # Final layer
        layers.append(
            nn.Conv2d(hidden_dim, channels, kernel_size=1, bias=True)
        )

        # ✅ Nome include "adapter"
        self.adapter_transform = nn.Sequential(*layers)

        self._init_weights()

        total_params = sum(p.numel() for p in self.parameters())
        print(f"\n   🔧 Conv2DFeatureAdapter initialized:")
        print(f"      Features: {h_in} → {h_out}")
        print(f"      Hidden channels: {hidden_dim}")
        print(f"      Kernel size: ({kernel_h}, 1)")
        print(f"      Num layers: {num_layers}")
        print(f"      Residual: {use_residual}")
        print(f"      Parameters: {total_params:,}")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x: [B, C, H_in, W]
        Returns:
            [B, C, H_out, W]
        """
        B, C, H, W = x.shape

        # Apply Conv2D transformation
        x_mapped = self.adapter_transform(x)

        # Resize if needed
        if x_mapped.shape[2] != self.h_out:
            x_mapped = F.interpolate(
                x_mapped,
                size=(self.h_out, W),
                mode='bilinear',
                align_corners=False
            )

        # Residual
        if self.use_residual:
            x_interp = F.interpolate(
                x,
                size=(self.h_out, W),
                mode='bilinear',
                align_corners=False
            )
            x_mapped = x_mapped + x_interp

        return x_mapped


# =============================================================================
# Factory Functions
# =============================================================================

def create_projection_adapter(
        pre_feats,
        pre_seq_len,
        fine_feats,
        fine_seq_len,
        cfg,
        adapter_type='input',
        use_nonlinear=False,
        hidden_dim=32
):
    """Create linear/non-linear projection adapter."""
    from config import activation_dict

    activation_name = cfg.model.get('activation', 'GELU')
    activation = activation_dict.get(activation_name, nn.GELU())

    if adapter_type == 'input':
        n_in, n_out = fine_feats, pre_feats
        print(f"\n🔧 Creating INPUT projection adapter:")
        print(f"   [{fine_feats}×{fine_seq_len}] → [{pre_feats}×{pre_seq_len}]")
    elif adapter_type == 'output':
        n_in, n_out = pre_feats, fine_feats
        print(f"\n🔧 Creating OUTPUT projection adapter:")
        print(f"   [{pre_feats}×{pre_seq_len}] → [{fine_feats}×{fine_seq_len}]")
    else:
        raise ValueError(f"adapter_type must be 'input' or 'output'")

    adapter = FeatureProjectionAdapter(
        n_in=n_in,
        n_out=n_out,
        use_nonlinear=use_nonlinear,
        hidden_dim=hidden_dim,
        activation=activation
    )

    return adapter


def create_conv2d_adapter(
        pre_feats,
        pre_seq_len,
        fine_feats,
        fine_seq_len,
        cfg,
        adapter_type='input',
        hidden_dim=64,
        kernel_size=5,
        num_layers=3,
        use_residual=True
):
    """Create Conv2D-based feature adapter."""
    from config import activation_dict

    activation_name = cfg.model.get('activation', 'GELU')
    activation = activation_dict.get(activation_name, nn.GELU())

    if adapter_type == 'input':
        h_in, h_out = fine_feats, pre_feats
        print(f"\n🔧 Creating INPUT Conv2D adapter:")
        print(f"   [{fine_feats}×{fine_seq_len}] → [{pre_feats}×{pre_seq_len}]")
    elif adapter_type == 'output':
        h_in, h_out = pre_feats, fine_feats
        print(f"\n🔧 Creating OUTPUT Conv2D adapter:")
        print(f"   [{pre_feats}×{pre_seq_len}] → [{fine_feats}×{fine_seq_len}]")
    else:
        raise ValueError(f"adapter_type must be 'input' or 'output'")

    adapter = Conv2DFeatureAdapter(
        h_in=h_in,
        h_out=h_out,
        channels=1,
        hidden_dim=hidden_dim,
        kernel_size=kernel_size,
        num_layers=num_layers,
        activation=activation,
        use_residual=use_residual
    )

    return adapter

# trainer/utils.py

def get_optimizazion_objects(cfg, model, opt_metric_dict):
    """
    Get optimizer, scheduler, criterion, and early stopping.

    Handles MULTIPLE learning rates:
    - lr: Main learning rate (backbone)
    - bottleneck_lr: Learning rate for bottleneck layers (0 = use main lr)
    - adapter_lr: Learning rate for adapters (0 = use main lr)

    Args:
        cfg: Configuration object
        model: Neural network model
        opt_metric_dict: Dictionary containing optimization metric info

    Returns:
        tuple: (optimizer, scheduler, criterion, early_stopping)
    """

    # ============================================================
    # HELPER: Robust numeric conversion
    # ============================================================
    def to_float(value, default=0.0, param_name="parameter"):
        """Convert parameter to float, handle string/None cases."""
        if value is None or value == "":
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            print(f"⚠️  Warning: Invalid {param_name} value '{value}', using {default}")
            return default

    # ============================================================
    # CONVERT ALL NUMERIC PARAMETERS
    # ============================================================
    lr = to_float(cfg.opt.lr, default=0.001, param_name="lr")

    bottleneck_lr = to_float(
        getattr(cfg.opt, "bottleneck_lr", 0),
        default=0.0,
        param_name="bottleneck_lr"
    )

    adapter_lr = to_float(
        getattr(cfg.opt, "adapter_lr", 0),
        default=0.0,
        param_name="adapter_lr"
    )

    lr_patience = int(to_float(
        cfg.opt.lr_patience,
        default=10,
        param_name="lr_patience"
    ))

    es_patience = int(to_float(
        cfg.opt.es_patience,
        default=15,
        param_name="es_patience"
    )) if hasattr(cfg.opt, 'es_patience') else None

    print("\n=================== OPTIMIZER SETUP ===================")
    print(f"Main LR:          {lr}")
    print(f"Bottleneck LR:    {bottleneck_lr}  (0 → use main LR)")
    print(f"Adapter LR:       {adapter_lr}  (0 → use main LR)")
    print(f"LR Patience:      {lr_patience}")
    print(f"ES Patience:      {es_patience}")
    print("--------------------------------------------------------")

    # ============================================================
    # IDENTIFY PARAMETER GROUPS
    # ============================================================
    adapter_params = []
    bottleneck_params = []
    main_params = []

    for name, param in model.named_parameters():
        # Check for adapter first (highest priority)
        if "adapter" in name.lower():
            adapter_params.append(param)
            print(f"   → Adapter param found: {name}")

        # Check for bottleneck
        elif "bottleneck" in name.lower():
            bottleneck_params.append(param)
            print(f"   → Bottleneck param found: {name}")

        # Everything else is main
        else:
            main_params.append(param)

    # ============================================================
    # STATISTICS
    # ============================================================
    print("\n📊 Parameter Groups:")
    print(f"   - Adapter params:    {len(adapter_params):4d}")
    print(f"   - Bottleneck params: {len(bottleneck_params):4d}")
    print(f"   - Main params:       {len(main_params):4d}")
    print(f"   - Total:             {len(adapter_params) + len(bottleneck_params) + len(main_params):4d}")

    # ============================================================
    # BUILD PARAMETER GROUPS FOR OPTIMIZER
    # ============================================================
    param_groups = []

    # ✅ MAIN PARAMS: Always add (may include adapter/bottleneck if their lr=0)
    if len(main_params) > 0:
        param_groups.append({"params": main_params, "lr": lr, "name": "main"})
        print(f"\n✓ Main params: LR = {lr}")

    # ✅ BOTTLENECK PARAMS
    if len(bottleneck_params) > 0:
        if bottleneck_lr > 0:
            # Separate LR for bottleneck
            param_groups.append({
                "params": bottleneck_params,
                "lr": bottleneck_lr,
                "name": "bottleneck"
            })
            print(f"✓ Bottleneck params: LR = {bottleneck_lr}  (separate)")
        else:
            # Use main LR (add to main group)
            if len(param_groups) > 0 and param_groups[0]["name"] == "main":
                param_groups[0]["params"].extend(bottleneck_params)
                print(f"✓ Bottleneck params: LR = {lr}  (same as main)")
            else:
                param_groups.append({
                    "params": bottleneck_params,
                    "lr": lr,
                    "name": "bottleneck_with_main"
                })
                print(f"✓ Bottleneck params: LR = {lr}  (same as main)")

    # ✅ ADAPTER PARAMS
    if len(adapter_params) > 0:
        if adapter_lr > 0:
            # Separate LR for adapter
            param_groups.append({
                "params": adapter_params,
                "lr": adapter_lr,
                "name": "adapter"
            })
            print(f"✓ Adapter params: LR = {adapter_lr}  🔥 (separate, HIGH for random init!)")
        else:
            # Use main LR (add to main group)
            if len(param_groups) > 0 and param_groups[0]["name"] == "main":
                param_groups[0]["params"].extend(adapter_params)
                print(f"✓ Adapter params: LR = {lr}  (same as main)")
            else:
                param_groups.append({
                    "params": adapter_params,
                    "lr": lr,
                    "name": "adapter_with_main"
                })
                print(f"✓ Adapter params: LR = {lr}  (same as main)")

    # ============================================================
    # WARNINGS & RECOMMENDATIONS
    # ============================================================
    if len(adapter_params) > 0:
        if adapter_lr == 0:
            print(f"\n⚠️  Adapter found but adapter_lr=0 → using MAIN lr={lr}")
            print(f"   ⚠️  Adapters are RANDOM INIT and may converge SLOWLY with low LR!")
            print(f"   💡 Recommendation: Set adapter_lr={lr * 100:.4f} (100x main lr)")
        elif adapter_lr < lr * 5:
            print(f"\n⚠️  adapter_lr={adapter_lr} is only {adapter_lr / lr:.1f}x main lr")
            print(f"   💡 Recommendation: adapter_lr should be 10-100x for random init adapters")

    if len(bottleneck_params) > 0 and bottleneck_lr == 0:
        print(f"\n✓ Bottleneck found, using main lr={lr} (bottleneck_lr=0)")

    # ============================================================
    # CREATE OPTIMIZER
    # ============================================================
    print(f"\n>>> Creating optimizer with {len(param_groups)} parameter group(s)")

    optimizer = torch.optim.Adam(param_groups)

    # ============================================================
    # SCHEDULER, CRITERION, EARLY STOPPING
    # ============================================================
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=opt_metric_dict["mode"],
        factor=0.8,
        patience=lr_patience,
        threshold=0.0001,
        threshold_mode='rel',
        cooldown=0,
        min_lr=9e-8,
        verbose=True
    )

    criterion = nn.MSELoss()
    min_delta = 1e-6 if opt_metric_dict["mode"] == "min" else 3e-3

    early_stopping = EarlyStopping(
        patience=es_patience,
        min_delta=min_delta,
        opt_metric_dict=opt_metric_dict
    ) if es_patience else None

    print("========================================================\n")
    return optimizer, scheduler, criterion, early_stopping