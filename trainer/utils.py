from pandas.core.computation.ops import isnumeric

from utils.general import resolve_paths, infer_model_type, reduce_anomaly_mask
from omegaconf import OmegaConf, ListConfig
import torch
import numpy as np
import math
from sklearn.metrics import roc_curve, auc, f1_score
from tqdm import tqdm
from typing import List, Optional
import warnings

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

def model_setup(config_file_name, config, root):

    cfg = OmegaConf.load(config_path + config_file_name)  # here use only vae conf file
    # Allow dynamic field insertion
    OmegaConf.set_struct(cfg, False)

    # Merge trial parameters from Ray Tune into OmegaConf config
    for k, v in config.items():
        OmegaConf.update(cfg, k, v, merge=True)
    # Construct dataset config and merge
    cfg = resolve_paths(cfg, root)

    return cfg

def update_input_output(cfg):
    """
    Infer input and output dimensions from the configuration.
    :param cfg: configuration object
    :return: input_dim, output_dim
    """
    if isinstance(cfg.dataset.feats, (list, ListConfig)):
        feats = cfg.dataset.feats
    elif cfg.dataset.feats == 'all':
        feats = all_feats_dict[cfg.dataset.name]
    else:
        feats = [cfg.dataset.feats]

    # Handle 'target'
    if isinstance(cfg.dataset.target, (list, ListConfig)):
        target = cfg.dataset.target
    elif isinstance(cfg.dataset.target, str):
        if cfg.dataset.target == 'all':
            target = all_feats_dict[cfg.dataset.name]
        else:
            target = [cfg.dataset.target]
    else:
        target = None

    # Merge model and opt into cfg
    cfg.dataset.feats = feats
    cfg.dataset.target = target

    return cfg, feats, target

def get_optimizazion_objects(cfg, model, opt_metric_dict):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.opt.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=0.8, patience=cfg.opt.lr_patience, threshold=0.0001,
        threshold_mode='rel', cooldown=0,min_lr=9e-8, verbose=True)
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience=cfg.opt.es_patience, min_delta=0.0000001,
                                   opt_metric_dict=opt_metric_dict) if cfg.opt.es_patience and opt_metric_dict else None

    return optimizer, scheduler, criterion, early_stopping

def get_opt_metric(cfg, metrics_loader=None):
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

def compute_errors(outputs: torch.Tensor, targets: torch.Tensor):
    return torch.abs(outputs.detach() - targets)

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
    model,
    dataloader: Optional[torch.utils.data.DataLoader],
    metric_loader: Optional[torch.utils.data.DataLoader],
    criterion,
    device,
    desc: str = "Validation",
    evaluate_metrics: bool = True,
    normal_anomalous_ratio: int = 1):

    model.eval()
    epoch_loss = 0
    all_errors = []

    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=desc, leave=False)
        for i, (inputs, targets, is_anomaly) in pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs).to(device)
            loss = criterion(outputs, targets)

            errors = compute_errors(outputs, targets)
            all_errors.append(errors.cpu())

            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

    all_errors = torch.cat(all_errors, dim=0)  # [N, C, L]
    #model_type, last_layer = infer_model_type(model)
    #channel_mean_errors, channel_std_errors = mean_std_per_channel(all_errors, model_type)

    # Core validation results
    results = {
        "val_loss": epoch_loss / len(dataloader),
        #"val_channel_mean_errors": channel_mean_errors,
        #"val_channel_std_errors": channel_std_errors,
    }

    # Optionally evaluate anomaly detection metrics

    if evaluate_metrics:
        test_results, indices = test_anomaly_step(
            model=model,
            metric_dataloader=metric_loader,
            device=device,
            external_normal_errors=all_errors,
            compare_external_with_loader=False,
            num_thresh=10,
            epsilon=1e-5,
            desc="Testing anomalies",
            normal_anomalies_ratio=normal_anomalous_ratio,
            seed=123,
            shuffle=True,
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

import pandas as pd

@torch.no_grad()
def test_anomaly_step(
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

    from sklearn.metrics import roc_curve, auc, f1_score

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



@torch.no_grad()
def test_anomaly_step_kp(
        model,
        metric_dataloader,
        device="cuda",
        external_normal_errors=None,  # only for alternative metrics
        num_thresh=10,
        epsilon=1e-5,
        desc="Testing anomalies",
        normal_anomalies_ratio=1,
        seed=123,
        shuffle=True,
):
    """
    Anomaly testing (Method B only: L1 feature-wise errors)

    Extra features:
    - Two tqdm bars:
        * Normal sequences: batch mean error + global mean error
        * Anomalous sequences: batch mean error + global mean error
    - Everything printed through tqdm bars (no raw prints)
    """

    model.eval()

    anomaly_errors_list = []
    anomaly_masks_list = []
    normal_errors_list = []

    # Running accumulators for tqdm bars
    normal_running_sum = 0.0
    normal_running_count = 0
    anom_running_sum = 0.0
    anom_running_count = 0

    # Two progress bars
    normal_bar = tqdm(total=0, position=0, leave=True, desc="Normals")
    anom_bar = tqdm(total=0, position=1, leave=True, desc="Anomalies")

    # ----------------------------------------
    # 1) Single pass over dataloader
    # ----------------------------------------
    for batch in tqdm(metric_dataloader, desc=desc, position=2):
        x, target, mask, *_ = batch
        x, target = x.to(device), target.to(device)

        is_anom = mask.view(mask.size(0), -1).sum(dim=1) > 0
        is_norm = ~is_anom

        # ---------- ANOMALOUS ----------
        if is_anom.any():
            x_anom, target_anom, mask_anom = x[is_anom], target[is_anom], mask[is_anom]
            recon = model(x_anom)
            err_anom = torch.abs(recon - target_anom).cpu()  # Method B: L1 error
            anomaly_errors_list.append(err_anom)
            anomaly_masks_list.append(mask_anom.cpu())

            # Batch mean
            batch_mean = err_anom.mean().item()

            # Update running global mean
            anom_running_sum += err_anom.sum().item()
            anom_running_count += err_anom.numel()
            global_mean = anom_running_sum / anom_running_count

            # Update tqdm bar
            anom_bar.set_postfix({
                "batch_mean": f"{batch_mean:.6f}",
                "global_mean": f"{global_mean:.6f}"
            })
            anom_bar.update(1)

        # ---------- NORMAL ----------
        if is_norm.any():
            x_norm, target_norm = x[is_norm], target[is_norm]
            recon = model(x_norm)
            err_norm = torch.abs(recon - target_norm).cpu()  # Method B
            normal_errors_list.append(err_norm)

            # Batch mean
            batch_mean = err_norm.mean().item()

            # Update running global mean
            normal_running_sum += err_norm.sum().item()
            normal_running_count += err_norm.numel()
            global_mean = normal_running_sum / normal_running_count

            # Update tqdm bar
            normal_bar.set_postfix({
                "batch_mean": f"{batch_mean:.6f}",
                "global_mean": f"{global_mean:.6f}"
            })
            normal_bar.update(1)

    # Close bars
    normal_bar.close()
    anom_bar.close()

    # ----------------------------------------
    # 2) Concatenate tensors
    # ----------------------------------------
    anomaly_errors = torch.cat(anomaly_errors_list, dim=0)
    anomaly_masks = torch.cat(anomaly_masks_list, dim=0)
    normal_errors_all = torch.cat(normal_errors_list, dim=0)

    # ----------------------------------------
    # 3) Sample normals according to ratio
    # ----------------------------------------
    N_anom = anomaly_errors.shape[0]
    N_norm_needed = int(N_anom * normal_anomalies_ratio)

    if N_norm_needed > normal_errors_all.shape[0]:
        print(f"WARNING: Not enough normal sequences. Using all available ({normal_errors_all.shape[0]})")
        N_norm_needed = normal_errors_all.shape[0]

    np.random.seed(seed)
    perm = np.random.permutation(normal_errors_all.shape[0])
    sampled_idx = perm[:N_norm_needed]
    normal_errors = normal_errors_all[sampled_idx]
    normal_indices = sampled_idx

    # ----------------------------------------
    # 4) Infer number of features
    # ----------------------------------------
    model_type, last_layer = infer_model_type(model)
    if last_layer == 'Conv2d':
        anomaly_errors = torch.squeeze(anomaly_errors)
        normal_errors = torch.squeeze(normal_errors)

    C = normal_errors.shape[1]

    # ----------------------------------------
    # 5) Feature-wise normalization (main metrics)
    # ----------------------------------------
    normal_perm = normal_errors.permute(0, 2, 1)
    anomaly_perm = anomaly_errors.permute(0, 2, 1)

    flat_norm = normal_perm.reshape(-1, C).float()
    normalization_factor = torch.quantile(flat_norm, 0.5, dim=0)
    norm = normalization_factor.view(1, 1, C) + epsilon

    normal_norm = normal_perm / norm
    anomaly_norm = anomaly_perm / norm

    all_errors = torch.cat([normal_norm, anomaly_norm], dim=0)
    masks_norm = torch.zeros((normal_norm.shape[0], anomaly_masks.shape[1], anomaly_masks.shape[2]), dtype=torch.int)
    all_masks = torch.cat([masks_norm, anomaly_masks], dim=0)

    # ----------------------------------------
    # 6) Compute per-sequence scores (main metrics)
    # ----------------------------------------
    anomaly_scores = all_errors.mean(dim=1)
    seq_true = (all_masks.view(all_masks.size(0), -1).sum(dim=1) > 0).numpy().astype(int)
    seq_scores = anomaly_scores.mean(dim=1).numpy()

    from sklearn.metrics import roc_curve, auc, f1_score

    fpr, tpr, thresholds = roc_curve(seq_true, seq_scores)
    roc_auc = auc(fpr, tpr)
    candidate_thresholds = np.quantile(seq_scores, np.linspace(0, 1, num_thresh))
    f1s = [f1_score(seq_true, (seq_scores >= t).astype(int)) for t in candidate_thresholds]

    ix_f1 = np.argmax(f1s)
    ix_youden = np.argmax(tpr - fpr)

    # ----------------------------------------
    # 7) Alternative metrics (optional)
    # ----------------------------------------
    metrics_alt = None
    if external_normal_errors is not None:
        model_type, last_layer = infer_model_type(model)
        if last_layer == 'Conv2d':
            external_normal_errors = torch.squeeze(external_normal_errors)

        N_ext = external_normal_errors.shape[0]
        flat_ext = external_normal_errors.permute(0, 2, 1).reshape(-1, C).float()
        norm_ext_factor = torch.quantile(flat_ext, 0.5, dim=0)
        norm_ext = norm_ext_factor.view(1, 1, C) + epsilon
        ext_norm = external_normal_errors.permute(0, 2, 1) / norm_ext

        all_errors_ext = torch.cat([ext_norm, anomaly_norm], dim=0)
        masks_ext = torch.cat([
            torch.zeros((N_ext, anomaly_masks.shape[1], anomaly_masks.shape[2]), dtype=torch.int),
            anomaly_masks
        ], dim=0)

        scores_ext = all_errors_ext.mean(dim=1)
        seq_scores_ext = scores_ext.mean(dim=1).numpy()
        seq_true_ext = (masks_ext.view(masks_ext.size(0), -1).sum(dim=1) > 0).numpy().astype(int)

        fpr_ext, tpr_ext, thresholds_ext = roc_curve(seq_true_ext, seq_scores_ext)
        roc_auc_ext = auc(fpr_ext, tpr_ext)
        candidate_thresholds_ext = np.quantile(seq_scores_ext, np.linspace(0, 1, num_thresh))
        f1s_ext = [f1_score(seq_true_ext, (seq_scores_ext >= t).astype(int)) for t in candidate_thresholds_ext]
        ix_f1_ext = np.argmax(f1s_ext)
        ix_youden_ext = np.argmax(tpr_ext - fpr_ext)

        print("\n--- Comparison Metrics ---")
        print(f"ROC AUC main: {roc_auc:.6f}")
        print(f"ROC AUC external: {roc_auc_ext:.6f}")
        print(f"F1 main: {f1s[ix_f1]:.6f}")
        print(f"F1 external: {f1s_ext[ix_f1_ext]:.6f}")

        metrics_alt = {
            "val_roc_auc_alt": roc_auc_ext,
            "val_fpr_alt": fpr_ext[ix_youden_ext],
            "val_tpr_alt": tpr_ext[ix_youden_ext],
            "val_best_thresh_youden_alt": thresholds_ext[ix_youden_ext],
            "val_best_thresh_f1_alt": candidate_thresholds_ext[ix_f1_ext],
            "val_f1_score_alt": f1s_ext[ix_f1_ext],
            "val_normalization_factor_alt": norm_ext_factor
        }

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
        }
    }

    if metrics_alt is not None:
        metrics_dict["metrics_results"].update(metrics_alt)

    return metrics_dict, normal_indices





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
    Adjust the model structure for fine-tuning when input dimensions change,
    and optionally freeze selected layers.
    """

    features_changed = pre_feats != fine_feats
    seq_changed = pre_seq_len != fine_seq_len

    print(f"🔍 Pre-training: feats={pre_feats}, seq={pre_seq_len}")
    print(f"🔍 Fine-tuning: feats={fine_feats}, seq={fine_seq_len}")

    # ============================================================
    # 0️⃣ FREEZE LAYERS (after modifications)
    # ============================================================

    def freeze_layers_with_logging(model, freeze_layers, fine_tuning_mode=None):
        """
        Congela layer con logging dettagliato.

        Args:
            model: PyTorch model
            freeze_layers: lista o singolo valore (es. 'encoder', '0', 'all', numeri)
            fine_tuning_mode: 'latent_space' o altro
        """

        print("\n" + "=" * 80)
        print("🧊 FREEZE-LAYERS: INIZIO PROCEDURA")
        print("=" * 80)

        # -----------------------
        # Normalizza freeze_layers
        # -----------------------
        if freeze_layers is None:
            freeze_layers = []
        elif isinstance(freeze_layers, int):
            freeze_layers = [str(freeze_layers)]
        elif isinstance(freeze_layers, str):
            freeze_layers = [freeze_layers]

        # "0" → nessun freeze
        if "0" in freeze_layers:
            print("\nℹ️ Freeze layers = '0' → nessun freeze applicato (solo protetti KEEP)")
            freeze_layers = []

        # -----------------------
        # Identifica componenti
        # -----------------------
        has_encoder = hasattr(model, "encoder")
        has_decoder = hasattr(model, "decoder")
        has_bottleneck = hasattr(model, "bottleneck")
        is_latent_strategy = fine_tuning_mode == "latent_space"

        print(f"🔍 Rilevamento componenti:")
        print(f"   - encoder:    {has_encoder}")
        print(f"   - decoder:    {has_decoder}")
        print(f"   - bottleneck: {has_bottleneck}")
        print(f"   - strategia latent_space: {is_latent_strategy}")

        # -----------------------
        # Definisci layer protetti (non congelabili)
        # -----------------------
        protected_keywords = ["adapter", "adaptive"]  # sempre protetti
        if is_latent_strategy:
            protected_keywords += ["latent", "bottleneck"]

        def is_protected(name):
            name = name.lower()
            return any(k in name for k in protected_keywords)

        # -----------------------
        # Funzioni di freeze / keep
        # -----------------------
        def freeze_module(name, module):
            print(f"❄️ FREEZE → {name}")
            for p in module.parameters(recurse=True):
                p.requires_grad = False

        def keep_module(name, module, reason="", protected=False):
            # Tutti i layer addestrabili → 🔥 KEEP
            icon = "🔥"
            suffix = " (protetto dal freeze)" if protected else ""
            print(f"   {icon} KEEP  → {name} {reason}{suffix}")
            for p in module.parameters(recurse=True):
                p.requires_grad = True

        # -----------------------
        # Caso 'all' → congela tutto tranne protetti
        # -----------------------
        if "all" in freeze_layers:
            print("\n🚨 Modalità FREEZE = 'all'")
            print("→ Congelo TUTTO tranne i layer protetti")
            for name, module in model.named_modules():
                if is_protected(name):
                    keep_module(name, module, "(protetto)", protected=True)
                else:
                    freeze_module(name, module)
            print("\n✅ FREEZE COMPLETATO (modalità 'all')")
            return

        # -----------------------
        # Modalità standard
        # -----------------------
        print("\n📌 Modalità FREEZE: standard")
        print(f"   Freeze layers richiesti: {freeze_layers if freeze_layers else 'nessuno'}\n")

        for name, module in model.named_modules():
            lname = name.lower()

            # Protetti → KEEP sempre
            if is_protected(name):
                keep_module(name, module, "(protetto)", protected=True)
                continue

            # Encoder
            if "encoder" in freeze_layers and "encoder" in lname:
                freeze_module(name, module)
                continue

            # Decoder
            if "decoder" in freeze_layers and "decoder" in lname:
                freeze_module(name, module)
                continue

            # Bottleneck / latent-space → freeze SOLO se non protetto
            if ("bottleneck" in lname or "latent" in lname):
                if not is_latent_strategy and "bottleneck" in freeze_layers:
                    freeze_module(name, module)
                else:
                    keep_module(name, module, "(protetto)" if is_latent_strategy else "")
                continue

            # Freeze numerico
            frozen = False
            for item in freeze_layers:
                if item.isdigit():
                    if f"_{item}" in lname or f"lay_{item}" in lname or f"layer{item}" in lname:
                        freeze_module(name, module)
                        frozen = True
                        break
            if not frozen:
                keep_module(name, module)

        print("\n✅ FREEZE COMPLETATO")
        print("=" * 80 + "\n")

    freeze_layers = fine_tuning_cfg.opt.get("freeze_layers", None)
    if freeze_layers:
        print(f"🧊 Requested freeze layers: {freeze_layers}")

    # ============================================================
    # 1️⃣ Conv1D case
    # ============================================================
    if conv_type.lower() == "conv_ae1d":
        warnings.warn("🧩 Detected Conv1D model — check adapter behavior", UserWarning)

        # -----------------------------------------
        # INPUT ADAPTER (C_fine → C_pre)
        # -----------------------------------------
        if features_changed:
            print(f"🔧 Adding Conv1D INPUT adapter: feats {fine_feats} → {pre_feats}")

            adapter_in = nn.Conv1d(
                in_channels=fine_feats,
                out_channels=pre_feats,
                kernel_size=1
            )
            nn.init.kaiming_normal_(adapter_in.weight)

            model.input_adapter = adapter_in.to(device)
            print("✅ Added Conv1D input adapter")

        # -----------------------------------------
        # OUTPUT ADAPTER (C_pre → C_fine)
        # -----------------------------------------
        if features_changed:
            print(f"🔧 Adding Conv1D OUTPUT adapter: feats {pre_feats} → {fine_feats}")

            adapter_out = nn.Conv1d(
                in_channels=pre_feats,
                out_channels=fine_feats,
                kernel_size=1
            )
            nn.init.kaiming_normal_(adapter_out.weight)

            model.output_adapter = adapter_out.to(device)
            print("✅ Added Conv1D output adapter")

        # freeze AFTER adapters
        if freeze_layers:
            freeze_layers_with_logging(model, freeze_layers, fine_tuning_mode=fine_tuning_cfg.opt.fine_tuning_mode)

        return model

    # ============================================================
    # 2️⃣ Conv2D case
    # ============================================================
    elif conv_type.lower() == "conv_ae2d":
        print("🧩 Detected Conv2D model")

        # If dimensions mismatch → input adapter and output adapter
        if features_changed or seq_changed:
            print("🔧 Adjusting Conv2D (features or sequence changed)")

            mode = fine_tuning_cfg.opt.get('fine_tuning_mode', None)
            if mode == "adaptive_layer":
                print("⚙️ Fine-tuning mode: 'adaptive_layer' (learnable resizer)")

                # -----------------------------------------
                # INPUT ADAPTER (Hf,Wf → Hp,Wp)
                # -----------------------------------------
                print(f"🔧 Adding Conv2D INPUT adapter: {fine_feats},{fine_seq_len} → {pre_feats},{pre_seq_len}")
                adapter_in = AdaptiveLearnableResizer2D(
                    h_in=fine_feats,
                    h_out=pre_feats,
                    channels=1
                )
                model.input_adapter = nn.Sequential(
                    adapter_in.to(device),
                    nn.BatchNorm2d(1),
                    nn.ReLU(inplace=True)
                )
                print(f"✅ Added learnable input adapter")

                # -----------------------------------------
                # OUTPUT ADAPTER (Hpre,Wpre → Hf,Wft)
                # -----------------------------------------
                print(
                    f"🔧 Adding Conv2D OUTPUT adapter (LEARNABLE): {pre_feats},{pre_seq_len} → {fine_feats},{fine_seq_len}")

                adapter_out = AdaptiveLearnableResizer2D(
                    h_in=pre_feats,
                    h_out=fine_feats,
                    channels=1
                )

                model.output_adapter = nn.Sequential(
                    adapter_out.to(device),
                    nn.BatchNorm2d(1),
                    nn.ReLU(inplace=True)
                )

                print("✅ Added learnable output adapter")

            else:
                # Diagnostic mode — prints architecture info
                print(f"ℹ️ Fine-tuning mode '{mode}' — structural diagnostics only")

                old_flattened = checkpoint.get('cfg', {}).get('model', {}).get("flattened_size", None)
                new_flattened = getattr(getattr(model, "encoder", None), "flattened_size", None)
                new_latent_dim = getattr(getattr(model, "encoder", None), "latent_dim", None)
                new_h = getattr(getattr(model, "encoder", None), "h_enc", None)
                new_w = getattr(getattr(model, "encoder", None), "w_enc", None)

                print(f"  - old_flattened: {old_flattened}")
                print(f"  - new_flattened: {new_flattened}")
                print(f"  - new_latent_dim: {new_latent_dim}")
                print(f"  - new_h: {new_h}, new_w: {new_w}")
                print(f"🧱 New model architecture: {model}")

            # freeze AFTER adding adapters
            if freeze_layers:
                freeze_layers_with_logging(model, freeze_layers, fine_tuning_mode=fine_tuning_cfg.opt.fine_tuning_mode)

            return model

        # ======================================================
        # No dimension mismatch → just freeze if requested
        # ======================================================
        else:
            print("✅ Feature and sequence dimensions identical — no adapter needed")

            if freeze_layers:
                freeze_layers_with_logging(model, freeze_layers, fine_tuning_mode=fine_tuning_cfg.opt.fine_tuning_mode)

            return model

    else:
        raise ValueError(f"Unsupported conv_type '{conv_type}' (expected 'conv_ae1d' or 'conv_ae2d')")




def load_compatible_weights(model, checkpoint_state_dict):
    """
    Load matching weights from a checkpoint into a model safely.
    Only parameters with identical shapes are loaded.
    """
    model_dict = model.state_dict()
    compatible_dict = {}

    for k, v in checkpoint_state_dict.items():
        if k in model_dict and isinstance(v, torch.Tensor) and v.shape == model_dict[k].shape:
            compatible_dict[k] = v.detach()
        else:
            print(f"⚠️ Skipping {k} (shape mismatch or missing key)")

    # Load only the compatible parameters
    msg = model.load_state_dict(compatible_dict, strict=False)

    # Summary
    print(f"✅ Loaded {len(compatible_dict)} compatible tensors")
    if msg.missing_keys:
        print(f"ℹ️ Missing keys in checkpoint: {len(msg.missing_keys)}")
    if msg.unexpected_keys:
        print(f"ℹ️ Unexpected keys in checkpoint: {len(msg.unexpected_keys)}")

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
    if not config.opt.get('fine_tuning', False):
        print("ℹ️ Training from scratch (no fine-tuning)")
        return model, False

    if not config.opt.get('checkpoint_path', False):
        print("⚠️ WARNING: fine_tuning=True but no checkpoint_path provided!")
        return model, False

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

            model = adjust_model_for_finetuning(
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
            print(f"  - Loss metric: {pretrained_loss}")
            print(f"  - Loss value: {pretrained_value_loss}")
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

        return model, True

    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        import traceback
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
    """
    Learnable resizer that adapts the spatial height (H_in → H_out)
    using Conv2D or ConvTranspose2D.
    Automatically computes padding / output_padding to reach the target size.
    """
    def __init__(self, h_in, h_out, kernel_size=3, channels=1):
        super().__init__()
        self.h_in = h_in
        self.h_out = h_out
        self.kernel_size = kernel_size
        self.channels = channels

        if h_in > h_out:
            # ✅ Downsample with Conv2d
            stride = math.ceil(h_in / h_out)
            padding = math.ceil(((h_out - 1) * stride - h_in + kernel_size) / 2)
            self.layer = nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=(kernel_size, 1),
                stride=(stride, 1),
                padding=(padding, 0)
            )
            self.mode = f"conv_down (stride={stride}, pad={padding})"

        elif h_in < h_out:
            # ✅ Upsample with ConvTranspose2d
            stride = math.floor(h_out / h_in)
            padding = kernel_size // 2

            H_calc = (h_in - 1) * stride - 2 * padding + kernel_size
            output_padding = max(0, h_out - H_calc)

            self.layer = nn.ConvTranspose2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=(kernel_size, 1),
                stride=(stride, 1),
                padding=(padding, 0),
                output_padding=(output_padding, 0)
            )
            self.mode = f"conv_transpose_up (stride={stride}, pad={padding}, out_pad={output_padding})"

        else:
            self.layer = nn.Identity()
            self.mode = "identity"

        # Init pesi (solo se layer learnable)
        if isinstance(self.layer, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(self.layer.weight, nonlinearity='relu')

    def forward(self, x):
        return self.layer(x)