from utils.general import resolve_paths, infer_model_type, reduce_anomaly_mask
from omegaconf import OmegaConf, ListConfig
import torch
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm
from typing import List, Optional
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
    n_std: Optional[List[float]] = None,
    anomaly_threshold: Optional[dict] = None):

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
    model_type, last_layer = infer_model_type(model)
    channel_mean_errors, channel_std_errors = mean_std_per_channel(all_errors, model_type)

    # Core validation results
    results = {
        "val_loss": epoch_loss / len(dataloader),
        "val_channel_mean_errors": channel_mean_errors,
        "val_channel_std_errors": channel_std_errors,
    }

    # Optionally evaluate anomaly detection metrics
    if evaluate_metrics:
        test_results = test_anomaly_step(
            model=model,
            dataloader=metric_loader,
            device=device,
            n_std=n_std,
            anomaly_threshold=anomaly_threshold,
        )

        # Flatten only necessary fields for logging
        if "metrics_results" in test_results:
            metrics = test_results["metrics_results"]
            results.update({
                "val_f1_score": metrics["best_f1_score"],
                "best_n_std": metrics["best_n_std"],
                "channel_means": metrics["channel_means"],
                "channel_stds": metrics["channel_stds"],
                # optionally more if needed for post-analysis
                # "channel_thresholds": metrics["channel_thresholds"]
            })

    return results

def test_anomaly_step(model, dataloader, device,
                      n_std: List[int]=None, anomaly_threshold=None,
                      desc="Testing for Anomalies"):
    model.eval()
    all_errors = []
    all_masks = []

    if n_std is None:
        n_std = [1]  # default value

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            x, target, mask = batch
            x = x.to(device)
            target = target.to(device)

            recon = model(x).to(device)
            error = torch.abs(recon - target)  # [B, C, L]
            all_errors.append(error.cpu())
            all_masks.append(mask.cpu())

    # Stack all errors and masks
    all_errors = torch.cat(all_errors, dim=0)  # [N, C, L]
    all_masks = torch.cat(all_masks, dim=0)    # [N, C, L]

    # Compute mean and std per channel
    if anomaly_threshold is not None:
        # Validate presence and shape
        assert "channel_means" in anomaly_threshold and "channel_stds" in anomaly_threshold, \
            "Missing 'channel_means' or 'channel_stds' in anomaly_threshold"

        channel_means = np.array(anomaly_threshold["channel_means"])  # shape: [C]
        channel_stds = np.array(anomaly_threshold["channel_stds"])  # shape: [C]

        assert channel_means.ndim == 1 and channel_stds.ndim == 1, "channel_means and channel_stds must be 1D arrays"
        assert channel_means.shape == channel_stds.shape, "channel_means and channel_stds must have the same shape"
        model_type, last_layer = infer_model_type(model)

        if model_type == "cnn":
            C = all_errors.shape[1]
        elif model_type == "lstm":
            C = all_errors.shape[2]
        else:
            raise ValueError("Unknown model type")

        assert channel_means.shape[0] == C, f"Expected {C} channels, but got {channel_means.shape[0]}"

    else:
        flat_errors = all_errors.view(all_errors.size(0), all_errors.size(1), -1)  # [N, C, L]
        mean_errors = flat_errors.mean(dim=2)  # [N, C]
        channel_means = mean_errors.mean(dim=0).numpy()  # [C]
        channel_stds = mean_errors.std(dim=0).numpy()  # [C]

    best_f1 = -1
    best_n_std = None
    best_pred_mask = None

    results_per_std = {}

    for curr_std in n_std:
        # Compute thresholds per channel
        thresholds = channel_means + curr_std * channel_stds  # [C]

        # Flatten for F1
        model_type, last_layer = infer_model_type(model)
        anomaly_mask_pred_reduced = reduce_anomaly_mask(all_errors, thresholds, model_type)

        # Flatten and evaluate
        y_pred = anomaly_mask_pred_reduced.view(-1).numpy()
        y_true = all_masks.view(-1).numpy()
        f1 = f1_score(y_true, y_pred)

        # Save results
        results_per_std[curr_std] = {
            "val_f1_score": f1,
            "thresholds": thresholds,
            "anomaly_mask_pred": anomaly_mask_pred_reduced,
        }

        # Track best
        if f1 > best_f1:
            best_f1 = f1
            best_n_std = curr_std
            best_pred_mask = anomaly_mask_pred_reduced

    return {"metrics_results":{
        "best_f1_score": best_f1,
        "best_n_std": best_n_std,
        "channel_means": channel_means,
        "channel_stds": channel_stds,
        "channel_thresholds": channel_means + best_n_std * channel_stds,
        "anomaly_mask_pred": best_pred_mask,
        "ground_truth_mask": all_masks,
        "results_per_std": results_per_std  # Optional: full history
    }}


