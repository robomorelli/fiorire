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
        test_results, indices = test_anomaly_step_normalized(
            model=model,
            dataloader=metric_loader,
            device=device,
            all_val_norm_errors=all_errors,
            normalization_factor=None,
            normal_anomalies_ratio=normal_anomalous_ratio
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
def test_anomaly_step_normalized(
    model,
    dataloader,
    device,
    all_val_norm_errors=None,
    normalization_factor=None,
    num_thresh: int = 10,
    epsilon=1e-5,
    desc="Testing anomalies (feature weighting approach)",
    normal_anomalies_ratio: int = 1,
    seed=123,
    shuffle: bool = True
):
    """
    Computes anomaly detection scores using reconstruction error.
    Only sequences with at least one anomalous point are used for inference.
    Normal sequences are used only for normalization and F1 computation.

    Args:
        model: trained model
        dataloader: dataloader containing both normal and anomalous sequences
        device: torch device
        all_val_norm_errors: errors already computed on normal validation data [N_normal, C, L]
        normalization_factor: optional normalization factor per channel
        epsilon: small number to avoid div by zero
        normal_to_anomalies_ratio: ratio of normal to anomaly sequences for F1 computation
        seed: for reproducible sampling of normal sequences
        shuffle: whether to shuffle normal samples when sampling
    Returns:
        dict with anomaly metrics
    """

    model.eval()
    anomaly_errors = []
    anomaly_masks = []

    # -----------------------------------------------------------
    # 1) Filter sequences with at least one anomalous point
    # -----------------------------------------------------------
    for batch in tqdm(dataloader, desc=desc):
        x, target, mask, *rest = batch

        # Consider a sequence anomalous if it has at least one non-zero in mask
        is_anomaly_seq = mask.view(mask.size(0), -1).sum(dim=1) > 0
        if is_anomaly_seq.sum() == 0:
            continue  # skip sequences fully normal

        # Keep only anomalous sequences
        x_anom = x[is_anomaly_seq].to(device)
        target_anom = target[is_anomaly_seq].to(device)
        mask_anom = mask[is_anomaly_seq]

        recon = model(x_anom)
        error = torch.abs(recon - target_anom)  # [B_anom, C, L]
        anomaly_errors.append(error.cpu())
        anomaly_masks.append(mask_anom.cpu())

    if len(anomaly_errors) == 0:
        raise ValueError("No anomalous sequences found in the dataloader.")

    anomaly_errors = torch.cat(anomaly_errors, dim=0)  # [N_anom, 1, C, L] or [N_anom, C, L]
    anomaly_masks = torch.cat(anomaly_masks, dim=0)    # [N_anom, 1, L]

    model_type, last_layer = infer_model_type(model)

    # -----------------------------------------------------------
    # 2) Sample normal sequences to maintain ratio
    # -----------------------------------------------------------
    if all_val_norm_errors is None:
        raise ValueError("all_val_norm_errors must be provided for normalization.")

    N_anom = anomaly_errors.shape[0]
    N_normal_needed = N_anom * normal_anomalies_ratio

    if shuffle:
        g = torch.Generator()
        g.manual_seed(seed)
        # 972, 1475, 2312
        indices = torch.randperm(all_val_norm_errors.shape[0], generator=g)[:N_normal_needed]
    else:
        indices = torch.arange(min(N_normal_needed, all_val_norm_errors.shape[0]))

    normal_errors_sampled = all_val_norm_errors[indices]  # [N_normal_needed, C, L] or [N_normal_needed, C, L]

    if model_type == "cnn":
        if last_layer == 'Conv2d':
            C = anomaly_errors.shape[2]
            anomaly_errors = torch.squeeze(anomaly_errors)
            normal_errors_sampled = torch.squeeze(normal_errors_sampled)
        else:
            C = anomaly_errors.shape[1]
    elif model_type == "lstm":
        C = anomaly_errors.shape[2]
    else:
        raise ValueError("Unknown model type")

    # -----------------------------------------------------------
    # 3) Concatenate normal + anomaly sequences
    # -----------------------------------------------------------

    all_errors_combined = torch.cat([normal_errors_sampled, anomaly_errors], dim=0)
    all_masks_combined = torch.cat([
        torch.zeros((len(normal_errors_sampled), anomaly_masks.shape[1], anomaly_masks.shape[2]), dtype=torch.int),  # normal = 0
        anomaly_masks
    ], dim=0)

    # -----------------------------------------------------------
    # 4) Compute normalization factor if not provided
    # -----------------------------------------------------------
    C = all_errors_combined.shape[1]
    if normalization_factor is None:
        flat = normal_errors_sampled.permute(0, 2, 1).reshape(-1, C).float()  # [N_normal * L, C]
        normalization_factor = torch.quantile(flat, 0.5, dim=0)  # median per channel

    norm = normalization_factor.view(1, C, 1) + epsilon
    normalized_errors = all_errors_combined / norm

    # -----------------------------------------------------------
    # 5) Compute anomaly scores (mean across channels) per sequence
    # -----------------------------------------------------------
    anomaly_scores = normalized_errors.mean(dim=1)  # [N, L]

    # Aggregate mask per sequence: 1 if at least one point is anomalous
    seq_true = (all_masks_combined.view(all_masks_combined.size(0), -1).sum(dim=1) > 0).numpy().astype(int)
    seq_scores = anomaly_scores.mean(dim=1).numpy()  # mean over time steps per sequence

    # -----------------------------------------------------------
    # 6) Compute ROC, AUC, thresholds, F1 per sequence
    # -----------------------------------------------------------
    fpr, tpr, thresholds_full = roc_curve(seq_true, seq_scores)
    roc_auc = auc(fpr, tpr)

    # candidate thresholds
    NUM_THRESH = num_thresh
    candidate_thresholds = np.quantile(seq_scores, np.linspace(0, 1, NUM_THRESH))
    f1s = []
    for t in candidate_thresholds:
        preds = (seq_scores >= t).astype(int)
        f1s.append(f1_score(seq_true, preds))

    ix_f1 = np.argmax(f1s)
    best_thresh_f1 = candidate_thresholds[ix_f1]
    best_f1 = f1s[ix_f1]

    # Youden index
    J = tpr - fpr
    ix_youden = np.argmax(J)
    best_thresh_youden = thresholds_full[ix_youden]

    metrics_dict =  {
        "metrics_results": {
            "val_anomaly_scores": anomaly_scores,
            "val_roc_auc": roc_auc,
            "val_fpr": fpr,
            "val_tpr": tpr,
            "val_best_thresh_youden": best_thresh_youden,
            "val_best_thresh_f1": best_thresh_f1,
            "val_f1_score": best_f1,
            "val_normalization_factor": normalization_factor,
        }
    }



    return metrics_dict, indices


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
            if last_layer == 'Conv2d':
                C = all_errors.shape[2]
            else:
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
    Adjust the model structure for fine-tuning when input dimensions change.

    Args:
        fine_tuning_cfg: fine-tuning configuration object
        model: PyTorch model
        checkpoint: pretrained model checkpoint
        pre_feats: number of features (height) in pre-training
        fine_feats: number of features (height) in fine-tuning
        pre_seq_len: input sequence length during pre-training (width)
        fine_seq_len: input sequence length during fine-tuning (width)
        conv_type: "conv_ae1d" or "conv_ae2d"
        device: computation device
    """
    features_changed = pre_feats != fine_feats
    seq_changed = pre_seq_len != fine_seq_len

    print(f"🔍 Pre-training: feats={pre_feats}, seq={pre_seq_len}")
    print(f"🔍 Fine-tuning: feats={fine_feats}, seq={fine_seq_len}")

    # -------------------------------
    # 1️⃣ Conv1D case
    # -------------------------------
    if conv_type.lower() == "conv_ae1d":
        warnings.warn("🧩 Detected Conv1D model — check adapter behavior", UserWarning)

        if features_changed:
            print(f"🔧 Adding Conv1D adapter: {fine_feats} → {pre_feats}")
            # Map fine-tuning input features → pretrained feature dimension
            adapter = nn.Conv1d(
                in_channels=fine_feats,
                out_channels=pre_feats,
                kernel_size=1
            )
            if hasattr(adapter, "weight"):
                nn.init.kaiming_normal_(adapter.weight)
            model.adapter_layer = adapter.to(device)
            print("✅ Added Conv1D adapter layer")

        return model

    # -------------------------------
    # 2️⃣ Conv2D case
    # -------------------------------
    elif conv_type.lower() == "conv_ae2d":
        print("🧩 Detected Conv2D model")

        if features_changed or seq_changed:
            print("🔧 Adjusting latent space for Conv2D (features or sequence changed)")

            mode = fine_tuning_cfg.opt.get('fine_tuning_mode', None)

            if mode == "adaptive_layer":
                print("⚙️ Fine-tuning mode: 'adaptive_layer' (learnable resizer)")

                if features_changed:
                    # Add a learnable resizer to map feature height fine→pre
                    adapter = AdaptiveLearnableResizer2D(
                        h_in=fine_feats,
                        h_out=pre_feats,
                        channels=1
                    )
                    model.input_adapter = nn.Sequential(
                        adapter.to(device),
                        nn.BatchNorm2d(1),
                        nn.ReLU(inplace=True)
                    )
                    print(f"✅ Added learnable adapter {adapter.mode}: {fine_feats} → {pre_feats}")
                else:
                    print("✅ Sequence length changed but feature height identical — no adapter needed")

            else:
                print(f"ℹ️ Fine-tuning mode '{mode}' — running structural diagnostics only")

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

            return model

        else:
            print("✅ Feature and sequence dimensions identical — no adapter needed")
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
        pre_trained_number_of_feats = list(checkpoint['model_state_dict'].items())[0][1].shape[0]
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