from utils.general import resolve_paths, infer_model_type, reduce_anomaly_mask
from omegaconf import OmegaConf, ListConfig
import torch
import numpy as np
from sklearn.metrics import f1_score
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


def adjust_model_for_finetuning(model, checkpoint, pre_feats, fine_feats, pre_seq_len, fine_seq_len,
                                conv_type="conv_ae2D", device='cuda:0', halve_both=True,  halve_feats=True, halve_time=True):
    """
    Adatta la struttura del modello per il fine-tuning se cambiano le dimensioni di input.

    Args:
        model: modello PyTorch
        pre_feats: numero di feature nel pre-training
        fine_feats: numero di feature nel fine-tuning
        pre_seq_len: lunghezza sequenza nel pre-training
        fine_seq_len: lunghezza sequenza nel fine-tuning
        conv_type: "conv1d" o "conv2d"
    """
    features_changed = pre_feats != fine_feats
    seq_changed = pre_seq_len != fine_seq_len

    print(f"🔍 Pre-training: feats={pre_feats}, seq={pre_seq_len}")
    print(f"🔍 Fine-tuning: feats={fine_feats}, seq={fine_seq_len}")

    # -------------------------------
    # 1️⃣ Caso Conv1D
    # -------------------------------
    if conv_type.lower() == "conv_ae1d":
        warnings.warn("🧩 Detected Conv1D model — check adapter behavior", UserWarning)

        if features_changed:
            print(f"🔧 Adding Conv1D adapter: {pre_feats} → {fine_feats}")
            adapter = nn.Conv1d(
                in_channels=pre_feats,
                out_channels=fine_feats,
                kernel_size=1
            )
            nn.init.kaiming_normal_(adapter.weight)
            model.adapter_layer = adapter.to(device)
            print("✅ Latent layers updated for Conv1D")

        return model

    # -------------------------------
    # 2️⃣ Caso Conv2D
    # -------------------------------
    elif conv_type.lower() == "conv_ae2d":
        print(f"🧩 Detected Conv2D model")

        if features_changed or seq_changed:
            print(f"🔧 Adjusting latent space for Conv2D (features or sequence changed)")

            old_flattened = checkpoint['cfg'].model.get("flattened_size", None)
            new_flattened = model.encoder.flattened_size
            new_latent_dim = model.encoder.latent_dim
            new_h = model.encoder.h_enc
            new_w = model.encoder.w_enc

            print(f"  - old_flattened: {old_flattened}")
            print(f"  - new_flattened: {new_flattened}")
            print(f"  - new_latent_dim: {new_latent_dim}")
            print(f"  - new_h: {new_h}, new_w: {new_w}")
            print(f'New Model arcitecture: {model}')

        return model

    else:
        raise ValueError(f"Unsupported conv_type '{conv_type}' (expected 'conv_ae1D' or 'conv_ae2D')")

def load_compatible_weights(model, checkpoint_state_dict):
    model_dict = model.state_dict()
    compatible_dict = {}

    for k, v in checkpoint_state_dict.items():
        if k in model_dict and v.size() == model_dict[k].size():
            compatible_dict[k] = v
        else:
            print(f"⚠️ Skipping {k} (shape mismatch {v.size()} vs {model_dict.get(k, 'MISSING')})")

    model_dict.update(compatible_dict)
    model.load_state_dict(model_dict)

    print(f"✅ Loaded {len(compatible_dict)} / {len(model_dict)} layers from checkpoint")


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
                model,
                checkpoint=checkpoint,
                pre_feats=pre_trained_number_of_feats,
                fine_feats=fine_tuning_number_of_feats,
                pre_seq_len=pre_training_seq_len,
                fine_seq_len=fine_tuning_seq_len,
                conv_type=config.model.name,
                device=device,
                halve_both=checkpoint['cfg'].model.get('halve_both', False),
                halve_feats=checkpoint['cfg'].model.get('halve_features', True),
                halve_time=checkpoint['cfg'].model.get('halve_time', True)
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