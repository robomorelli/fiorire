"""
Generic trainer for all model types.
"""

import torch
import numpy as np
import os
from ray import tune

from utils.load_model import get_model
from utils.load_dataset import (
    load_sequences_for_trial,
    get_transform,
    create_dataloaders,
    load_metric_loader_with_metadata  # ← Aggiungi questo
)
from dataset.sentinel import Dataset_seq
from trainer.utils import (
    get_opt_metric, update_input_output, model_setup,
    train_one_epoch, validate_one_epoch, get_optimizazion_objects,
    load_pretrained_checkpoint
)
from config import opt_metric_dict_keys
from preprocessing.scaling import serialize_scaler


class Trainer(tune.Trainable):
    """Generic trainer that works with all model types."""

    def setup(self, config):
        """Setup trial."""

        self.cfg = model_setup(config.get('opt.config_file_path'), config, None)
        self.cfg, _, _ = update_input_output(self.cfg)

        # Training state
        self.max_epochs = self.cfg.opt.epochs
        self.current_epoch = 0
        self.best_val_loss = float(np.inf)
        self.best_f1_score = -float(np.inf)
        self.best_val_roc_auc = -float(np.inf)
        self.best_fpr = float(np.inf)
        self.best_tpr = 0
        self.best_thresh_f1 = -float(np.inf)

        # Handle fine-tuning scaler params
        if config.get('opt.fine_tuning') and self.cfg.opt.get('checkpoint_path'):
            try:
                checkpoint = torch.load(self.cfg.opt.checkpoint_path)
                self.scaler_pre_training_params = checkpoint.get('scaler_params_pre_training')
            except:
                self.scaler_pre_training_params = None
        else:
            self.scaler_pre_training_params = None

        # Load sequences and create datasets
        shared_config = config['shared_config']
        overlap = config.get('dataset.perc_overlap', 0.0)

        train_sequences, val_sequences = load_sequences_for_trial(
            cfg=self.cfg,
            shared_config=shared_config,
            overlap=overlap
        )

        transform = get_transform(self.cfg)
        train_dataset = Dataset_seq(sequences=train_sequences, transform=transform)
        val_dataset = Dataset_seq(sequences=val_sequences, transform=transform)

        # Create dataloaders
        self.trainloader, self.valloader = create_dataloaders(train_dataset, val_dataset, self.cfg)

        # Store scaler
        self.scaler = shared_config['scaler']
        self.scaler_params = serialize_scaler(self.scaler)

        # ✅ Load metric loader and validate standardization
        metric_loader_path = shared_config.get('metric_loader_path')
        if self.cfg.opt.evaluate_metrics and metric_loader_path and os.path.exists(metric_loader_path):
            self.metrics_loader, self.metric_loader_metadata = load_metric_loader_with_metadata(
                metric_loader_path,
                verbose=True
            )

            # ✅ ASSERT: Data must be standardized (for now)
            is_standardized = self.metric_loader_metadata.get('is_standardized', True)

            if not is_standardized:
                error_msg = (
                        "\n" + "=" * 80 + "\n"
                                          "❌ ERROR: Metric loader contains NON-STANDARDIZED data\n"
                                          "=" * 80 + "\n"
                                                     f"File: {metric_loader_path}\n"
                                                     f"is_standardized: {is_standardized}\n"
                                                     "\n"
                                                     "The metric loader was saved with force_destandardization=True,\n"
                                                     "which means the data is in ORIGINAL SCALE (not standardized).\n"
                                                     "\n"
                                                     "Currently, the trainer requires pre-standardized metric loaders.\n"
                                                     "Automatic standardization will be implemented in the future.\n"
                                                     "\n"
                                                     "SOLUTION:\n"
                                                     "  1. Regenerate the metric loader with force_destandardization=False (default)\n"
                                                     "  2. Or use a different metric loader that is already standardized\n"
                                                     "\n"
                                                     "To regenerate:\n"
                                                     f"  - Remove file: {metric_loader_path}\n"
                                                     "  - Set in config: force_destandardization: false  (or omit it)\n"
                                                     "  - Re-run metric loader generation\n"
                        + "=" * 80
                )
                raise ValueError(error_msg)

            print(f"   ✅ Metric loader validation: data is standardized")

        else:
            self.metrics_loader = None
            self.metric_loader_metadata = None

        # Optimization metrics
        self.opt_metric_dict = get_opt_metric(self.cfg, self.metrics_loader)
        self.metric_key, self.mode, self.best_metric = (
            self.opt_metric_dict[k] for k in opt_metric_dict_keys
        )

        # Device and model
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.cfg.resources.gpu_trial else "cpu")
        self.model = get_model(self.cfg).to(self.device)
        self.cfg.model.parameter_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.parameters_number = self.cfg.model.parameter_count
        self.data_path = self.cfg.dataset.data_path

        # Load pretrained if fine-tuning
        self.model, self.pretrained_loaded = load_pretrained_checkpoint(
            model=self.model, config=self.cfg, device=self.device
        )

        # Optimizer, scheduler, criterion
        self.optimizer, self.scheduler, self.criterion, self.early_stopping = \
            get_optimizazion_objects(self.cfg, self.model, self.opt_metric_dict)

    def step(self):
        """Single training step - required by Ray Tune."""
        self.current_ip()
        self.result = self.train_step(checkpoint_dir=None)
        return self.result

    def train_step(self, checkpoint_dir=None):
        """Execute one training epoch."""
        self.current_epoch += 1

        # Train
        train_results = train_one_epoch(
            model=self.model, dataloader=self.trainloader, criterion=self.criterion,
            optimizer=self.optimizer, device=self.device, desc=f"Epoch {self.current_epoch} [Train]"
        )
        train_loss = train_results["train_loss"]
        print(f"Epoch {self.current_epoch} - Avg Train Loss: {train_loss:.6f}")

        # Validate
        evaluate_metrics = self.cfg.opt.evaluate_metrics and (
                self.metrics_loader is not None and
                (self.current_epoch % self.cfg.opt.detect_anomaly_epoch_freq == 0)
        )

        self.val_results, indices = validate_one_epoch(
            model=self.model, dataloader=self.valloader, metric_loader=self.metrics_loader,
            criterion=self.criterion, device=self.device, desc=f"Epoch {self.current_epoch} [Val]",
            evaluate_metrics=evaluate_metrics, normal_anomalous_ratio=self.cfg.opt.normal_anomalous_ratio,
            num_thresholds=self.cfg.opt.get("num_thresholds", 100), use_error=self.cfg.opt.get("use_error", "abs")
        )

        # Build result
        result = {
            "epoch": self.current_epoch, "train_loss": train_loss,
            "parameters_number": self.parameters_number, "data_path": self.data_path
        }

        current_val_loss = self.val_results["val_loss"]
        print(f"Epoch {self.current_epoch} - Avg Val Loss: {current_val_loss:.6f}")

        # Optional metrics
        current_f1 = self.val_results.get("val_f1_score", -float(np.inf))
        current_roc_auc = self.val_results.get("val_roc_auc", -float(np.inf))
        current_fpr = self.val_results.get("val_fpr", None)
        current_tpr = self.val_results.get("val_tpr", None)
        current_thresh_f1 = self.val_results.get("val_best_thresh_f1", None)

        result["val_f1_score"], result["val_roc_auc"], result[
            'val_loss'] = current_f1, current_roc_auc, current_val_loss
        result["val_fpr"], result["val_tpr"], result["val_thresh_f1"] = current_fpr, current_tpr, current_thresh_f1

        # Update bests
        try:
            if current_f1 > self.best_f1_score:
                self.best_f1_score = current_f1
                print(f"INFO: New best F1: {self.best_f1_score:.4f} at epoch {self.current_epoch}")
            if current_roc_auc > self.best_val_roc_auc:
                self.best_val_roc_auc = current_roc_auc
                print(f"INFO: New best ROC AUC: {self.best_val_roc_auc:.4f} at epoch {self.current_epoch}")
            if current_val_loss < self.best_val_loss:
                self.best_val_loss = current_val_loss
                print(f"INFO: New best Val Loss: {self.best_val_loss:.6f} at epoch {self.current_epoch}")
            if current_fpr and current_fpr < self.best_fpr:
                self.best_fpr = current_fpr
            if current_tpr and current_tpr > self.best_tpr:
                self.best_tpr = current_tpr
            if current_thresh_f1 and current_thresh_f1 > self.best_thresh_f1:
                self.best_thresh_f1 = current_thresh_f1
        except:
            print('Metrics not available')

        result["best_val_loss"] = self.best_val_loss
        result["best_val_roc_auc"] = self.best_val_roc_auc
        result["best_val_f1_score"] = self.best_f1_score
        result["best_val_fpr"] = self.best_fpr
        result["best_val_tpr"] = self.best_tpr
        result["best_val_thresh_f1"] = self.best_thresh_f1

        # Check improvement
        current_metric = result[self.metric_key]
        improved, self.best_metric = self.early_stopping(current_metric)
        self.scheduler.step(current_val_loss)

        result["should_checkpoint"] = improved
        result[f"best_{self.metric_key}"] = self.best_metric

        # Check stopping
        stop_training = False
        stop_reason = None

        if self.early_stopping.early_stop:
            print(f"INFO: Early stopping triggered at epoch {self.current_epoch}")
            stop_training = True
            stop_reason = "early_stopping"
        elif self.current_epoch >= self.max_epochs:
            print(f"INFO: Max epochs ({self.max_epochs}) reached at epoch {self.current_epoch}")
            stop_training = True
            stop_reason = "max_epochs"

        result["done"] = stop_training
        if stop_reason:
            result["stop_reason"] = stop_reason

        return result

    def test_step(self, checkpoint_dir=None):
        raise NotImplementedError("test_step method is not implemented yet.")

    def save_checkpoint(self, checkpoint_dir):
        checkpoint_path = f"{checkpoint_dir}/model.pt"
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.metric_key,
            'loss_value': self.result[f"best_{self.metric_key}"],
            'cfg': self.cfg,
            'scaler_params_pre_training': self.scaler_pre_training_params if self.scaler_pre_training_params else self.scaler_params,
            'scaler_params_fine_tuning': self.scaler_params if self.cfg.opt.get('fine_tuning', False) else None,
            'parameters_number': self.parameters_number,
            'param_conf': self.parameters_number,
            'metric_score': self.val_results if "metric_score" in self.val_results else None,
            "indices": self.val_results.get("indices", None),
        }, checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def current_ip(self):
        import socket
        hostname = socket.getfqdn(socket.gethostname())
        self._local_ip = socket.gethostbyname(hostname)
        return self._local_ip