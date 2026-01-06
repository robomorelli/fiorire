"""
Generic trainer for all model types.
"""

import torch
import numpy as np
import os
from ray import tune
import ray

from torch.utils.data import DataLoader
from utils.load_model import get_model
from utils.load_dataset import (
    load_and_preprocess_dataframe,
    create_datasets_from_splits,  # ← UPDATED
    create_dataloaders
)
from preprocessing.scaling import apply_scaler
from trainer.utils import (
    get_opt_metric, update_input_output, model_setup,
    train_one_epoch, validate_one_epoch, get_optimizazion_objects,
    load_pretrained_checkpoint, compute_indices_with_overlap
)
from config import opt_metric_dict_keys, root


class Trainer(tune.Trainable):
    """Generic trainer that works with all model types."""

    def setup(self, config):
        """Setup trial with DataFrame + indices loaded from disk."""
        import numpy as np
        import pickle

        # ✅ 1. Load config and shared_config
        self.cfg, shared_config = model_setup(
            config.get('opt.config_file_path'),
            config,
            root=root
        )

        # ✅ 2. Update input/output dimensions - CRITICAL!
        self.cfg, _, _ = update_input_output(self.cfg)
        self.effective_mode = None

        # ✅ 3. Get parameters for this trial
        overlap = self.cfg.dataset.get('perc_overlap', 0)
        val_overlap = self.cfg.dataset.get('val_overlap', 0)
        seq_len = shared_config['seq_len']
        self.cfg.dataset.feats, feature_columns = shared_config['feature_columns'], shared_config['feature_columns']
        self.cfg.dataset.n_features = len(feature_columns)

        print(f"\n📂 Loading data for trial (overlap={overlap})...")

        # ✅ 4. Load indices and scaler from DISK (NOT ray.get()!)
        self.indices_path = shared_config['indices_path']
        self.scaler_path = shared_config['scaler_path']
        print(f"   - Loading indices from: {self.indices_path}")
        indices_data = np.load(self.indices_path)
        train_base_indices = indices_data['train_indices']
        val_base_indices = indices_data['val_indices']

        print(f"   - Loading scaler from: {self.scaler_path}")
        with open(self.scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

        print(f"   ✓ Loaded from disk:")
        print(f"      - train_base_indices: {len(train_base_indices):,}")
        print(f"      - val_base_indices: {len(val_base_indices):,}")
        print(f"      - scaler: {self.scaler.__class__.__name__}")

        # ✅ 5. Compute final indices with overlap
        train_indices = compute_indices_with_overlap(train_base_indices, overlap, seq_len)
        val_indices = compute_indices_with_overlap(val_base_indices, val_overlap, seq_len)
        self.train_indices, self.val_indices = train_indices, val_indices

        print(f"   ✓ Indices with overlap:")
        print(f"      - train_indices: {len(train_indices):,}")
        print(f"      - val_indices: {len(val_indices):,} (overlap={val_overlap})")

        # ✅ 6. Load DataFrame
        df = load_and_preprocess_dataframe(self.cfg)

        # ✅ 7. Apply scaler (already fitted!)
        df_scaled = apply_scaler(df, self.scaler, feature_columns)

        print(f"   ✓ DataFrame loaded and scaled: {df_scaled.shape}")

        print(f"\n🔍 DEBUG - Indici:")
        print(f"   - train_indices: {len(train_indices)} elementi")
        print(f"   - val_indices: {len(val_indices)} elementi")
        print(f"   - train_indices[0:10]: {train_indices[:10]}")
        print(f"   - val_indices[0:10]: {val_indices[:10]}")

        # Verifica che NON siano uguali
        if np.array_equal(train_indices, val_indices):
            print("❌ ERROR: train_indices == val_indices!")
        else:
            print("✓ Indici diversi")

        # ✅ 8. Create datasets using create_datasets_from_splits
        # Create datasets
        datasets, samplers = create_datasets_from_splits(
            df_scaled=df_scaled,
            splits={
                'train': {
                    'indices': train_indices,
                    'shuffle': self.cfg.dataset.shuffle_train
                },
                'val': {
                    'indices': val_indices,
                    'shuffle': False
                }
            },
            cfg=self.cfg
        )

        train_dataset = datasets['train']
        val_dataset = datasets['val']
        train_sampler = samplers['train']
        val_sampler = samplers['val']

        train_dataset.indices = train_indices
        val_dataset.indices = val_indices

        # ✅ 9. Create dataloaders WITH samplers
        self.trainloader, self.valloader = create_dataloaders(self.cfg, train_dataset, val_dataset, train_sampler, val_sampler)

        print(f"   ✓ Dataloaders created")

        # ✅ 10. Load metric dataset with standardization verification
        self.metrics_loader = None  # Initialize to None
        metric_dataset_path, self.metric_dataset_path = shared_config.get('metric_loader_path'), shared_config.get('metric_loader_path')

        if self.cfg.opt.evaluate_metrics and metric_dataset_path and os.path.exists(metric_dataset_path):
            print(f"\n📊 Loading metric dataset...")
            print(f"   - Path: {metric_dataset_path}")

            saved_dict = torch.load(metric_dataset_path, map_location='cpu')
            metric_dataset = saved_dict['dataset']
            metadata = saved_dict.get('metadata', {})

            # ✅ Verify standardization
            is_standardized = metadata.get('is_standardized', True)

            if not is_standardized:
                raise ValueError(
                    f"❌ Metric dataset is NOT standardized!\n"
                    f"   Path: {metric_dataset_path}\n"
                    f"   Please regenerate with force_regenerate=True"
                )

            print(f"   ✓ Standardized: {is_standardized}")
            print(f"   ✓ Strategy: {metadata.get('strategy', 'N/A')}")
            print(f"   ✓ Sequences: {len(metric_dataset)}")

            # Add transform if missing
            if not hasattr(metric_dataset, 'transform') or metric_dataset.transform is None:
                from utils.load_dataset import get_transform
                metric_dataset.transform = get_transform(self.cfg)

            self.metrics_loader = DataLoader(
                metric_dataset,
                batch_size=self.cfg.opt.batch_size,
                shuffle=False
            )
            print(f"   ✓ Metric loader created")

        # ✅ 11. Store scaler params
        self.scaler_params = shared_config.get('scaler_params', {})
        self.scaler_pre_training_params = None

        # ✅ 12. Optimization metrics
        self.opt_metric_dict = get_opt_metric(self.cfg, self.metrics_loader)
        self.metric_key, self.mode, self.best_metric = (
            self.opt_metric_dict[k] for k in opt_metric_dict_keys
        )

        # ✅ 13. Device and model
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.cfg.resources.gpu_trial else "cpu")
        self.model = get_model(self.cfg).to(self.device)
        print("=== NAMED MODULES ===")

        self.cfg.model.parameter_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        try:
            self.latent_dim = self.model.latent_dim
        except:
            self.latent_dim = None
        self.parameters_number = self.cfg.model.parameter_count
        print('Parameters count', self.parameters_number)
        self.data_path = self.cfg.dataset.data_path

        # ✅ 14. Load pretrained if fine-tuning
        self.model, self.pretrained_loaded, self.effective_mode = load_pretrained_checkpoint(
            model=self.model, config=self.cfg, device=self.device
        )

        config['opt.fine_tuning_mode'] = self.effective_mode

        # ✅ 15. Optimizer, scheduler, criterion
        self.optimizer, self.scheduler, self.criterion, self.early_stopping = \
            get_optimizazion_objects(self.cfg, self.model, self.opt_metric_dict)

        # ✅ 16. Initialize tracking variables
        self.current_epoch = 0
        self.max_epochs = self.cfg.opt.get('epochs', 200)

        # Initialize best metrics
        if self.mode == 'min':
            self.best_val_loss = float('inf')
            self.best_fpr = float('inf')
        else:
            self.best_val_loss = float('inf')
            self.best_fpr = float('inf')

        self.best_f1_score = -float('inf')
        self.best_val_roc_auc = -float('inf')
        self.best_tpr = 0
        self.best_thresh_f1 = -float('inf')

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

        self.val_results, indices = validate_one_epoch(cfg=self.cfg,
            model=self.model, dataloader=self.valloader, metric_loader=self.metrics_loader,
            criterion=self.criterion, device=self.device, desc=f"Epoch {self.current_epoch} [Val]",
            evaluate_metrics=evaluate_metrics, normal_anomalous_ratio=int(self.cfg.opt.get('corruption_ratio', 1)),
            num_thresholds=self.cfg.opt.get("num_thresholds", 100), use_error=self.cfg.opt.get("use_error", "abs")
        )

        # Build result
        result = {
            "epoch": self.current_epoch, "train_loss": train_loss,
            "parameters_number": self.parameters_number, "data_path": self.data_path, "latent_dim": self.latent_dim,
            "effective_mode": self.effective_mode
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
        #result["best_val_thresh_f1"] = self.best_thresh_f1

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
            'data_path': self.data_path,
            'indices_path': self.indices_path,
            'scaler_path': self.scaler_path,
            'metric_dataset_path': self.metric_dataset_path,
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