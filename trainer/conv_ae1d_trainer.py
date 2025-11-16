from utils.load_model import get_model
from utils.load_dataset import get_train_val_dataloader, get_metric_dataloader
from trainer.utils import (get_opt_metric, update_input_output, model_setup, train_one_epoch, validate_one_epoch,
                           load_pretrained_checkpoint, get_optimizazion_objects)
from omegaconf import ListConfig
import numpy as np

from config import *
from models.utils.losses import *

class trainCONVAE1D(tune.Trainable):

    def setup(self, config):
        # Load and set up the configuration
        self.cfg = model_setup(conv_ae_1D_config_file, config, root)
        self.cfg, _, _ = update_input_output(self.cfg)  # convert feats and target to lists if they are not already (e.g "all" means all features of dataset)
        self.max_epochs = self.cfg.opt.epochs
        self.current_epoch = 0
        self.n_std = self.cfg.opt.n_std if isinstance(self.cfg.opt.n_std, (list, ListConfig)) else [self.cfg.opt.n_std]
        if  config.get('opt.fine_tuning') and 'scaler_params_pre_training' in torch.load(self.cfg.opt.get('checkpoint_path')).keys():
            self.scaler_pre_training_params = None if not(self.cfg.opt.get('fine_tuning', False)) else torch.load(self.cfg.opt.get('checkpoint_path'))['scaler_params_pre_training']
        else:
            self.scaler_pre_training_params = None

        # Load data
        # try to separate the anomalous sequences (using "is_anomaly_column") from the main dataset anyway. If they are not present, the dataset (metric loader) will be empty
        self.trainloader, self.valloader, self.metrics_loader, self.scaler, self.scaler_params = get_train_val_dataloader(
            self.cfg, filter_anomalies=True)
        # If the anomalous sequences are not present in the main dataset, the metrics_loader will be None. Try to load it from the path specified in the config file
        self.metrics_loader, _, _ = get_metric_dataloader(self.cfg, self.metrics_loader,
                          data_path=self.cfg.opt.metrics_dataset_path,
                          scale=True,
                          scaler=self.scaler) if self.cfg.opt.evaluate_metrics else None

        self.opt_metric_dict = get_opt_metric(self.cfg, self.metrics_loader)
        self.metric_key, self.mode, self.best_metric = (
            self.opt_metric_dict[k] for k in opt_metric_dict_keys
        )

        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.cfg.resources.gpu_trial else "cpu")

        # Build model
        self.model = get_model(self.cfg).to(self.device)
        self.cfg.model.parameter_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.parameters_number = self.cfg.model.parameter_count

        # ==========================================
        # LOAD PRETRAINED WEIGHTS IF FINE-TUNING
        # ==========================================
        self.model, self.pretrained_loaded = load_pretrained_checkpoint(
            model=self.model,
            config=self.cfg,
            device=self.device
        )

        # Optimizer and scheduler
        self.optimizer, self.scheduler, self.criterion, self.early_stopping = get_optimizazion_objects(self.cfg,
                                                                                                       self.model,
                                                                                                       self.opt_metric_dict)

    def step(self):
        self.current_ip()
        self.result = self.train_step(checkpoint_dir=None)
        return self.result

    def train_step(self, checkpoint_dir=None):

        self.current_epoch += 1

        train_results = train_one_epoch(
            model=self.model,
            dataloader=self.trainloader,
            criterion=self.criterion,
            optimizer=self.optimizer,
            device=self.device,
            desc=f"Epoch {self.current_epoch} [Train]",
        )
        train_loss = train_results["train_loss"]
        print(f"Epoch {self.current_epoch} - Avg Train Loss: {train_loss:.6f}")

        evaluate_metrics = self.cfg.opt.evaluate_metrics and (
                self.metrics_loader is not None and
                (self.current_epoch % self.cfg.opt.detect_anomaly_epoch_freq == 0)
        )

        self.val_results = validate_one_epoch(
            model=self.model,
            dataloader=self.valloader,
            metric_loader=self.metrics_loader,
            criterion=self.criterion,
            device=self.device,
            desc=f"Epoch {self.current_epoch} [Val]",
            evaluate_metrics=evaluate_metrics,
            n_std=self.n_std,
            anomaly_threshold=train_results.get('anomaly_threshold', None)
        )
        val_loss = self.val_results["val_loss"]
        print(f"Epoch {self.current_epoch} - Avg Val Loss: {val_loss:.6f}")

        self.scheduler.step(val_loss)
        # Combine loggable metrics
        result = {
            "epoch": self.current_epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            'val_f1_score': self.val_results.get('val_f1_score', -float(np.inf)),  # 👈 Always included
            "parameters_number": self.parameters_number,
        }

        if self.metric_key in self.val_results:
            result[f"{self.metric_key}"] = self.val_results.get(self.metric_key, -float(np.inf))  # 👈 Always included

        # Track best model
        # example of self.cfg.opt_metric: {'val_loss': 'min'}
        # Step 2: Save model only if current metric is better
        # 🔑 Unified improvement + early stopping
        current_metric = result[self.metric_key]
        improved, best_metric = self.early_stopping(current_metric)

        result["should_checkpoint"] = improved
        result[f"best_{self.metric_key}"] = best_metric
        result["best_n_std"] = self.val_results.get("best_n_std", 0.0)

        # Check early stopping conditions
        stop_training = False
        stop_reason = None

        # Check if early stopping triggered
        if self.early_stopping.early_stop:
            print(f"INFO: Early stopping triggered at epoch {self.current_epoch}.")
            stop_training = True
            stop_reason = "early_stopping"

        # Check if max epochs reached
        elif self.current_epoch >= self.max_epochs:
            print(f"INFO: Maximum epochs ({self.max_epochs}) reached at epoch {self.current_epoch}.")
            stop_training = True
            stop_reason = "max_epochs"

        # Add stopping information to result
        result["done"] = stop_training
        if stop_reason:
            result["stop_reason"] = stop_reason

        return result

    def test_step(self, checkpoint_dir=None):
        raise NotImplementedError("test_lstm method is not implemented yet.")

    def save_checkpoint(self, checkpoint_dir):
        print("this is the checkpoint dir {}".format(checkpoint_dir))
        torch.save({
            'epoch': self.current_epoch, 'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(), 'loss': self.metric_key,
            'loss_value': self.result[f"best_{self.metric_key}"] ,
            'cfg': self.cfg, 'scaler_params_pre_training': self.scaler_pre_training_params if self.scaler_pre_training_params else self.scaler_params,
            'scaler_params_fine_tuning': self.scaler_params if self.cfg.opt.get('fine_tuning', False) else None,
            'parameters_number': self.parameters_number, 'param_conf': self.parameters_number,
            'metric_score': self.val_results if "metric_score" in self.val_results else None,
        }, f"{checkpoint_dir}/model.pt")
        return os.path.join(checkpoint_dir, "model.pt")

    def load_checkpoint(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))
        # this is currently needed to handle Cori GPU multiple interfaces

    def current_ip(self):
        import socket
        hostname = socket.getfqdn(socket.gethostname())
        self._local_ip = socket.gethostbyname(hostname)
        return self._local_ip
