from ray.data.datasource import FileMetadataProvider

from utils.load_model import get_model
from utils.load_dataset import get_train_val_dataloader, get_metric_loader
from trainer.utils import update_input_output, model_setup, train_one_epoch, validate_one_epoch, get_optimizazion_objects
from omegaconf import ListConfig

from config import *
from models.utils.losses import *

class trainCONVAE1D(tune.Trainable):

    def setup(self, config):
        # Load and set up the configuration
        self.cfg = model_setup(conv_ae_1D_config_file, config, root)
        self.cfg, _, _ = update_input_output(self.cfg)    # convert feats and target to lists if they are not already (e.g "all" means all features of dataset)
        self.epochs = self.cfg.opt.epochs
        self.current_epoch = 0
        self.metric_key, self.mode = list(self.cfg.opt.opt_metric.items())[0]
        self.best_metric = float("inf") if self.mode == "min" else -float("inf")
        self.n_std = self.cfg.opt.n_std if isinstance(self.cfg.opt.n_std, (list, ListConfig)) else [self.cfg.opt.n_std]

        # Load data
        # try to separate the anomalous sequences from the main dataset anyway. If they are not present, the dataset will be empty
        self.trainloader, self.valloader, self.metrics_loader, self.scaler, self.scaler_params = get_train_val_dataloader(self.cfg, filter_anomalies=True)
        # If the anomalous sequences are not present in the main dataset, the metrics_loader will be None. Try to load it from the path specified in the config file
        self.metrics_loader = get_metric_loader(self.cfg, self.metrics_loader, data_path=self.cfg.opt.metrics_dataset_path,
                                                scale=True, scaler=self.scaler) if self.cfg.opt.evaluate_metrics else None

        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.cfg.resources.gpu_trial else "cpu")

        # Build model
        self.model = get_model(self.cfg).to(self.device)
        self.cfg.model.parameter_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.parameters_number = self.cfg.model.parameter_count

        # Optimizer and scheduler
        self.optimizer, self.scheduler, self.criterion = get_optimizazion_objects(self.model, self.cfg)

    def step(self):
        self.current_ip()
        result = self.train_step(checkpoint_dir=None)
        return result

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
            'f1_score': self.val_results.get('f1_score', 0.0),  # 👈 Always included
            "parameters_number": self.parameters_number,
        }

        if self.metric_key in self.val_results:
            result[f"{self.metric_key}"] = self.val_results.get(self.metric_key, 0.0)  # 👈 Always included

        # Track best model
        # example of self.cfg.opt_metric: {'val_loss': 'min'}
        # Step 2: Save model only if current metric is better
        current_metric = result.get(self.metric_key)
        result["should_checkpoint"],  result[f"best_{self.metric_key}"]= self.check_improvements(current_metric)
        result["best_n_std"] = self.val_results.get("best_n_std", 0.0)  # 👈 Always included

        return result

    def test_step(self, checkpoint_dir=None):
        raise NotImplementedError("test_lstm method is not implemented yet.")

    def save_checkpoint(self, checkpoint_dir):
        print("this is the checkpoint dir {}".format(checkpoint_dir))
        torch.save({
                'epoch': self.current_epoch, 'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(), 'loss': self.metric_key,
                'cfg': self.cfg, 'scaler_params': self.scaler_params,
                'parameters_number': self.parameters_number,'param_conf': self.parameters_number,
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

    def check_improvements(self, current_metric):
        """
        Check if the current metric is better than the best metric.
        """
        if (self.mode == "min" and current_metric < self.best_metric) or \
                (self.mode == "max" and current_metric > self.best_metric):
            self.best_metric = current_metric
            return True, current_metric
        else:
            return False, self.best_metric

