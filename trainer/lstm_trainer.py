from utils.load_model import get_model
from utils.load_dataset import get_train_val_dataloader, get_metric_loader
from trainer.utils import update_input_output, model_setup, train_one_epoch, validate_one_epoch, get_optimizazion_objects
from omegaconf import ListConfig

from config import *
from models.utils.losses import *

class trainLSTM(tune.Trainable):

    def setup(self, config):
        # Load and set up the configuration
        self.cfg = model_setup(lstm_config_file, config)
        self.cfg, _, _ = update_input_output(self.cfg)    # convert feats and target to lists if they are not already (e.g "all" means all features of dataset)
        self.epochs = self.cfg.opt.epochs
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.n_std = self.cfg.opt.n_std if isinstance(self.cfg.opt.n_std, (list, ListConfig)) else [self.cfg.opt.n_std]

        # Load data
        # try to separate the anomalous sequences from the main dataset anyway. If they are not present, the dataset will be empty
        self.trainloader, self.valloader, self.metrics_loader, self.scaler, self.scaler_params = get_train_val_dataloader(self.cfg, filter_anomalies=True)
        # If the anomalous sequences are not present in the main dataset, the metrics_loader will be None. Try to load it from the path specified in the config file
        self.metrics_loader = get_metric_loader(self.cfg, self.metrics_loader, data_path=self.cfg.opt.metrics_dataset_path,
                                                scale=True, scaler=self.scaler) if self.cfg.opt.evaluate_detection_metrics else None

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
        for epoch in range(self.epochs):
            self.current_epoch = epoch

            train_results = train_one_epoch(
                model=self.model, dataloader=self.trainloader,
                criterion=self.criterion, optimizer=self.optimizer,
                device=self.device, desc=f"Epoch {epoch + 1} [Train]",
            )

            train_loss = train_results["train_loss"]
            print(f"Epoch {epoch + 1} - Avg Train Loss: {train_loss:.6f}")

            evaluate_detection_metrics = self.cfg.opt.evaluate_detection_metrics and (self.metrics_loader is not None and
                                                        (self.current_epoch + 1)%self.cfg.opt.detect_anomaly_epoch_freq==0)
            val_results = validate_one_epoch(
                model=self.model, dataloader=self.valloader,
                metric_loader =self.metrics_loader,
                criterion=self.criterion,device=self.device,
                desc=f"Epoch {epoch + 1} [Val]",
                evaluate_detection_metrics=evaluate_detection_metrics,
                n_std=self.n_std,
                anomaly_threshold=train_results['anomaly_threshold'] if 'anomaly_threshold' in train_results else None
            )
            val_loss = val_results["val_loss"]

            print(f"Epoch {epoch + 1} - Avg Val Loss: {val_loss:.6f}")

            self.scheduler.step(val_loss)

            result = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "parameters_number": self.parameters_number}

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                result["should_checkpoint"] = True

            return result  # For Ray Tune: return after one epoch

        def test_step(self, checkpoint_dir=None):
            raise NotImplementedError("test_lstm method is not implemented yet.")

        def save_checkpoint(self, checkpoint_dir):
            print("this is the checkpoint dir {}".format(checkpoint_dir))
            torch.save({
                'epoch': self.current_epoch, 'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(), 'loss': self.best_val_loss,
                'cfg': self.cfg, 'scaler_params': self.scaler_params,
                'parameters_number': self.parameters_number, 'param_conf': self.parameters_number
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