from utils.load_model import get_model
from utils.load_dataset import get_dataset, get_train_val_dataset
from trainer.utils import infer_input_output, model_setup
from trainer.utils import train_one_epoch, validate_one_epoch
from config import *
from models.utils.losses import *

class trainLSTM(tune.Trainable):

    def setup(self, config):
        # Load and set up the configuration
        self.cfg = model_setup(lstm_config_file, config)
        # Handle 'feats'
        feats, target = infer_input_output(self.cfg)

        # Merge model and opt into cfg
        self.cfg.dataset.feats = feats
        self.cfg.dataset.target = target
        self.epochs = self.cfg.opt.epochs
        self.current_epoch = 0
        self.model_name = os.path.join(self.cfg.model.name + '.h5')  # the name of the saved model

        # Load data
        self.trainloader, self.valloader, self.n_features, self.scaler, self.scaler_params = get_train_val_dataset(self.cfg)

        # Add dataset info into cfg.dataset
        self.cfg.dataset.n_features = self.n_features    # Needed to specify the input channel of the model
        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.cfg.resources.gpu_trial else "cpu")
        # Build model
        self.cfg.model.output_size = self.n_features if not self.cfg.dataset.target else len(self.cfg.target)    # Needed to specify the output channel of the model
        self.model = get_model(self.cfg).to(self.device)
        # Optimizer and scheduler
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.opt.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min', factor=0.8,
            patience=self.cfg.opt.lr_patience, threshold=0.0001,
            threshold_mode='rel', cooldown=0,
            min_lr=9e-8, verbose=True)
        self.criterion = nn.MSELoss()
        self.best_val_loss = 10 ** 16

        # Store parameter count for logging
        self.cfg.model.parameter_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.parameters_number = self.cfg.model.parameter_count

    def step(self):
        self.current_ip()
        result = self.train_step(checkpoint_dir=None)
        return result

    def train_step(self, checkpoint_dir=None):
        for epoch in range(self.epochs):
            self.current_epoch = epoch

            train_loss = train_one_epoch(
                model=self.model,
                dataloader=self.trainloader,
                criterion=self.criterion,
                optimizer=self.optimizer,
                device=self.device,
                desc=f"Epoch {epoch + 1} [Train]",
            )

            print(f"Epoch {epoch + 1} - Avg Train Loss: {train_loss:.6f}")

            val_loss = validate_one_epoch(
                model=self.model,
                dataloader=self.valloader,
                criterion=self.criterion,
                device=self.device,
                desc=f"Epoch {epoch + 1} [Val]",
            )

            print(f"Epoch {epoch + 1} - Avg Val Loss: {val_loss:.6f}")

            self.scheduler.step(val_loss)

            result = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "parameters_number": self.parameters_number,
            }

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                result["should_checkpoint"] = True

            return result  # For Ray Tune: return after one epoch

    def test_step(self, checkpoint_dir=None):
        raise NotImplementedError("test_lstm method is not implemented yet.")

    def save_checkpoint(self, checkpoint_dir):
        print("this is the checkpoint dir {}".format(checkpoint_dir))
        torch.save({
                'epoch': self.current_epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': self.best_val_loss,
                'parameters_number': self.parameters_number,
                'cfg': self.cfg,
            'param_conf': self.param_conf
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