from utils.load_model import get_model
from utils.load_dataset import get_dataset
from trainer.utils import infer_input_output, model_setup
from tqdm import tqdm
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
        self.model_name = os.path.join(self.cfg.model.name + '.h5')  # the name of the saved model

        # Merge model and opt into cfg
        self.cfg.dataset.feats = feats
        self.cfg.dataset.target = target
        self.epochs = self.cfg.opt.epochs

        # Load data
        self.trainloader, self.valloader, self.n_features, self.scaler, self.scaler_params = get_dataset(self.cfg)

        print('number of training data {}'.format(len(self.trainloader.dataset)))
        # Add dataset info into cfg.dataset
        self.cfg.dataset.n_features = self.n_features    # Needed to specify the input channel of the model
        self.cfg.model.output_size = self.n_features if not self.cfg.dataset.target else len(self.cfg.target)    # Needed to specify the output channel of the model
        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.cfg.resources.gpu_trial else "cpu")
        # Build model
        self.model = get_model(self.cfg).to(self.device)

        # Optimizer and scheduler
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.opt.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min',
            factor=0.8,
            patience=self.cfg.opt.lr_patience,
            threshold=0.0001,
            threshold_mode='rel',
            cooldown=0,
            min_lr=9e-8,
            verbose=True
        )
        self.criterion = nn.MSELoss()
        self.best_val_loss = 10 ** 16

        # Store parameter count for logging
        self.cfg.model.parameter_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.parameters_number = self.cfg.model.parameter_count

    def step(self):
        self.current_ip()
        result = self.train_lstm(checkpoint_dir=None)
        return result

    def train_lstm(self, checkpoint_dir=None):
        """
        Train loop for LSTM with tqdm progress bars
        """
        for epoch in tqdm(range(self.epochs), unit='epoch', desc="Epochs"):
            self.current_epoch = epoch
            temp_train_loss = 0
            train_steps = 0

            # Training phase with tqdm
            self.model.train()
            train_loader_tqdm = tqdm(self.trainloader, desc=f"Training Epoch {epoch + 1}", unit="batch", leave=False)
            for i, batch in enumerate(train_loader_tqdm):
                self.optimizer.zero_grad()

                inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                loss.backward()
                self.optimizer.step()

                temp_train_loss += loss.item()
                train_steps += 1

                if i % 10 == 0:
                    train_loader_tqdm.set_postfix(loss=loss.item())

            train_loss = temp_train_loss / train_steps
            print(f"[Epoch {epoch + 1}] Train loss: {train_loss:.4f}")

            # Validation phase with tqdm
            self.model.eval()
            temp_val_loss = 0
            val_steps = 0

            with torch.no_grad():
                val_loader_tqdm = tqdm(self.valloader, desc="Validating", unit="batch", leave=False)
                for batch in val_loader_tqdm:
                    inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets).item()
                    temp_val_loss += loss
                    val_steps += 1

            val_loss = temp_val_loss / val_steps
            print(f"[Epoch {epoch + 1}] Validation loss: {val_loss:.4f}")

            self.scheduler.step(val_loss)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                return {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "parameters_number": self.parameters_number,
                    "should_checkpoint": True
                }
            else:
                return {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "parameters_number": self.parameters_number
                }

    def test_lstm(self, checkpoint_dir=None):
        self.model.eval()
        test_loss = 0.0
        test_steps = 0

        with torch.no_grad():
            for batch in tqdm(self.valloader, desc="Testing", unit="batch"):
                inputs, _, targets = batch[0].to(self.device), batch[1], batch[1].to(self.device)
                outputs, enc, preds = self.model(inputs)
                loss = self.criterion(preds, inputs).item()
                test_loss += loss
                test_steps += 1

        test_loss /= test_steps
        print(f"Test loss: {test_loss:.4f}")
        return {"test_loss": test_loss}

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