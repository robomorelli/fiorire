from ray import tune
from utils.load_model import get_model
from utils.load_dataset import get_dataset
from omegaconf import OmegaConf
from tqdm import tqdm
from omegaconf import ListConfig
from config import *
from models.utils.losses import *

class trainCONVAE1D(tune.Trainable):

    def setup(self, config):

        self.cfg = OmegaConf.load(config_path + conv_ae_1D_config_file) #here use only vae conf file
        self.model_name = os.path.join(self.cfg.model.name + '.h5')
        trial_config = config

        # Model trial params
        model_config = {
            'seq_in_length': trial_config['seq_in_length'],
            'n_layers': trial_config['n_layers'],
            'filter_num': trial_config['filter_num'],
            'activation': trial_config['activation'],
            'kernel_size': trial_config['kernel_size'],
            'pool': trial_config['pool'],
            'dilation': trial_config['dilation'],
            'increasing': trial_config.get('increasing', False),
            'flattened': trial_config.get('flattened', False),
            'latent_dim': trial_config.get('latent_dim', 60),
            'stride': 1 if trial_config['pool'] else 2,
            'name': self.cfg.model.name  # preserve model name
        }

        # Optimization trial params
        opt_config = {
            'lr': trial_config['lr'],
            'batch_size': trial_config['batch_size'],
            'epochs': trial_config['epochs'],
            'lr_patience': trial_config['lr_patience']
        }

        # Construct dataset config and merge
        dataset_config = {
            #'scaler': trial_config['scaler'],
            'feats': self.cfg.dataset.feats if isinstance(self.cfg.dataset.feats, (list, ListConfig))
             else all_feats_dict[self.cfg.dataset.name] if  self.cfg.dataset.feats == 'all'
             else [],
            'dataset_subset': self.cfg.dataset.dataset_subset,
            'train_val_split': self.cfg.dataset.train_val_split,
            'batch_size': trial_config['batch_size'],
            'data_path': self.cfg.dataset.data_path
        }

        # Merge model and opt into cfg
        self.cfg.model = OmegaConf.merge(self.cfg.model, model_config)
        self.cfg.opt = OmegaConf.merge(self.cfg.opt, opt_config)
        self.cfg.dataset = OmegaConf.merge(self.cfg.dataset, dataset_config)
        self.epochs = self.cfg.opt.epochs

        # Load data
        self.trainloader, self.valloader, self.n_features, self.scaler, self.scaler_params = get_dataset(self.cfg)


        # Add dataset info into cfg.dataset
        self.cfg.dataset.n_features = self.n_features    # Needed to specify the input channel of the model
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

    def step(self):
        self.current_ip()
        result = self.train_conv_ae1D(checkpoint_dir=None)
        return result

    def train_conv_ae1D(self, checkpoint_dir=None):
        ####Train Loop####
        """
        Set the models to the training mode first and train
        """
        for epoch in tqdm(range(self.epochs), unit='epoch'):
            self.current_epoch = epoch
            temp_train_loss = 0
            train_steps = 0
            for i, batch in tqdm(enumerate(self.trainloader), total=len(self.trainloader), unit="batch"):
                self.model.train()
                self.optimizer.zero_grad()

                # y.requires_grad_(True)
                y_o = self.model(batch[0].to(self.device))
                loss = self.criterion(y_o.to(self.device), batch[1].to(self.device))
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                temp_train_loss += loss.item()
                train_steps += 1
                # if (i + 1) % config['gradient_accumulation_steps'] == 0:
                self.optimizer.step()

                if i % 10 == 0:
                    print("Loss:")
                    print(loss.item())

            temp_train_loss = temp_train_loss / train_steps
            train_loss = temp_train_loss
            print('train loss at the end of epoch is ', train_loss)

            self.model.eval()
            val_steps = 0
            temp_val_loss = 0
            with torch.no_grad():
                for i, batch in tqdm(enumerate(self.valloader), total=len(self.valloader), desc="Evaluating"):
                    y_o = self.model(batch[0].to(self.device))
                    loss = self.criterion(batch[0].to(self.device), y_o.to(self.device)).item()
                    temp_val_loss += loss
                    val_steps += 1

            temp_val_loss = temp_val_loss / val_steps
            val_loss = temp_val_loss
            print('eval loss {}'.format(val_loss))
            self.scheduler.step(val_loss)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                return {"train_loss": train_loss,
                        "val_loss": val_loss, 'parameters_number': self.parameters_number,
                        "should_checkpoint": True}
            else:
                return {"train_loss": train_loss,'parameters_number': self.parameters_number,
                        "val_loss": val_loss}

    def test_conv_ae(self, checkpoint_dir=None):
        test_loss = 0.0
        test_steps = 0
        self.model.eval()

        with torch.no_grad():
            for i, batch in tqdm(enumerate(self.valloader), total=len(self.valloader), desc="Evaluating"):
                y_o = self.model(batch[0].to(self.device))
                loss = self.criterion(y_o.to(self.device), batch[1].to(self.device)).item()
                test_loss += loss
                test_steps += 1

        test_loss = test_loss / test_steps
        test_loss_cpu = test_loss.cpu()
        print('test_loss {}'.format(test_loss_cpu))
        return {"test_loss": test_loss_cpu}


    def save_checkpoint(self, checkpoint_dir):
        print("this is the checkpoint dir {}".format(checkpoint_dir))
        torch.save({
                'epoch': self.current_epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': self.best_val_loss,
                'cfg': self.cfg,
                'scaler_params': self.scaler_params,
                'parameters_number': self.parameters_number,
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