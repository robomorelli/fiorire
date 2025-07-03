from ray import tune
from trainer.utils import get_model
from trainer.utils import get_dataset
from omegaconf import OmegaConf
from tqdm import tqdm
from torch import nn
from config import *
from models.utils.losses import *

class trainLSTMVAE(tune.Trainable):

    def setup(self, config):
        #print(' print args config file', args.config_file)
        self.cfg = OmegaConf.load(config_path + lstm_vae_config_file) #here use only vae conf file
        self.model_name = '_'.join((self.cfg.model.name, self.cfg.dataset.name)) + '.h5'

        #following config keys have to be in the config file of this model
        self.seq_in_length = config['seq_in_length']
        self.embedding_dim = config['embedding_dim']
        self.latent_dim = config['latent_dim']
        self.n_layers_1 = config['n_layers_1']
        self.n_layers_2 = config['n_layers_2']
        self.lr = config['lr']
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        self.lr_patience = config['lr_patience']
        self.kld_factor = torch.tensor(config['kld_factor']).float()
        self.recon_loss_type = config['recon_loss_type']

        self.sample_rate = self.cfg.dataset.sample_rate
        self.target = self.cfg.dataset.target
        self.predict = self.cfg.dataset.predict

        # to write on cfg to have later on load dataset
        self.cfg.dataset.sequence_length = config['seq_in_length']
        self.cfg.dataset.out_window = config['seq_in_length']

        self.trainloader, self.valloader, n_features, scaled, columns_subset, dataset_subset,_, dataset_name, data_path\
                                = get_dataset(self.cfg, batch_size=self.batch_size, sequence_length = config['seq_in_length'])

        self.scaled = scaled
        self.n_features = n_features
        self.columns_subset = columns_subset
        self.dataset_subset = dataset_subset
        self.Nf_lognorm = n_features

        self.Nf_binomial = n_features - self.Nf_lognorm

        self.device = torch.device("cuda" if torch.cuda.is_available() and self.cfg.resources.gpu_trial else "cpu")
        self.model = get_model(self.cfg, sequence_length=self.seq_in_length, no_features=n_features,
                               embedding_dim=self.embedding_dim,
                                latent_dim=self.latent_dim, n_layers_1=self.n_layers_1,
                               n_layers_2=self.n_layers_2, output_size=n_features,
                               recon_loss_type=self.recon_loss_type).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.8,
                                                            patience=self.lr_patience, threshold=0.0001,
                                                            threshold_mode='rel',
                                                            cooldown=0, min_lr=9e-8, verbose=True)

        self.best_val_loss = 10 ** 16

        self.param_conf = {'columns': self.n_features, 'sequence_length': self.seq_in_length,
                            'batch_size':self.batch_size,'predict': self.predict,
                            'device': self.device, 'out_window': self.seq_in_length,
                            'n_features': self.n_features, 'scaled':self.scaled,
                            'sampling_rate': self.sample_rate, 'output_size': self.n_features,
                            'embedding_dim': self.embedding_dim, 'latent_dim': self.latent_dim,
                            'n_layers_1': self.n_layers_1, 'n_layers_2': self.n_layers_2,
                            'kld_factor': self.kld_factor,
                            'Nf_lognorm': self.Nf_lognorm, "recon_loss_type": self.recon_loss_type}

        self.parameters_number = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def step(self):
        self.current_ip()
        result = self.train_lstm_vae(checkpoint_dir=None)
        return result

    def train_lstm_vae(self, checkpoint_dir=None):
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

                if self.recon_loss_type == 'custom':
                    x, mu, log_var, pars = self.model(batch[0].to(self.device))
                    recon_loss = loss_function(x, pars, Nf_lognorm=self.Nf_lognorm,
                                               Nf_binomial = self.Nf_binomial).mean()
                    mse_loss = nn.MSELoss()(x, pars[0])
                else:
                    x, mu, log_var, y = self.model(batch[0].to(self.device))
                    recon_loss = nn.MSELoss()(x, y)
                    mse_loss = nn.MSELoss()(x, y)

                KLD = KL_loss(mu, log_var)

                loss = recon_loss + self.kld_factor * KLD # the sum of KL is added to the mean of MSE
                loss.backward()

                # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                temp_train_loss += loss.item()
                train_steps += 1

                # if (i + 1) % config['gradient_accumulation_steps'] == 0:
                self.optimizer.step()

                if i % 10 == 0:
                    if self.recon_loss_type == 'custom':
                        print("loss {}: recon loss {} and kld {} mse_loss {}".format(loss, recon_loss, KLD, mse_loss))
                    else:
                        print("loss {}: recon loss (mse_loss) {} and kld {} ".format(loss, recon_loss, KLD))
                    print(loss.item())

            temp_train_loss = temp_train_loss / train_steps
            train_loss = temp_train_loss
            print('train loss at the end of epoch is ', train_loss)

            self.model.eval()
            val_steps = 0
            temp_val_loss = 0
            mse_val_loss = 0
            with torch.no_grad():
                for i, batch in tqdm(enumerate(self.valloader), total=len(self.valloader), desc="Evaluating"):

                    if self.recon_loss_type == 'custom':
                        x, mu, log_var, pars = self.model(batch[0].to(self.device))
                        recon_loss = loss_function(x, pars, Nf_lognorm=self.Nf_lognorm,
                                                   Nf_binomial=self.Nf_binomial).mean()
                        mse_val_loss += nn.MSELoss()(x, pars[0]).item()
                    else:
                        x, mu, log_var, y = self.model(batch[0].to(self.device))
                        recon_loss = nn.MSELoss()(x, y)
                        mse_val_loss += nn.MSELoss()(x, y).item()

                    KLD = KL_loss(mu, log_var)

                    loss = recon_loss + self.kld_factor * KLD
                    temp_val_loss += loss
                    val_steps += 1

            temp_val_loss = temp_val_loss / val_steps
            mse_val_loss = mse_val_loss / val_steps
            val_loss = temp_val_loss.item()

            if i % 10 == 0:
                if self.recon_loss_type == 'custom':
                    print("loss {}: recon loss {} and kld {} mse_loss {}".format(val_loss
                                                                                 , recon_loss, KLD, mse_val_loss))
                else:
                    print("loss {}: recon loss (mse_loss) {} and kld {} ".format(val_loss
                                                                                 , recon_loss, KLD))
                print(loss.item())

            self.scheduler.step(val_loss)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                return {"train_loss": train_loss, 'parameters_number': self.parameters_number,
                        "val_loss": val_loss, "mse_val_loss":mse_val_loss, "should_checkpoint": True}
            else:
                return {"train_loss": train_loss,'parameters_number': self.parameters_number,
                        "val_loss": val_loss, "mse_val_loss":mse_val_loss}

    def test_lstm_vae(self, checkpoint_dir=None):
        test_loss = 0.0
        test_steps = 0
        self.model.eval()

        with torch.no_grad():
            for i, batch in tqdm(enumerate(self.valloader), total=len(self.valloader), desc="Evaluating"):
                x, mu, log_var, pars = self.model(batch[0].to(self.device))
                recon_loss = loss_function(x, pars, self.Nf_lognorm,
                                           self.Nf_binomial).mean()
                KLD = KL_loss(mu, log_var)
                loss = recon_loss + self.kld_factor * KLD
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
                'parameters_number': self.parameters_number,
                'cfg': self.cfg,
            'param_conf': self.param_conf
            }, f"{checkpoint_dir}/model.pt")
        return os.path.join(checkpoint_dir, "model.pt")

    def load_checkpoint(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))


    def current_ip(self):
        import socket
        hostname = socket.getfqdn(socket.gethostname())
        self._local_ip = socket.gethostbyname(hostname)
        return self._local_ip