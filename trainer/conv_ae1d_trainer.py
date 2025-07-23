from ray import tune
from utils.load_model import get_model
from utils.load_dataset import get_dataset
from omegaconf import OmegaConf
from tqdm import tqdm
from torch import nn
from config import *
from models.utils.losses import *

class trainCONVAE1D(tune.Trainable):

    def setup(self, config):
        self.cfg = OmegaConf.load(config_path + conv_ae_1D_config_file) #here use only vae conf file
        self.model_name = '_'.join((self.cfg.model.name, self.cfg.dataset.name)) + '.h5'

        self.length = config['seq_in_length']
        self.n_layers = config['n_layers']
        self.filter_num = config['filter_num']
        self.lr = config['lr']
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        self.lr_patience = config['lr_patience']
        self.activation = config['activation']
        self.kernel_size = config['kernel_size']
        self.pool = config['pool']
        self.dilation = config['dilation']

        self.increasing = config['increasing'] if 'increasing' in list(config.keys()) else False
        self.flattened = config['flattened'] if 'flattened' in list(config.keys()) else False
        self.latent_dim = config['latent_dim'] if 'latent_dim' in list(config.keys()) else 60

        if not self.pool:
            self.stride = 2
        self.predict = 0

        # to write on cfg to have later on load dataset
        self.cfg.dataset.sequence_length = self.length
        self.cfg.dataset.out_window = self.length
        self.sample_rate = self.cfg.dataset.sample_rate
        self.target = False

        self.act_dict = {'Relu': nn.ReLU(), 'Elu': nn.ELU(), 'Selu': nn.SELU(),'LRelu': nn.LeakyReLU()}
        self.activation = self.act_dict[self.activation]

        self.trainloader, self.valloader, n_features, scaled, scale, columns_subset,\
        dataset_subset, train_val_split, dataset, data_path = get_dataset(self.cfg, batch_size=self.batch_size
                                                                          , sequence_length=config['seq_in_length'])
        self.scaled = scaled
        self.n_features = n_features
        self.columns_subset = columns_subset
        self.dataset_subset = dataset_subset
        self.train_val_split = train_val_split
        self.dataset = dataset
        self.data_path = data_path
        self.scale = scale
        self.padding = int((self.dilation * (self.kernel_size-1)/2))


        if self.pool == False:
            self.stride = 2
        else:
            self.stride = 1

        self.device = torch.device("cuda" if torch.cuda.is_available() and self.cfg.resources.gpu_trial else "cpu")

        self.model = get_model(self.cfg, in_channel=self.n_features ,  length=self.length,
                               kernel_size=self.kernel_size,
                               filter_num=self.filter_num, n_layers=self.n_layers,
                               activation=self.activation, pool=self.pool,
                               stride = self.stride, padding = self.padding,
                               increasing=self.increasing, flattened=self.flattened,
                               latent_dim=self.latent_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.8,
                                                            patience=self.lr_patience, threshold=0.0001, threshold_mode='rel',
                                                            cooldown=0, min_lr=9e-8, verbose=True)

        print(self.model)

        self.criterion = nn.MSELoss()
        self.best_val_loss = 10 ** 16

        self.param_conf = {'columns': self.n_features, 'length':self.length ,'predict':self.predict, 'sequence_length':self.length,
                        'device': self.device, 'out_window':self.length, 'in_channel':self.n_features, 'n_features':self.n_features, 'scaled':self.scaled,
                        'sampling_rate':self.sample_rate, 'columns_subset': self.columns_subset, 'dataset_subset':self.dataset_subset,
                        'target': self.target, 'batch_size': self.batch_size, 'train_val_split': self.train_val_split,
                        'data_path': self.data_path, 'dataset': self.dataset, 'activation': self.activation, 'kernel_size':self.kernel_size,
                        'filter_num':self.filter_num, 'n_layers':self.n_layers,
                        'pool':self.pool, 'stride':self.stride, 'padding':self.padding
                        ,'increasing':self.increasing, 'flattened':self.flattened,
                        'latent_dim':self.latent_dim, 'scale':self.scale}

        self.parameters_number = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

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