from ray import tune
from utils.load_model import get_model
from utils.load_dataset import get_train_val_dataloader, get_metric_loader
from trainer.utils import update_input_output, model_setup, train_one_epoch, validate_one_epoch, get_optimizazion_objects
from omegaconf import OmegaConf
from tqdm import tqdm
from config import *
from models.utils.losses import *

class trainLSTMAE(tune.Trainable):

    def setup(self, config):
        self.cfg = OmegaConf.load(config_path + lstm_ae_config_file) #here use only vae conf file
        self.model_name = '_'.join((self.cfg.model.name, self.cfg.dataset.name)) + '.h5'

        #following config keys have to be in the config file of this model
        self.seq_in_length = config['seq_in_length']
        self.embedding_dim = config['embedding_dim']
        self.latent_dim = config['latent_dim']
        self.n_layers_1 = config['n_layers_cell_1']
        self.n_layers_2 = config['n_layers_cell_2']
        self.lr = config['lr']
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        self.lr_patience = config['lr_patience']
        self.activation = config['activation']

        self.act_dict = {'Relu': nn.ReLU(), 'Elu': nn.ELU(), 'Selu': nn.SELU(),'LRelu': nn.LeakyReLU(), 'None': None}
        self.activation = self.act_dict[self.activation]

        self.sample_rate = self.cfg.dataset.sample_rate
        self.target = self.cfg.dataset.target
        self.predict = self.cfg.dataset.predict

        # to write on cfg to have later on load dataset
        self.cfg.dataset.sequence_length = self.seq_in_length
        self.cfg.dataset.out_window = self.seq_in_length

        self.trainloader, self.valloader, n_features, scaled, columns_subset, dataset_subset,_, dataset_name, data_path\
                                = get_dataset(self.cfg, batch_size=self.batch_size, sequence_length = config['seq_in_length'])

        print('number of training data {}'.format(len(self.trainloader.dataset)))

        self.scaled = scaled
        self.n_features = n_features
        self.columns_subset = columns_subset
        self.dataset_subset = dataset_subset

        self.device = torch.device("cuda" if torch.cuda.is_available() and self.cfg.resources.gpu_trial else "cpu")
        self.model = get_model(self.cfg, seq_in_length=self.seq_in_length, n_features=n_features, embedding_dim=self.embedding_dim,
                                latent_dim=self.latent_dim, n_layers_1=self.n_layers_1, n_layers_2=self.n_layers_2,
                               seq_out_lenght=self.seq_in_length, activation=self.activation).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.8,
                                                            patience=self.lr_patience, threshold=0.0001, threshold_mode='rel',
                                                            cooldown=0, min_lr=9e-8, verbose=True)

        self.criterion = nn.MSELoss()

        self.best_val_loss = 10 ** 16


        self.param_conf = {'columns': self.n_features, 'sequence_length':self.seq_in_length,
                           'batch_size':self.batch_size ,'predict':self.predict,
                        'device': self.device, 'out_window':self.seq_in_length,
                           'n_features':self.n_features, 'scaled':self.scaled,
                        'sampling_rate':self.sample_rate, 'output_size': self.n_features,
                           'embedding_dim': self.embedding_dim, 'latent_dim':self.latent_dim,
                           'n_layers_1': self.n_layers_1,
                           'n_layers_2': self.n_layers_2,
                           'activation':self.activation}
        self.parameters_number = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def step(self):
        self.current_ip()
        result = self.train_lstm_ae(checkpoint_dir=None)
        return result

    def train_lstm_ae(self, checkpoint_dir=None):
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
                x, enc, y_o = self.model(batch[0].to(self.device))
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
                    x_o, enc, y_o = self.model(batch[0].to(self.device))
                    loss = self.criterion(y_o.to(self.device), x_o.to(self.device)).item()
                    temp_val_loss += loss
                    val_steps += 1

            temp_val_loss = temp_val_loss / val_steps
            val_loss = temp_val_loss
            print('eval loss {}'.format(val_loss))
            self.scheduler.step(val_loss)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                return {"train_loss": train_loss, 'parameters_number': self.parameters_number,
                        "val_loss": val_loss, "should_checkpoint": True}
            else:
                return {"train_loss": train_loss,'parameters_number': self.parameters_number,
                        "val_loss": val_loss}

    def test_lstm_ae(self, checkpoint_dir=None):
        test_loss = 0.0
        test_steps = 0
        self.model.eval()

        with torch.no_grad():
            for i, batch in tqdm(enumerate(self.valloader), total=len(self.valloader), desc="Evaluating"):
                x_o, enc, y_o = self.model(batch[0].to(self.device))
                loss = self.criterion(y_o.to(self.device), x_o.to(self.device)).item()
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

        # this is currently needed to handle Cori GPU multiple interfaces

    def current_ip(self):
        import socket
        hostname = socket.getfqdn(socket.gethostname())
        self._local_ip = socket.gethostbyname(hostname)
        return self._local_ip