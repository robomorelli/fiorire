import torch.nn
from ray import tune
from utils.load_model import get_model
from utils.load_dataset import get_dataset
from omegaconf import OmegaConf
from config import *
from models.utils.losses import *

class trainAe(tune.Trainable):

    def setup(self, config):
        # setup function is invoked once training starts
        # setup function is invoked once training starts
        # setup function is invoked once training starts
        self.cfg = OmegaConf.load(config_path + ae_config_file) #here use only vae conf file
        self.model_name = '_'.join((self.cfg.model.name, self.cfg.dataset.name)) + '.h5'

        #following config keys have to be in the config file of this model
        self.lr = config['lr']
        self.intermediate_dim = config['intermediate_dim']
        self.code_dim = config['code_dim']
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']

        self.trainloader, self.valloader, _, _\
            , self.original_dim = get_dataset(self.cfg, batch_size=self.batch_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() and self.cfg.resources.gpu_trial else "cpu")
        self.model = get_model(self.cfg, original_dim=self.original_dim, intermediate_dim=self.intermediate_dim,
                                code_dim=self.code_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def step(self):
        self.current_ip()
        val_loss = self.train_ae(checkpoint_dir=None)
        result = {"val_loss": val_loss}
        return result

    def train_ae(self, checkpoint_dir=None):
        ####Train Loop####
        """
        Set the models to the training mode first and train
        """
        train_loss = []
        patience = 1
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.8, patience=patience
                                                               , threshold=0.0001, threshold_mode='rel',
                                                               cooldown=0, min_lr=9e-8, verbose=True)

        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            for i, (x, y) in enumerate(self.trainloader):
                self.optimizer.zero_grad()

                x = x.to(self.device)
                y = y.to(self.device)
                out = self.model(x)

                loss = torch.nn.BCELoss()(out, y)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                train_loss.append(loss.item())

                if i % 10 == 0:
                    print("Loss: {}".format(loss.item()))

            ###############################################
            # eval mode for evaluation on validation dataset
            ###############################################

            # Validation loss
            # val_loss = 0.0
            temp_val_loss = 0.0
            val_steps = 0
            self.model.eval()
            for i, (x, y) in enumerate(self.valloader, 0):
                with torch.no_grad():
                    x = x.to(self.device)

                    self.optimizer.zero_grad()

                    x = x.to(self.device)
                    y = y.to(self.device)
                    out = self.model(x)

                    temp_val_loss += torch.nn.BCELoss()(out, y)

                    val_steps += 1
            val_loss = temp_val_loss / len(self.valloader)
            val_loss_cpu = val_loss.cpu().item()
            print('validation_loss {}'.format(val_loss_cpu))
            scheduler.step(val_loss)

            return val_loss_cpu

    def save_checkpoint(self, checkpoint_dir):
        print("this is the checkpoint dir {}".format(checkpoint_dir))
        checkpoint_path = os.path.join(checkpoint_dir, self.model_name)
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))

        # this is currently needed to handle Cori GPU multiple interfaces

    def current_ip(self):
        import socket
        hostname = socket.getfqdn(socket.gethostname())
        self._local_ip = socket.gethostbyname(hostname)
        return self._local_ip