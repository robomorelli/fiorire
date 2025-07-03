from ray import tune
from trainer.utils import get_model
from trainer.utils import get_dataset
from omegaconf import OmegaConf
from config import *
from models.utils.losses import *

class trainVae(tune.Trainable):

    def setup(self, config):
        # setup function is invoked once training starts
        # setup function is invoked once training starts
        # setup function is invoked once training starts
        self.cfg = OmegaConf.load(config_path + vae_config_file) #here use only vae conf file
        self.model_name = '_'.join((self.cfg.model.name, self.cfg.dataset.name)) + '.h5'
        self.lr = config['lr']

        self.intermediate_dim = config['intermediate_dim']
        self.latent_dim = config['latent_dim']
        self.weight_KL_loss = config['weight_KL_loss']
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']

        self.trainloader, self.valloader, self.n_lognorm, self.n_binomial\
            , self.original_dim = get_dataset(self.cfg, batch_size=self.batch_size)

        if 'weights_loss' in config:
            if len(config['weights_loss']) == self.original_dim:
                self.weights_loss = config['weights_loss']
            elif len(config['weights_loss']) < self.original_dim:
                discrepancy = self.original_dim - len(config['weights_loss'])
                print('padding the weight loss with {} 1 weight'.format(discrepancy))
                padding = [1] * discrepancy
                self.weights_loss = config['weights_loss'] + padding
            elif len(config['weights_loss']) > self.original_dim:
                discrepancy = self.original_dim - len(config['weights_loss'])
                print('padding the weight loss with {} 1 weight'.format(discrepancy))
                padding = [1] * discrepancy
                self.weights_loss = config['weights_loss'][:self.original_dim]
        else:
            self.weights_loss = [1]*self.original_dim

        self.device = torch.device("cuda" if torch.cuda.is_available() and self.cfg.resources.gpu_trial else "cpu")
        self.model = get_model(self.cfg, original_dim=self.original_dim, intermediate_dim=self.intermediate_dim,
                                latent_dim=self.latent_dim, n_lognorm=self.n_lognorm,
                               n_binomial=self.n_binomial).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def step(self):
        self.current_ip()
        val_loss = self.train_vae(checkpoint_dir=None)
        result = {"val_loss": val_loss}
        # if detect_instance_preemption():
        #    result.update(should_checkpoint=True)
        # acc = test(self.models, self.test_loader, self.device)
        # don't call report here!
        return result

    def train_vae(self, checkpoint_dir=None):

        #cuda = torch.cuda.is_available()
        #if cuda:
        #    print('added visible gpu')
        #    ngpus = torch.cuda.device_count() #needed if more trainer per gpu o more gpu per trainer

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
            for i, (x, x) in enumerate(self.trainloader):
                self.optimizer.zero_grad()

                x = x.to(self.device)
                pars, mu, sigma, mu_prior, sigma_prior = self.model(x)

                recon_loss = RecoProb_forVAE_wrapper(x, pars, self.n_lognorm,
                                           self.n_binomial, self.weights_loss).mean()

                KLD = KL_loss_forVAE(mu, sigma, mu_prior, sigma_prior).mean()
                loss = recon_loss + self.weight_KL_loss * KLD  # the mean of KL is added to the mean of MSE
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                train_loss.append(loss.item())

                if i % 10 == 0:
                    print("Loss: {}".format(loss.item()))
                    print("kl div {}".format(KLD))

            ###############################################
            # eval mode for evaluation on validation dataset
            ###############################################

            # Validation loss
            temp_val_loss = 0.0
            val_steps = 0
            self.model.eval()
            for i, (x, x) in enumerate(self.valloader, 0):
                with torch.no_grad():
                    x = x.to(self.device)
                    pars, mu, sigma, mu_prior, sigma_prior = self.model(x)
                    recon_loss = RecoProb_forVAE_wrapper(x, pars, self.n_lognorm,
                                               self.n_binomial, self.weights_loss).mean()

                    KLD = KL_loss_forVAE(mu, sigma, mu_prior, sigma_prior).mean()
                    temp_val_loss += recon_loss + self.weight_KL_loss * KLD

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