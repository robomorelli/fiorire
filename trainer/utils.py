from utils.general import make_paths_absolute, extract_config, extract_fixed_config
from omegaconf import ListConfig
from omegaconf import OmegaConf
from config import *

class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True


def model_setup(config_file_name, config):

    cfg = OmegaConf.load(config_path + config_file_name)  # here use only vae conf file
    make_paths_absolute(cfg)
    # Allow dynamic field insertion
    OmegaConf.set_struct(cfg, False)

    # Merge trial parameters from Ray Tune into OmegaConf config
    for k, v in config.items():
        OmegaConf.update(cfg, k, v, merge=True)
    # Construct dataset config and merge

    return cfg

def infer_input_output(cfg):
    """
    Infer input and output dimensions from the configuration.
    :param cfg: configuration object
    :return: input_dim, output_dim
    """
    if isinstance(cfg.dataset.feats, (list, ListConfig)):
        feats = cfg.dataset.feats
    elif cfg.dataset.feats == 'all':
        feats = all_feats_dict[cfg.dataset.name]
    else:
        feats = [cfg.dataset.feats]

    # Handle 'target'
    if isinstance(cfg.dataset.target, (list, ListConfig)):
        target = cfg.dataset.target
    elif isinstance(cfg.dataset.target, str):
        if cfg.dataset.target == 'all':
            target = all_feats_dict[cfg.dataset.name]
        else:
            target = [cfg.dataset.target]
    else:
        target = None

    return feats, target