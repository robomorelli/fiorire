from utils.general import make_paths_absolute, extract_config, extract_fixed_config
from omegaconf import ListConfig
from omegaconf import OmegaConf
import torch
from tqdm import tqdm
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

def update_input_output(cfg):
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

    # Merge model and opt into cfg
    cfg.dataset.feats = feats
    cfg.dataset.target = target

    return cfg, feats, target

def get_optimizazion_objects(model, cfg):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.opt.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=0.8, patience=cfg.opt.lr_patience, threshold=0.0001,
        threshold_mode='rel', cooldown=0,min_lr=9e-8, verbose=True)
    criterion = nn.MSELoss()

    return optimizer, scheduler, criterion


def train_one_epoch(model, dataloader, criterion, optimizer, device, desc="Train"):
    model.train()
    epoch_loss = 0
    steps = 0

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=desc, leave=False)
    for i, (inputs, targets) in pbar:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs).to(device)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        steps += 1
        pbar.set_postfix(loss=loss.item())

    return epoch_loss / steps

@torch.no_grad()
def validate_one_epoch(model, dataloader, criterion, device, desc="Val"):
    model.eval()
    epoch_loss = 0
    steps = 0

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=desc, leave=False)
    for i, (inputs, targets) in pbar:
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs).to(device)
        loss = criterion(outputs, targets)

        epoch_loss += loss.item()
        steps += 1
        pbar.set_postfix(val_loss=loss.item())

    return epoch_loss / steps



def test_anomaly_step(self, n_std=3):
    self.model.eval()
    all_errors = []

    with torch.no_grad():
        for batch in tqdm(self.testloader, desc="Testing for Anomalies"):
            x = batch[0].to(self.device)
            recon = self.model(x)
            error = torch.abs(recon - x)  # [batch_size, channels, sequence_length]
            all_errors.append(error.cpu())

    # Stack all errors: shape -> [num_samples, channels, seq_len]
    all_errors = torch.cat(all_errors, dim=0)  # total_samples x C x L

    # Mean over sequence dimension: [num_samples, channels]
    mean_errors = all_errors.mean(dim=2)  # (batch, channels)

    # Get channel-wise error stats
    channel_errors = mean_errors.numpy()  # shape: [num_samples, channels]
    channel_means = channel_errors.mean(axis=0)  # shape: [channels]
    channel_stds = channel_errors.std(axis=0)    # shape: [channels]

    # Compute anomaly threshold per channel
    thresholds = channel_means + n_std * channel_stds

    # Detect anomalies: [samples, channels] > thresholds
    anomalies = (channel_errors > thresholds).astype(int)  # 1 if anomaly, 0 otherwise

    # Optionally, compute anomaly count per channel
    anomaly_counts = anomalies.sum(axis=0)

    return {
        "channel_thresholds": thresholds,
        "anomaly_counts": anomaly_counts,
        "anomaly_mask": anomalies,
        "channel_means": channel_means,
        "channel_stds": channel_stds,
    }
