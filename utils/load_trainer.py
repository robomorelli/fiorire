#!/usr/bin/env python3
from torch.utils.data import DataLoader
from trainer.vae_trainer import trainVae
from trainer.ae_trainer import trainAe
from trainer.cnn3d_trainer import trainCNN3D
from trainer.lstm_ae_trainer import trainLSTMAE
from trainer.lstm_trainer import trainLSTM
from trainer.lstm_vae_trainer import trainLSTMVAE
from trainer.conv_ae_trainer import trainCONVAE
from trainer.conv_ae1d_trainer import trainCONVAE1D

def get_trainer(cfg, **kwargs):
    """
    Get the dataset.
    :param cfg:  configuration file
    :param transform: transform to be applied to the dataset
    :return: dataset train, dataset test
    """
    if cfg.model.name == "vae":
        return trainVae
    if cfg.model.name == "ae":
        return trainAe
    if cfg.model.name == "cnn3d":
        return trainCNN3D
    if cfg.model.name == "lstm_ae":
        return trainLSTMAE
    if cfg.model.name == "lstm":
        return trainLSTM
    if cfg.model.name == "conv_ae":
        return trainCONVAE
    if cfg.model.name == "conv_ae1D":
        return trainCONVAE1D
    if cfg.model.name == "lstm_vae":
        return trainLSTMVAE
