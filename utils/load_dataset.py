from preprocessing.sentinel_preprocessing import get_scaled_train_val_df,  get_train_val_samplers
from dataset.sentinel import Dataset_seq
import torch
import numpy as np
from torchvision.transforms import transforms as T
from torchvision.transforms import Lambda
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

from config import *

scaler_dict = {'standard': StandardScaler(), 'robust': RobustScaler()}




def load_dataframe(file_path):
    """
    Load a pandas DataFrame from a CSV, pickle, or Parquet file based on the file extension.

    Parameters:
        file_path (str): The path to the data file.

    Returns:
        pd.DataFrame: The loaded DataFrame.

    Raises:
        ValueError: If the file extension is unsupported.
        FileNotFoundError: If the file doesn't exist.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".csv":
        return pd.read_csv(file_path)
    elif ext in [".pkl", ".pickle"]:
        return pd.read_pickle(file_path)
    elif ext == ".parquet":
        return pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

def  get_dataset(cfg, **kwargs):
    """
    Get the dataset.
    :param cfg:  configuration file
    :param transform: transform to be applied to the dataset
    :return: dataset train, dataset test
    """
    if cfg.dataset.name == "fiorire":

        if cfg.model.name == "conv_ae":
            transform = T.Compose([
                T.ToTensor(),
            ])

        elif cfg.model.name == "conv_ae1D":
            transform = T.Compose([
                T.ToTensor(),
                Lambda(lambda x: x.permute((0, 2, 1))),
                Lambda(lambda x: x.squeeze(0))])
        else:
            transform = None

        if not cfg.dataset.seq_out_length:
            cfg.dataset.seq_out_length = cfg.dataset.seq_in_length
        batch_size = cfg.dataset.batch_size

        df = load_dataframe(cfg.dataset.data_path)

        # get train and validation dataframes
        train_df, val_df, scaler, df, scaler_params = get_scaled_train_val_df(cfg, df)
        # get train and validation samplers
        train_sampler, val_sampler = get_train_val_samplers(cfg, df)
        # Dataset for dataloader definition, left the argsument other than cfg because we want to use also without config file
        train_dataset = Dataset_seq(df, target=cfg.dataset.target, sequence_length=cfg.dataset.seq_in_length,
                                    out_window=cfg.dataset.seq_out_length, forecast=cfg.dataset.forecast, transform=transform)
        trainloader = DataLoader(dataset=train_dataset, batch_size=batch_size
                                 ,sampler=train_sampler)#, shuffle=True)
        test_dataset = Dataset_seq(df, target=cfg.dataset.target, sequence_length=cfg.dataset.seq_in_length,
                                    out_window=cfg.dataset.seq_out_length, forecast=cfg.dataset.forecast, transform=transform)
        valloader = DataLoader(dataset=test_dataset, batch_size=batch_size, sampler=val_sampler)

        n_features = len(cfg.dataset.feats)

        if cfg.dataset.save_dataloaders:
            torch.save(trainloader, os.path.join(root,'dataloader/train_dataloader_{}_ft_{}_length.pth'.format(
                n_features, cfg.dataset.seq_in_length)))
            torch.save(valloader, os.path.join(root,'dataloader/test_dataloader_{}_ft_{}_length.pth'.format(
                n_features, cfg.dataset.seq_in_length)))

        return trainloader, valloader, n_features, scaler, scaler_params
