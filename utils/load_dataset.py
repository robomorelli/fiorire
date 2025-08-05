from preprocessing.sentinel_preprocessing import get_scaled_train_val_dataloader, get_scaled_dataloader
from dataset.sentinel import Dataset_seq
import torch
from torchvision.transforms import transforms as T
from torchvision.transforms import Lambda
from torch.utils.data import DataLoader, ConcatDataset
import pandas as pd

from config import *


def get_transform(cfg):
    # Define the dataset name to apply specific transformations
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
    else:
        raise ValueError(f"Unsupported dataset name: {cfg.dataset.name}")

    return transform


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


def get_train_val_dataloader(cfg, filter_anomalies=True, **kwargs):
    """
    Get the dataset.
    :param cfg:  configuration file
    :param transform: transform to be applied to the dataset
    :return: dataset train, dataset test
    """
    transform = get_transform(cfg)

    # If seq_out_length is not specified, set it equal to seq_in_length
    if not cfg.dataset.seq_out_length:
        cfg.dataset.seq_out_length = cfg.dataset.seq_in_length

    # Load the dataframe from the specified path
    df = load_dataframe(cfg.dataset.data_path)
    # get train and validation (and eventually metrics (anomalous+normal) sampler)
    # TO DO: the normal part of metric dataloader should be a subset of the validation set but the threshold shouls
    # be defined from a separate normal sample (train?)
    trainloader, valloader, metric_loader, scaler, scaler_params = get_scaled_train_val_dataloader(cfg, df,
                                                                                                   seq_len=cfg.dataset.seq_in_length,
                                                                                                   filter_anomalies=filter_anomalies,
                                                                                                   transform=transform,
                                                                                                   ano_col=cfg.dataset.is_anomaly_column)

    n_features = len(cfg.dataset.feats)
    cfg.dataset.n_features = n_features  # Needed to specify the input channel of the model
    cfg.model.output_size = n_features if not cfg.dataset.target else len(
        cfg.dataset.target)  # Needed to specify the output channel of the model

    if cfg.dataset.save_dataloaders:
        torch.save(trainloader, os.path.join(root, 'dataloader/train_dataloader_{}_ft_{}_length.pth'.format(
            n_features, cfg.dataset.seq_in_length)))
        torch.save(valloader, os.path.join(root, 'dataloader/val_dataloader_{}_ft_{}_length.pth'.format(
            n_features, cfg.dataset.seq_in_length)))
        torch.save(valloader, os.path.join(root, 'dataloader/metric_dataloader_{}_ft_{}_length.pth'.format(
            n_features, cfg.dataset.seq_in_length)))

    return trainloader, valloader, metric_loader, scaler, scaler_params


def get_metric_loader(cfg, metric_loader=None, data_path=None, scale=True, scaler=None):
    """
    Get the metrics loader.
    :param cfg: configuration file
    :param metrics_loader: optional, if None it will be loaded from the path specified in the config file
    :return: metrics loader
    """
    metric_datasets_list = []
    # existing metric_loader (e.g., from get_train_val_dataset)
    if metric_loader is not None:
        metric_datasets_list.append(metric_loader.dataset)

    # Define the dataset name to apply specific transformations
    transform = get_transform(cfg)

    # If seq_out_length is not specified, set it equal to seq_in_length
    if not cfg.dataset.seq_out_length:
        cfg.dataset.seq_out_length = cfg.dataset.seq_in_length

    # Load the dataframe from the specified path
    metric_df = load_dataframe(data_path)

    metric_loader, scaler, scaler_params = get_scaled_dataloader(cfg, metric_df,
                                        seq_len=cfg.dataset.seq_in_length,
                                        transform=transform,
                                        scale=scale,
                                        scaler=scaler,
                                        ano_col=cfg.dataset.is_anomaly_column)

    if metric_loader is not None:
        metric_datasets_list.append(metric_loader.dataset)

    # concatenate if we have at least one dataset
    if metric_datasets_list:
        metric_dataset = ConcatDataset(metric_datasets_list)
        #df_merge = pd.concat([ds.df for ds in metric_datasets_list], ignore_index=True)
        #dataset_args = dict(df=df_merge, target=cfg.dataset.target,
        #                    sequence_length=cfg.dataset.seq_in_length, out_window=cfg.dataset.seq_out_length,
        #                    forecast=cfg.dataset.forecast, transform=transform)
        #metrics_dataset = Dataset_seq(**dataset_args)
        metric_loader = DataLoader(metric_dataset, batch_size=cfg.opt.batch_size,
                                    shuffle=False)
    else:
        metric_loader = None

    return metric_loader
