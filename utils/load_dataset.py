
from preprocessing.sentinel_preprocessing import prep_sentinel
from dataset.sentinel import Dataset_seq
import torch
from torchvision.transforms import transforms as T
from torchvision.transforms import Lambda
from torch.utils.data import DataLoader
import pandas as pd

from config import *

def get_dataset(cfg, **kwargs):
    """
    Get the dataset.
    :param cfg:  configuration file
    :param transform: transform to be applied to the dataset
    :return: dataset train, dataset test
    """
    if cfg.dataset.name == "sentinel":

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

        sample_rate = cfg.dataset.sample_rate
        feats = cfg.dataset.feats
        clean = cfg.dataset.clean
        scaled = cfg.dataset.scaled
        columns_subset = cfg.dataset.columns_subset
        dataset_subset = cfg.dataset.dataset_subset
        train_val_split = cfg.dataset.train_val_split
        forecast_all = cfg.dataset.forecast_all
        sampling_rate = cfg.dataset.sample_rate
        scale = cfg.dataset.scale
        random_split = cfg.dataset.random_split
        shuffle_train = cfg.dataset.shuffle_train
        perc_overlap = cfg.dataset.perc_overlap

        print('DATASET length ', cfg.dataset.dataset_subset)

        sequence_length = kwargs['sequence_length']

        if scaled:
            dataset_name = 'dataset_4s/all_2016-2018_clean_std_4s.pkl'
        else:
            dataset_name = 'dataset_4s/all_2016-2018_clean_4s.pkl'

        data_path = os.path.join(sentinel_path, dataset_name)
        df = pd.read_pickle(data_path)

        '''
        df_train, df_test, df = prep_sentinel(df, cfg, cfg.dataset.columns, columns_subset=cfg.dataset.columns_subset,
                                        dataset_subset=cfg.dataset.dataset_subset, train_val_split=train_val_split,
                                        scale=scale)
        
        train_dataset = Dataset_seq(df_train, target=cfg.dataset.target, sequence_length=cfg.dataset.sequence_length,
                                    out_window=cfg.dataset.out_window, prediction=False,
                                    forecast_all = cfg.dataset.forecast_all, transform=transform)
        trainloader = DataLoader(dataset=train_dataset, batch_size=kwargs['batch_size'], shuffle=True)

        test_dataset = Dataset_seq(df_test, target=cfg.dataset.target, sequence_length=cfg.dataset.sequence_length,
                                    out_window=cfg.dataset.out_window, prediction=False,
                                   forecast_all = cfg.dataset.forecast_all, transform=transform)
        valloader = DataLoader(dataset=test_dataset, batch_size=kwargs['batch_size'], shuffle=False)
        '''

        train_sampler, val_sampler, df = prep_sentinel(df, cfg, cfg.dataset.columns, columns_subset=cfg.dataset.columns_subset,
                                        dataset_subset=cfg.dataset.dataset_subset, train_val_split=train_val_split,
                                         scale=scale, perc_overlap = perc_overlap, random_split=random_split,
                                                       shuffle_train=shuffle_train)

        n_features = len(df.columns)

        # Dataset for dataloader definition
        train_dataset = Dataset_seq(df, target=cfg.dataset.target, sequence_length=cfg.dataset.sequence_length,
                                    out_window=cfg.dataset.out_window, prediction=False,
                                    forecast_all = cfg.dataset.forecast_all, transform=transform)
        trainloader = DataLoader(dataset=train_dataset, batch_size=kwargs['batch_size']
                                 ,sampler=train_sampler)#, shuffle=True)
        test_dataset = Dataset_seq(df, target=cfg.dataset.target, sequence_length=cfg.dataset.sequence_length,
                                    out_window=cfg.dataset.out_window, prediction=False,
                                   forecast_all = cfg.dataset.forecast_all, transform=transform)
        valloader = DataLoader(dataset=test_dataset, batch_size=kwargs['batch_size'], sampler=val_sampler)
                                #, shuffle=False)


        if 'conv' not in cfg.model.name:
            if scaled:
                if not shuffle_train:
                    torch.save(trainloader, os.path.join(root,'dataloader/train_dataloader_{}_ft_{}_{}.pth'.format(n_features,
                                                                                                     sampling_rate,
                                                                                                     sequence_length)))
                    torch.save(valloader, os.path.join(root,'dataloader/test_dataloader_{}_ft_{}_{}.pth'.format(n_features,
                                                                                                   sampling_rate,
                                                                                                   sequence_length)))
                else:
                    torch.save(trainloader,
                        os.path.join(root,'dataloader/train_dataloader_{}_ft_{}_{}_shuffle.pth'.format(n_features,
                                                                                                 sampling_rate,
                                                                                                 sequence_length)))
                    torch.save(valloader,
                        os.path.join(root,'dataloader/test_dataloader_{}_ft_{}_{}_shuffle.pth'.format(n_features,
                                                                                                sampling_rate,
                                                                                                sequence_length)))
            else:
                if not shuffle_train:
                    torch.save(trainloader,
                        os.path.join(root,'dataloader/train_dataloader_not_scaled_{}_ft_{}_{}.pth'.format(n_features,
                                                                                                    sampling_rate,
                                                                                                    sequence_length)))
                    torch.save(valloader,
                        os.path.join(root,'dataloader/test_dataloader_not_scaled_{}_ft_{}_{}.pth'.format(n_features,
                                                                                                   sampling_rate,
                                                                                                   sequence_length)))
                else:
                    torch.save(trainloader, os.path.join(root,'dataloader/train_dataloader_not_scaled_{}_ft_{}_{}_shuffle.pth'.format(
                        n_features, sampling_rate, sequence_length)))
                    torch.save(valloader, os.path.join(root,'dataloader/test_dataloader_not_scaled_{}_ft_{}_{}_shuffle.pth'.format(
                        n_features, sampling_rate, sequence_length)))

        return trainloader, valloader, n_features, scaled, scale, columns_subset, dataset_subset, train_val_split, dataset_name, data_path


