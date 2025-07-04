import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
from torch.utils.data import SubsetRandomSampler
import numpy as np

def prep_sentinel(df, cfg, columns, columns_subset,
                  dataset_subset = 10000, train_val_split=0.8,
                  scale=True, perc_overlap=0, random_split=False, shuffle_train=True):

    if columns_subset:
        columns = columns[:columns_subset]
    dataRaw = df[columns].dropna()

    if dataset_subset:
        dataRaw = dataRaw.iloc[:dataset_subset, :]

    df = dataRaw.copy()

    x = df.values
    if scale:
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
        dfNorm = pd.DataFrame(x_scaled, columns=df.columns)
    else:
        print('the data is not going to be scaled')
        dfNorm = pd.DataFrame(x, columns=df.columns)

    np.random.seed(101)
    step = cfg.dataset.sequence_length - int(cfg.dataset.sequence_length * perc_overlap)
    print('step', step)
    if step == 0:
        step = 1

    dataset_size = len(dfNorm)
    idxs = np.arange(0, dataset_size, step)
    print('idxs', idxs[:10])
    print('len idxs', len(idxs))

    if random_split:
        np.random.shuffle(idxs)

    train_split_idx = int(np.floor(train_val_split * len(idxs)))
    train_idx, val_idx = idxs[:train_split_idx], idxs[train_split_idx:]

    if shuffle_train:
        print('train idx before shuffle', train_idx[:10])
        np.random.shuffle(train_idx)
        print('train idx after shuffle', train_idx[:10])

    print('train idx', train_idx[:10])
    print('val idx', val_idx[:10])

    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    return train_sampler, val_sampler, dfNorm
