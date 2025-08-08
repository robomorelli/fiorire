from omegaconf import ListConfig
from torch.utils.data import Dataset
import torch
import numpy as np

class Dataset_seq(Dataset):

    def __init__(self, df, target=None, sequence_length=16, out_window=16,  is_anomaly_column=None,
                 reconstruction=True, forecast=False, remove_target=False, transform=None, sampler=None, indices=None):

        self.df = df
        self.forecast = forecast
        self.reconstruction = True if not forecast else reconstruction
        self.remove_target = remove_target
        self.is_anomaly_column = (
            is_anomaly_column if isinstance(is_anomaly_column, (list, ListConfig))
            else [is_anomaly_column] if isinstance(is_anomaly_column, str)
            else None
        )
        self.transform = transform
        self.sequence_length = sequence_length
        self.out_window = out_window
        self.sampler_type = type(sampler).__name__
        self.indices = indices

        if not (self.forecast or self.reconstruction):
            raise Exception('You should define at least one of the modes: reconstruction, prediction or forecasting')

        # Determine columns to exclude from model input
        exclude_cols = []
        if self.remove_target and target is not None:
            exclude_cols.extend(target if isinstance(target, list) else [target])
        if self.is_anomaly_column is not None:
            exclude_cols.append(self.is_anomaly_column) if isinstance(self.is_anomaly_column, str) else exclude_cols.extend(self.is_anomaly_column)

        # Get binary classification target if is_anomaly_column is specified
        self.df_is_anomaly = self.df.loc[:, self.is_anomaly_column] if self.is_anomaly_column is not None and all(item in list(self.df.columns) for item in self.is_anomaly_column) else None

        # Drop excluded columns for model input
        self.df_data = df.drop(columns=exclude_cols) if exclude_cols else df

        # Define targets (always includes target columns, never excludes is_anomaly_column)
        self.targets = df[target] if target is not None else self.df_data

    def __len__(self):
        return len(self.df_data)

    def __getitem__(self, idx):
        if self.forecast:
            if (idx + self.sequence_length + self.out_window) > len(self.df_data):
                indexes = list(range(len(self.df_data) - self.sequence_length - self.out_window,
                                     len(self.df_data) - self.out_window))
                indexes_out = list(range(len(self.df_data) - self.out_window, len(self.df_data)))
            else:
                indexes = list(range(idx, idx + self.sequence_length))
                indexes_out = list(range(idx + self.sequence_length, idx + self.sequence_length + self.out_window))

        else:
            if (idx + self.sequence_length) > len(self.df_data):
                indexes = list(range(len(self.df_data) - self.sequence_length, len(self.df_data)))
            else:
                indexes = list(range(idx, idx + self.sequence_length))
            indexes_out = indexes[-self.out_window:]

        data = self.df_data.iloc[indexes, :].values
        target = self.targets.iloc[indexes_out, :].values
        anomaly_labels = (
            self.df_is_anomaly.iloc[indexes_out, :].values
            if self.df_is_anomaly is not None
            else np.full((len(indexes_out), 1), np.nan)
        )

        if self.transform is not None:
            data = self.transform(data)
            target = self.transform(target)
            anomaly_labels = self.transform(anomaly_labels)

        return data.float(), target.float(), anomaly_labels.float()
