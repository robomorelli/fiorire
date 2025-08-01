from torch.utils.data import Dataset
import torch

class Dataset_seq(Dataset):

    def __init__(self, df, target=None, sequence_length=16, out_window=16,  is_anomaly_column=None,
                 reconstruction=True, forecast=False, remove_target=False, transform=None):

        self.forecast = forecast
        self.reconstruction = True if not forecast else reconstruction
        self.remove_target = remove_target
        self.is_anomaly_column = is_anomaly_column

        if not (self.forecast or self.reconstruction):
            raise Exception('You should define at least one of the modes: reconstruction, prediction or forecasting')

        self.transform = transform
        if self.forecast:
            self.df_data = df.drop(columns=target) if self.remove_target else df  # In case of forecasting, the target is not removed from the input data
            self.targets = df[target] if target is not None else df  # In case of forecasting, the target is the same as the input data
        elif self.reconstruction: # In case of recontruction
            self.df_data = df.drop(columns=target) if self.remove_target else df
            self.targets = df[target] if target is not None else df  # In case of recontruction

        self.sequence_length = sequence_length
        self.out_window = out_window

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

        if self.transform is not None:
            data = self.transform(data)
            target = self.transform(target)

        if self.is_anomaly_column is not None:
            anomaly_data = data[:, self.is_anomaly_column]
            return torch.tensor(data).float(), torch.tensor(target).float(), torch.tensor(anomaly_data).float()
        else:
            return torch.tensor(data).float(), torch.tensor(target).float()
