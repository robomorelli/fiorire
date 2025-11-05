from preprocessing.sentinel_preprocessing import get_scaled_train_val_dataloader, get_scaled_dataloader
from torch.utils.data import Sampler
import torch
from torchvision.transforms import transforms as T
from torchvision.transforms import Lambda
from torch.utils.data import DataLoader, ConcatDataset, SubsetRandomSampler, SequentialSampler
import pandas as pd
import csv

from config import *


class ConcatSampler(Sampler):
    def __init__(self, samplers, dataset_lengths):
        """
        :param samplers: list of individual samplers for each dataset
        :param dataset_lengths: list of lengths of each dataset in the same order
        """
        self.samplers = samplers
        self.dataset_lengths = dataset_lengths
        self.index_offsets = self._compute_offsets()

    def _compute_offsets(self):
        offsets = [0]
        for length in self.dataset_lengths[:-1]:
            offsets.append(offsets[-1] + length)
        return offsets

    def __iter__(self):
        for offset, sampler in zip(self.index_offsets, self.samplers):
            for idx in sampler:
                yield offset + idx  # Shift index according to dataset offset

    def __len__(self):
        return sum(len(s) for s in self.samplers)


def get_transform(cfg):
    # Define the dataset name to apply specific transformations
    if cfg.dataset.name in fiorire_family:
        if cfg.model.name == "conv_ae1D":
            transform = T.Compose([
                T.ToTensor(),
                Lambda(lambda x: x.permute((0, 2, 1))),
                Lambda(lambda x: x.squeeze(0))])
        elif cfg.model.name == 'conv_ae2D':
            transform = T.Compose([
                T.ToTensor(),
                Lambda(lambda x: x.permute((0, 2, 1)))])
        else:
            transform = T.Compose([
                T.ToTensor(),
                Lambda(lambda x: x.squeeze(0))  # Removes channel dim added by ToTensor
            ])
    else:
        raise ValueError(f"Unsupported dataset name: {cfg.dataset.name}")

    return transform


def load_dataframe(cfg):
    """
    Load a pandas DataFrame from CSV, TSV, Excel, Parquet, Pickle, or text-like files.
    Automatically detects file type and delimiter for text files. Validates expected columns if provided.

    Parameters:
        file_path (str): Path to the data file.
        expected_cols (list of str, optional): List of expected column names. Used to validate delimiter detection.

    Returns:
        pd.DataFrame: Loaded DataFrame.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the file format is unsupported or cannot be parsed.
    """
    file_path = cfg.dataset.data_path
    expected_cols = cfg.dataset.feats

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()

    def detect_delimiter(path):
        """Auto-detect delimiter from first line of a text file."""
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            sample = f.read(2048)
        try:
            return csv.Sniffer().sniff(sample).delimiter
        except csv.Error:
            for d in [",", "\t", ";", "|"]:
                if d in sample:
                    return d
            return ","  # default

    def try_read_csv(path):
        """Try multiple delimiters and validate expected columns."""
        delimiters = [",", "\t", ";", "|"]
        for d in delimiters:
            try:
                df = pd.read_csv(path, delimiter=d)
                if expected_cols is None or all(col in df.columns for col in expected_cols):
                    return df  # valid delimiter
            except Exception:
                continue
        # last resort: read with auto-detected delimiter
        d = detect_delimiter(path)
        df = pd.read_csv(path, delimiter=d)
        return df

    try:
        # CSV / text-like files
        if ext in [".csv", ".txt", ".dat"] or ext not in [".xlsx", ".xlsm", ".xls", ".xlsb", ".odf", ".ods", ".odt", ".pkl", ".pickle", ".parquet"]:
            return try_read_csv(file_path)

        # Pickle
        elif ext in [".pkl", ".pickle"]:
            return pd.read_pickle(file_path)

        # Parquet
        elif ext == ".parquet":
            return pd.read_parquet(file_path)

        # Excel modern
        elif ext in [".xlsx", ".xlsm"]:
            try:
                df = pd.read_excel(file_path, engine="openpyxl")
            except Exception:
                df = try_read_csv(file_path)
            return df

        # Excel old
        elif ext == ".xls":
            try:
                df = pd.read_excel(file_path, engine="xlrd")
            except Exception:
                try:
                    df = pd.read_excel(file_path, engine="openpyxl")
                except Exception:
                    df = try_read_csv(file_path)
            return df

        # Excel binary
        elif ext == ".xlsb":
            return pd.read_excel(file_path, engine="pyxlsb")

        # OpenDocument formats
        elif ext in [".odf", ".ods", ".odt"]:
            return pd.read_excel(file_path, engine="odf")

        else:
            # Last resort: treat as text
            return try_read_csv(file_path)

    except Exception as e:
        raise ValueError(f"Failed to load file '{file_path}': {e}")



def get_train_val_dataloader(cfg, filter_anomalies=True,
                             **kwargs):
    """
    Load and prepare train/validation dataloaders.

    Args:
        cfg: configuration object
        filter_anomalies (bool): whether to filter anomalies
        align_data (bool): if True, align all DataFrame columns to the same length
        detect_flag (bool): if True, detect first change in flag column and trim data
        **kwargs: other optional arguments
    """
    transform = get_transform(cfg)

    # If seq_out_length is not specified, set it equal to seq_in_length
    if not cfg.dataset.seq_out_length:
        cfg.dataset.seq_out_length = cfg.dataset.seq_in_length

    print("📂 Loading dataset from:", cfg.dataset.data_path)
    df = load_dataframe(cfg)

    flag_col = getattr(cfg.dataset, "flag_col", None)
    align_data = getattr(cfg.dataset, "align_data", False)
    detect_flag = getattr(cfg.dataset, "detect_flag", False)

    # Optional alignment and flag trimming
    if align_data or detect_flag:
        col_to_rem = [c for c in df.columns if c.startswith("ANT47") or c.startswith("Frame")]
        if col_to_rem:
            print(f"🔧 Preprocessing: removing {len(col_to_rem)} ANT47 columns")
            df = df.drop(columns=[c for c in df.columns if c.startswith("ANT47") or c.startswith("Frame")])
        print("🔧 Preprocessing: align_data =", align_data, ", detect_flag =", detect_flag)
        # --- 1️⃣ Align columns if required ---
        if align_data:
            series_dict = {col: df[col] for col in df.columns}
            min_len = min(len(s) for s in series_dict.values())
            aligned_data = {col: s.iloc[:min_len].reset_index(drop=True) for col, s in series_dict.items()}
            df = pd.DataFrame(aligned_data)
            print(f"✅ Data aligned to {min_len} samples")

        # --- 2️⃣ Detect flag and trim if required ---
        if detect_flag:
            if flag_col and flag_col in df.columns:
                print(f"⚙️ Detecting first change in flag column: '{flag_col}'")
                changes = df[flag_col].diff().fillna(0)
                change_idxs = changes[changes != 0].index
                if len(change_idxs) > 0:
                    first_change_idx = change_idxs[0]
                    df = df.loc[first_change_idx:].reset_index(drop=True)
                    print(f"✅ Trimmed dataset from first flag change at index {first_change_idx}")
                else:
                    print("⚠️ No flag change detected — dataset not trimmed.")
            else:
                print("⚠️ No valid flag column found in cfg.dataset.flag_column.")

    # --- Continue with normal scaling & dataloader creation ---
    trainloader, valloader, metric_loader, scaler, scaler_params = get_scaled_train_val_dataloader(
        cfg, df,
        seq_len=cfg.dataset.seq_in_length,
        filter_anomalies=filter_anomalies,
        transform=transform,
        ano_col=cfg.dataset.is_anomaly_column
    )

    n_features = len(cfg.dataset.feats)
    cfg.dataset.n_features = n_features
    cfg.model.output_size = n_features if not cfg.dataset.target else len(cfg.dataset.target)

    # Optionally save dataloaders
    if cfg.dataset.save_dataloaders:
        root = getattr(cfg, "root", ".")
        torch.save(trainloader, os.path.join(root, f'dataloader/train_dataloader_{n_features}_ft_{cfg.dataset.seq_in_length}_length.pth'))
        torch.save(valloader, os.path.join(root, f'dataloader/val_dataloader_{n_features}_ft_{cfg.dataset.seq_in_length}_length.pth'))
        torch.save(metric_loader, os.path.join(root, f'dataloader/metric_dataloader_{n_features}_ft_{cfg.dataset.seq_in_length}_length.pth'))

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
    metric_df = load_dataframe(cfg)

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
        samplers = []
        dataset_lengths = []

        for dataset in metric_datasets_list:
            dataset_lengths.append(len(dataset))

            # Use default logic to choose a sampler (customize as needed)
            if hasattr(dataset, 'sampler_type'):
                # Custom datasets can define their own sampler preference
                sampler_type = dataset.sampler_type
            else:
                sampler_type = 'SequentialSampler'  # Default

            if sampler_type == 'SubsetRandomSampler':
                sampler = SubsetRandomSampler(dataset.indices)
            elif sampler_type == 'Subset' and hasattr(dataset, 'indices'):
                # For Dataset objects that already define a subset of indices
                sampler = SubsetRandomSampler(dataset.indices)
            else:
                sampler = SequentialSampler(dataset)

            samplers.append(sampler)

        concat_sampler = ConcatSampler(samplers=samplers, dataset_lengths=dataset_lengths)
        metric_dataset = ConcatDataset(metric_datasets_list)
        metric_loader = DataLoader(
            metric_dataset,
            batch_size=cfg.opt.batch_size,
            sampler=concat_sampler,
        )

    else:
        metric_loader = None

    return metric_loader
