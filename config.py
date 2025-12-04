import os

from ray import tune
import torch.nn as nn

module_dir = os.path.dirname(os.path.abspath(__file__))  # path to the current Python file
root = os.path.abspath(os.path.join(module_dir))   # go up one level (repo root)

model_results_path = root + '/model_results/'

config_path = root + '/train_configurations/'
dataset_config_path = root + '/data/dataset_configurations/'
ray_mapper = {'tune.choice': tune.choice}

lstm_ae_config_file = 'lstm_ae.yaml'
lstm_config_file = 'lstm.yaml'
conv_ae_config_file = 'conv_ae.yaml'
conv_ae_1D_config_file = 'conv_ae1D.yaml'
conv_ae_2D_config_file = 'conv_ae2D.yaml'

lstm_ae_ft_config_file = 'lstm_ae_ft.yaml'
lstm_ft_config_file = 'lstm_ft.yaml'
conv_ae_ft_config_file = 'conv_ae_ft.yaml'
conv_ae_1D_ft_config_file = 'conv_ae1D_ft.yaml'
conv_ae_2D_ft_config_file = 'conv_ae2D_ft.yaml'


available_metrics = ["val_loss", "val_f1_score", "val_roc_auc", "val_fpr", "val_tpr"]
available_modes = ["min", "max"]
opt_metric_dict_keys = ["metric_key", "mode", "best_metric"]

DEFAULT_METRIC_MODES = {
    "loss": "min",
    "f1_score": "max",
    "precision": "max",
    "recall": "max",
    "accuracy": "max",
    "roc_auc": "max",
    "fpr": "min",
    "tpr": "max",
    "error": "min",
    "mae": "min",
    "mse": "min",
}

PREFERRED_METRICS = ["val_loss", "val_roc_auc", "val_f1_score", "val_tpr", "val_fpr"]

all_feats_dict = {'fiorire_1': ['RW1_motcurr', 'RW2_motcurr', 'RW3_motcurr', 'RW4_motcurr',
                  'RW1_therm', 'RW2_therm', 'RW3_therm', 'RW4_therm',
                  'RW1_speed', 'RW2_speed', 'RW3_speed', 'RW4_speed',
                  'RW1_cmd_volt', 'RW2_cmd_volt', 'RW3_cmd_volt', 'RW4_cmd_volt']}

activation_dict = {'Relu': nn.ReLU(inplace=True), 'Elu': nn.ELU(inplace=True), 'Selu': nn.SELU(), 'LRelu': nn.LeakyReLU(), 'Tanh': nn.Tanh(), 'Sigmoid': nn.Sigmoid()}

fiorire_family = ['fiorire_1', 'fiorire_2', 'fiorire_1']
fiorire_1_data_path = root + '/data/fiorire_1/fiorire_1/all_2016-2018_clean_4s.pkl'
fiorire_2_data_path = root + '/data/fiorire_1/fiorire_2/nominal_merged.csv'

fiorire_1_conf_path = os.path.join(dataset_config_path, 'fiorire_1.yaml')
fiorire_2_conf_path = os.path.join(dataset_config_path, 'fiorire_2.yaml')

enabled_comma_names = ['opt.checkpoint_path']

anomalies_labels = [
    'GWN', 'Constant', 'Step', 'Impulse',
    'GNN',
    'TimeWarping', 'SpectralAlteration', 'PrincipalSubspaceAlteration',
    'MixingGWN', 'MixingConstant',
    'Clipping', 'Dead-Zone'
]