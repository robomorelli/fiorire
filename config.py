import os

from ray import tune
import torch.nn as nn

module_dir = os.path.dirname(os.path.abspath(__file__))  # path to the current Python file
root = os.path.abspath(os.path.join(module_dir))   # go up one level (repo root)

model_results_path = root + '/model_results/'

config_path = root + '/train_configurations/'
ray_mapper = {'tune.choice': tune.choice}

lstm_ae_config_file = 'lstm_ae.yaml'
lstm_config_file = 'lstm.yaml'
conv_ae_config_file = 'conv_ae.yaml'
conv_ae_1D_config_file = 'conv_ae1D.yaml'
conv_ae_2D_config_file = 'conv_ae2D.yaml'

available_metrics = ['val_loss', 'val_f1_score']
available_modes = ['min', 'max']
opt_metric_dict_keys = ["metric_key", "mode", "best_metric"]

# sentinel
sentinel_path = root + '/data/fiorire/sentinel/'

all_feats_dict = {'fiorire': ['RW1_motcurr', 'RW2_motcurr', 'RW3_motcurr', 'RW4_motcurr',
                  'RW1_therm', 'RW2_therm', 'RW3_therm', 'RW4_therm',
                  'RW1_speed', 'RW2_speed', 'RW3_speed', 'RW4_speed',
                  'RW1_cmd_volt', 'RW2_cmd_volt', 'RW3_cmd_volt', 'RW4_cmd_volt']}

activation_dict = {'Relu': nn.ReLU(), 'Elu': nn.ELU(), 'Selu': nn.SELU(), 'LRelu': nn.LeakyReLU(), 'Tanh': nn.Tanh(), 'Sigmoid': nn.Sigmoid()}

fiorire_family = ['fiorire_1', 'fiorire_2', 'fiorire']