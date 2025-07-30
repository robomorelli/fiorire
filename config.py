import os
from ray import tune
import torch.nn as nn

root = os.getcwd()
model_results_path = root + '/model_results/'

if 'fiorire' in root:
    print('paths should be fine')
else:
    root = os.path.join(root ,'artificial_intelligence/repos/fiorire')

paths_to_exclude = ['data', 'dataset', 'models', 'notebook', 'preprocessing', 'test', 'train_configurations',
                    'trainers', 'utils']
root_parts = root.split('/')
root = [x if x not in paths_to_exclude else '' for x in root_parts]
root = '/'.join(root)

config_path = root + '/train_configurations/'
ray_mapper = {'tune.choice': tune.choice}

vae_config_file = 'vae.yaml'
ae_config_file = 'ae.yaml'
lstm_ae_config_file = 'lstm_ae.yaml'
lstm_config_file = 'lstm.yaml'
conv_ae_config_file = 'conv_ae.yaml'
conv_ae_1D_config_file = 'conv_ae1D.yaml'
lstm_vae_config_file = 'lstm_vae.yaml'
cnn3d_config_file = 'cnn3d.yaml'

# sentinel
sentinel_path = root + '/data/fiorire/sentinel/'

all_feats_dict = {'fiorire': ['RW1_motcurr', 'RW2_motcurr', 'RW3_motcurr', 'RW4_motcurr',
                  'RW1_therm', 'RW2_therm', 'RW3_therm', 'RW4_therm',
                  'RW1_speed', 'RW2_speed', 'RW3_speed', 'RW4_speed',
                  'RW1_cmd_volt', 'RW2_cmd_volt', 'RW3_cmd_volt', 'RW4_cmd_volt']}

activation_dict = {'Relu': nn.ReLU(), 'Elu': nn.ELU(), 'Selu': nn.SELU(), 'LRelu': nn.LeakyReLU()}
