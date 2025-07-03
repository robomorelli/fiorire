import pandas as pd
import argparse
import os
from os.path import expanduser
from omegaconf import OmegaConf
from pathlib import Path
import sys
sys.path.append('../fdir')
from config import *

def main(args):
    config_file = args.experiment_name.split('/')[0]
    config_path_rebase = root
    #cfg = OmegaConf.load(os.path.join(config_path_rebase,'train_configurations', config_file + '.yaml'))
    folder = os.path.join(args.experiment_path, args.experiment_name)
    exp_names = os.listdir(folder)
    metrics = list(cfg.opt.metrics) + ['parameters_number']

    losses = [x for x in metrics if 'loss' in x]
    losses = losses

    for l in losses:
        metrics.remove(l)

    summary_df = pd.DataFrame()
    for en in exp_names:
        try:
            df_exp = pd.read_csv(os.path.join(folder,en,"progress.csv"))
        except:
            continue
        if len(df_exp) > 0:
            dict = {'name': en}
            for m in metrics:
                dict[m] = df_exp[m].sort_values(ascending=False).iloc[0]
            for l in losses:
                dict[l] = df_exp[l].sort_values(ascending=True).iloc[0]
            summary_df = summary_df.append(dict, ignore_index=True)

    summary_df.sort_values(by=[cfg.opt.order_by], ascending=False, inplace=True)
    summary_df.to_csv(os.path.join(folder, 'summary.csv'))

if __name__ == "__main__":

    home = expanduser("~")
    base_path = os.path.join(home,'ray_results')
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_path", default=base_path, help="the model you want to hpo")
    parser.add_argument("--experiment_name", default='conv_ae1D/4_wheels_system_04-14-23:15:49:11_conv_ae1D_sl_40_12M_unscaled_unscaled_overlap_0', help="the model you want to hpo")
    args = parser.parse_args()#4_wheels_system_04-14-23:12:41:18_conv_ae1D_sl_40_1M_unscaled_unscaled_overlap_1
    main(args)


#4_wheel_system_01-24-23:18:55:37_extended_params_sl_100_run_2
#4_wheel_system_01-24-23:13:40:52_extended_params_sl_100
#third_wheel_01-24-23:11:20:34_extended_params_sl_100
#4_wheel_system_01-24-23:12:00:52_extended_params_sl_16
#4_wheel_system_01-25-23:10:09:11_conv_extended_params_sl_16
#4_wheel_system_01-26-23:10:28:37_conv_extended_params_sl_16_run_2
