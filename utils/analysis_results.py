import pandas as pd
import argparse
import os
from os.path import expanduser
from omegaconf import OmegaConf
from pathlib import Path
from config import *

def main(args):
    config_file = args.experiment_name.split('/')[0]
    config_path_rebase = Path(config_path).parents[1].as_posix()
    cfg = OmegaConf.load(os.path.join(config_path_rebase,'train_configurations', config_file + '.yaml'))
    folder = os.path.join(args.experiment_path, args.experiment_name)
    exp_names = os.listdir(folder)
    metrics = list(cfg.opt.metrics)
    losses = [x for x in metrics if 'loss' in x]

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
    parser.add_argument("--experiment_name", default='cnn3d/10-12-22:10:53:11', help="the model you want to hpo")
    args = parser.parse_args()

    main(args)
