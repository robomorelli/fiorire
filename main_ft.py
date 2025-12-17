import argparse
import os
import shutil
import ray
import torch
from ray.tune.schedulers import ASHAScheduler
from utils.load_trainer import get_trainer
from utils.general import (extract_config, extract_fixed_config, get_sync_config,
                           merge_pretraining_finetuning_configs, trial_dirname_creator, get_finetuning_local_dir)
from datetime import datetime
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.tune import CLIReporter
from trainer.utils import get_opt_metric
from omegaconf import OmegaConf

from config import *


def main(args):

    # Get date to name the results folder
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d_%H-%M-%S")

    # Set the path to the configuration file
    cfg_path_ft = os.path.join(config_path, args.config_file + '.yaml')
    ray_config_ft, cfg_ft = extract_config(cfg_path_ft, fine_tuning=True)

    assert cfg_ft.opt.get('fine_tuning') and cfg_ft.model.get('checkpoint_path')

    loaded_cfg = torch.load(cfg_ft.model.checkpoint_path)['cfg']
    _, cfg_pre = extract_config(cfg_path=None, cfg=loaded_cfg)
    cfg = merge_pretraining_finetuning_configs(pretraining_cfg=cfg_pre, finetuning_cfg=cfg_ft)

    # =====================================================
    # FREEZE CONFIG TO PREVENT CHANGES DURING EXECUTION
    # =====================================================
    print("\n" + "=" * 80)
    print("📌 FREEZING CONFIG")
    print("=" * 80)

    cfg_frozen = OmegaConf.to_container(cfg, resolve=True)
    cfg = OmegaConf.create(cfg_frozen)

    # =====================================================
    # CREATE LOCAL DIR AND COPY YAML (snapshot)
    # =====================================================
    local_dir, pretrained_trial_id = get_finetuning_local_dir(cfg_ft.model.checkpoint_path, date_str)
    os.makedirs(local_dir, exist_ok=True)
    shutil.copy(cfg_path_ft, os.path.join(local_dir, f'cfg_{args.config_file}.yaml'))
    print(f"\n✅ YAML snapshot copied to experiment folder: {local_dir}")

    print(f"\nFine-tuning from trial: {pretrained_trial_id}")
    print(f"Saving to: {local_dir}\n")

    # Debug mode: simulate one training iteration
    if args.debug_mode:
        ray_config, cfg = extract_fixed_config(cfg_path=None, cfg=cfg)
        trainer_test = get_trainer(cfg.model.name)(config=ray_config)
        result = trainer_test.step()
        print("Debug mode training result:", result)
        return

    # Ray Tune configuration
    ray_config, cfg = extract_config(cfg_path=None, cfg=cfg)
    trainer = get_trainer(cfg.model.name)

    callbacks = []
    if args.wandb:
        callbacks = [WandbLoggerCallback(
            project=args.project_name,
            entity=args.entity,
            log_config=True,
            api_key=args.wandb_key,
            upload_checkpoints=True
        )]

    resources_per_trial = {"cpu": cfg.resources.cpu_trial}
    if cfg.resources.gpu_trial != 0:
        resources_per_trial["gpu"] = cfg.resources.gpu_trial

    metric_loader_path = cfg.opt.metrics_dataset_path
    metrics_dict = get_opt_metric(cfg=cfg, metrics_loader=metric_loader_path)
    metric, mode = metrics_dict['metric_key'], metrics_dict['mode']

    progress_reporter = CLIReporter(
        metric_columns=[metric, f'best_{metric}'] + list(cfg.opt.metrics_to_report) + list(cfg.opt.other_reports)
    )
    sched = ASHAScheduler(metric=metric, mode=mode, max_t=10**18, grace_period=50)
    sync_config = get_sync_config()

    analysis = ray.tune.run(
        trainer,
        scheduler=sched,
        resources_per_trial=resources_per_trial,
        num_samples=int(args.num_samples),
        local_dir=local_dir,
        name=cfg.opt.exp_name,
        progress_reporter=progress_reporter,
        sync_config=sync_config,
        config=ray_config,
        callbacks=callbacks,
        checkpoint_at_end=False,
        checkpoint_freq=0,
        keep_checkpoints_num=1,
        trial_dirname_creator=lambda trial: trial_dirname_creator(trial, max_params=5),
        stop={"training_iteration": cfg.opt.max_epochs},
    )

    print("Best config is:", analysis.get_best_config(metric="val_loss", mode="min"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", default='10.141.1.28:6379', help="address of master")
    parser.add_argument("--password", help="password to connect to master")
    parser.add_argument("--config_file", default='conv_ae2D_ft_MGM')
    parser.add_argument("--num_samples", default=100)
    parser.add_argument("--wandb", default=0, type=int)
    parser.add_argument("--project_name", default='hpo_full_2D_3anomalies_delta8_ft_fast_shot')
    parser.add_argument("--entity", default='robmorelli')
    parser.add_argument("--wandb_key", default="56b6f7f0b13c4d89207e51c28ceb90c24201eab5")
    parser.add_argument("--debug_mode", default=0, type=int)
    args = parser.parse_args()

    os.environ['TUNE_MAX_PENDING_TRIALS_PG'] = "12"

    ray.init(address='auto')
    main(args)
