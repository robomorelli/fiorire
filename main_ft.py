import argparse
import os
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

    # =====================================================
    # LOAD FINE-TUNING CONFIG
    # =====================================================
    cfg_path_ft = os.path.join(config_path, args.config_file + '.yaml')
    ray_config_ft, cfg_ft = extract_config(cfg_path_ft, fine_tuning=True)

    assert cfg_ft.opt.get('fine_tuning') and cfg_ft.opt.get('checkpoint_path'), \
        "Fine-tuning mode requires 'fine_tuning=True' and 'checkpoint_path' in config"

    # =====================================================
    # HANDLE MULTIPLE CHECKPOINT PATHS
    # =====================================================
    checkpoint_paths = cfg_ft.opt.checkpoint_path

    if isinstance(checkpoint_paths, list):
        # Multiple checkpoints available
        first_checkpoint = checkpoint_paths[0]
        print(f"\n📦 Found {len(checkpoint_paths)} checkpoint(s)")
        print(f"📌 Using first checkpoint for config loading: {first_checkpoint}")
        for i, path in enumerate(checkpoint_paths):
            print(f"   [{i}] {path}")
    else:
        # Single checkpoint
        first_checkpoint = checkpoint_paths
        print(f"\n📦 Single checkpoint: {first_checkpoint}")

    # =====================================================
    # LOAD PRE-TRAINING CONFIG FROM CHECKPOINT
    # =====================================================
    print(f"\n📂 Loading pre-training config from checkpoint...")
    loaded_cfg = torch.load(first_checkpoint)['cfg']
    _, cfg_pre = extract_config(cfg_path=None, cfg=loaded_cfg)

    # =====================================================
    # MERGE PRE-TRAINING AND FINE-TUNING CONFIGS
    # =====================================================
    print(f"🔀 Merging pre-training and fine-tuning configs...")
    cfg = merge_pretraining_finetuning_configs(pretraining_cfg=cfg_pre, finetuning_cfg=cfg_ft)

    # =====================================================
    # FREEZE CONFIG TO PREVENT CHANGES DURING EXECUTION
    # =====================================================
    print("\n" + "=" * 80)
    print("📌 FREEZING CONFIG")
    print("=" * 80)

    # Convert OmegaConf → pure dict (no more file references!)
    cfg_frozen = OmegaConf.to_container(cfg, resolve=True)
    cfg = OmegaConf.create(cfg_frozen)

    # Save frozen config snapshot (optional, for debugging)
    frozen_config_path = os.path.join('/tmp', f'frozen_config_ft_{args.project_name}_{date_str}.yaml')
    OmegaConf.save(cfg, frozen_config_path)
    print(f"✅ Config frozen and saved to: {frozen_config_path}")

    # Debug: show sample parameters
    print(f"Sample params from frozen config:")
    print(f"  - opt.lr: {cfg.opt.get('lr', 'N/A')}")
    print(f"  - opt.fine_tuning: {cfg.opt.get('fine_tuning', 'N/A')}")
    print(f"  - opt.fine_tuning_mode: {cfg.opt.get('fine_tuning_mode', 'N/A')}")
    print(f"  - opt.freeze_layers: {cfg.opt.get('freeze_layers', 'N/A')}")
    print(f"  - model.name: {cfg.model.get('name', 'N/A')}")
    print("=" * 80 + "\n")

    # =====================================================
    # DEBUG MODE
    # =====================================================
    if args.debug_mode:
        print("🐛 DEBUG MODE: Running single trial")
        ray_config, cfg = extract_fixed_config(cfg_path=None, cfg=cfg)
        trainer_test = get_trainer(cfg.model.name)(config=ray_config)
        result = trainer_test.step()
        print("Debug mode training result:", result)
        return

    # =====================================================
    # SETUP OUTPUT DIRECTORY
    # =====================================================
    local_dir, pretrained_trial_id = get_finetuning_local_dir(first_checkpoint, date_str)
    print(f"\n📁 Fine-tuning from trial: {pretrained_trial_id}")
    print(f"📁 Results will be saved to: {local_dir}\n")

    # =====================================================
    # EXTRACT RAY CONFIG FROM FROZEN CONFIG
    # =====================================================
    ray_config, cfg = extract_config(cfg_path=None, cfg=cfg)

    # Verify ray_config is pure dict
    if not isinstance(ray_config, dict):
        print("⚠️ WARNING: ray_config is not a pure dict! Converting...")
        ray_config = dict(ray_config)

    print(f"Ray config type: {type(ray_config)}")

    # =====================================================
    # SETUP RAY TUNE
    # =====================================================
    trainer = get_trainer(cfg.model.name)

    # Callbacks
    if args.wandb:
        callbacks = [WandbLoggerCallback(
            project=args.project_name,
            entity=args.entity,
            log_config=True,
            api_key=args.wandb_key,
            upload_checkpoints=True
        )]
    else:
        callbacks = []

    # Resources per trial
    resources_per_trial = {
        "cpu": cfg.resources.cpu_trial,
        "gpu": cfg.resources.gpu_trial
    } if cfg.resources.gpu_trial != 0 else {"cpu": cfg.resources.cpu_trial}

    # Metrics
    metric_loader_path = cfg.opt.metrics_dataset_path
    metrics_dict = get_opt_metric(cfg=cfg, metrics_loader=metric_loader_path)
    metric, mode = metrics_dict['metric_key'], metrics_dict['mode']

    # Progress reporter
    progress_reporter = CLIReporter(
        metric_columns=[metric, f'best_{metric}'] +
                       list(cfg.opt.metrics_to_report) +
                       list(cfg.opt.other_reports)
    )

    # Scheduler
    sched = ASHAScheduler(metric=metric, mode=mode, max_t=10 ** 18, grace_period=50)
    sync_config = get_sync_config()

    # =====================================================
    # RUN RAY TUNE (with FROZEN config)
    # =====================================================
    print("\n🚀 Starting Ray Tune with FROZEN config...")

    analysis = tune.run(
        trainer,
        scheduler=sched,
        resources_per_trial=resources_per_trial,
        num_samples=int(args.num_samples),
        local_dir=local_dir,
        name="{}".format(cfg.opt.exp_name),
        progress_reporter=progress_reporter,
        sync_config=sync_config,
        config=ray_config,  # ← FROZEN config (pure dict)
        callbacks=callbacks,
        checkpoint_at_end=False,
        checkpoint_freq=0,
        keep_checkpoints_num=1,
        trial_dirname_creator=lambda trial: trial_dirname_creator(trial, max_params=5),
        stop={"training_iteration": cfg.opt.max_epochs},
    )

    print("\n✅ Ray Tune completed!")
    print("Best config is:", analysis.get_best_config(metric="val_loss", mode="min"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", default='10.141.1.28:6379', help="address of master")
    parser.add_argument("--password", help="password to connect to master")
    parser.add_argument("--config_file", default='conv_ae2D_ft', help="[conv_ae1D_ft, conv_ae2D_ft, lstm_ft]")
    parser.add_argument("--num_samples", default=100, help="number of trials")
    parser.add_argument("--wandb", default=0, type=int, help="use wandb logging")
    parser.add_argument("--project_name", default='hpo_full_2D_3anomalies_delta8_ft_fast_shot', help="project name")
    parser.add_argument("--entity", default='robmorelli', help="wandb entity")
    parser.add_argument("--wandb_key", default="56b6f7f0b13c4d89207e51c28ceb90c24201eab5", help="wandb API key")
    parser.add_argument("--debug_mode", default=1, type=int, help="debug mode (0 or 1)")
    args = parser.parse_args()

    os.environ['TUNE_MAX_PENDING_TRIALS_PG'] = "12"

    # Initialize Ray
    ray.init(address='auto')

    # Run main
    main(args)