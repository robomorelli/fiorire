import argparse
import ray
import torch
from ray.tune.schedulers import ASHAScheduler
from utils.load_trainer import get_trainer
from utils.general import (extract_config, extract_fixed_config, get_sync_config, clean_loaded_cfg,
                           merge_pretraining_finetuning_configs, trial_dirname_creator, get_finetuning_local_dir)
from utils.load_dataset import prepare_shared_configuration
from datetime import datetime
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.tune import CLIReporter
from trainer.utils import get_opt_metric
from omegaconf import OmegaConf
import os

from config import *


def main(args):
    """Main fine-tuning function with shared datasets."""

    print("\n" + "=" * 80)
    print("🔍 COMMAND LINE ARGUMENTS")
    print("=" * 80)
    print(f"   - config_file: {args.config_file}")
    print(f"   - project_name: {args.project_name}")
    print(f"   - n_gpus: {args.n_gpus}")
    print(f"   - n_cpus: {args.n_cpus}")
    print(f"   - num_samples: {args.num_samples}")
    print(f"   - debug_mode: {args.debug_mode}")
    print("=" * 80)

    # Get date
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d_%H-%M-%S")

    # Load fine-tuning config
    cfg_path_ft = os.path.join(config_path, args.config_file + '.yaml')
    ray_config_ft, cfg_ft = extract_config(cfg_path_ft, fine_tuning=True)

    assert cfg_ft.opt.get('fine_tuning') and cfg_ft.model.get('checkpoint_path')

    # Load pretrained config
    loaded_cfg = torch.load(cfg_ft.model.checkpoint_path)['cfg']
    loaded_cfg = clean_loaded_cfg(loaded_cfg)
    _, cfg_pre = extract_config(cfg_path=None, cfg=loaded_cfg)

    # Merge configs
    cfg = merge_pretraining_finetuning_configs(pretraining_cfg=cfg_pre, finetuning_cfg=cfg_ft)

    # =====================================================
    # FREEZE CONFIG
    # =====================================================
    print("\n" + "=" * 80)
    print("📌 FREEZING CONFIG")
    print("=" * 80)

    #cfg_frozen = OmegaConf.to_container(cfg, resolve=True)
    #cfg = OmegaConf.create(cfg_frozen)

    #frozen_config_path = os.path.join('/tmp', f'frozen_config_ft_{args.project_name}_{date_str}.yaml')
    #OmegaConf.save(cfg, frozen_config_path)
    #print(f"✅ Config frozen and saved to: {frozen_config_path}")

    # =====================================================
    # PREPARE SHARED CONFIGURATION
    # =====================================================
    print("\n" + "=" * 80)
    print("📊 PREPARING SHARED CONFIGURATION")
    print("=" * 80)

    shared_config = prepare_shared_configuration(cfg)

    # Extract ray config and add shared config
    ray_config, cfg = extract_config(cfg_path=None, cfg=cfg)
    ray_config['shared_config'] = shared_config

    # =====================================================
    # DEBUG MODE
    # =====================================================
    if args.debug_mode:
        print("\n🐛 DEBUG MODE: Running single trial")

        ray_config_debug, cfg_debug = extract_fixed_config(cfg_path=None, cfg=cfg)
        ray_config_debug['shared_config'] = shared_config

        trainer_test = get_trainer(cfg.model.name)(config=ray_config_debug)

        print("\n" + "=" * 80)
        print("Running debug training steps...")
        print("=" * 80)

        for step in range(15):
            result = trainer_test.step()
            print(f"\nStep {step + 1}/15:")
            print(f"  - Epoch: {result.get('epoch', 'N/A')}")
            print(f"  - Train loss: {result.get('train_loss', 'N/A'):.6f}")
            print(f"  - Val loss: {result.get('val_loss', 'N/A'):.6f}")

            if result.get('val_f1_score'):
                print(f"  - Val F1: {result['val_f1_score']:.4f}")
            if result.get('val_roc_auc'):
                print(f"  - Val ROC-AUC: {result['val_roc_auc']:.4f}")

        print("\n✅ Debug mode completed")
        return

    # =====================================================
    # SETUP RAY TUNE
    # =====================================================

    # Get fine-tuning directory
    local_dir, pretrained_trial_id = get_finetuning_local_dir(cfg_ft.model.checkpoint_path, date_str)
    print(f"\n📁 Fine-tuning setup:")
    print(f"   - Pretrained trial: {pretrained_trial_id}")
    print(f"   - Results directory: {local_dir}")

    # Set trainer
    trainer = get_trainer(cfg.model.name)

    # Callbacks
    if args.wandb:
        callbacks = [WandbLoggerCallback(
            project=args.project_name,
            entity=args.entity,
            log_config=True,
            api_key=args.wandb_key,
            upload_checkpoints=False
        )]
    else:
        callbacks = []

    # Resources per trial
    gpu_per_trial = args.n_gpus // args.trials_per_node
    cpu_per_trial = args.n_cpus // args.trials_per_node

    # Ensure at least some resources per trial
    if gpu_per_trial == 0 and args.n_gpus > 0:
        print(f"\n⚠️  WARNING: Not enough GPUs for {args.trials_per_node} trials!")
        print(f"   - Available GPUs per node: {args.n_gpus}")
        print(f"   - Requested trials per node: {args.trials_per_node}")
        print(f"   - Setting gpu_per_trial to fractional: {args.n_gpus / args.trials_per_node:.2f}")
        gpu_per_trial = args.n_gpus / args.trials_per_node  # Fractional GPU

    if cpu_per_trial == 0:
        print(f"\n⚠️  WARNING: Not enough CPUs for {args.trials_per_node} trials!")
        print(f"   - Available CPUs per node: {args.n_cpus}")
        print(f"   - Requested trials per node: {args.trials_per_node}")
        cpu_per_trial = 1  # Minimum 1 CPU

    cfg.resources.gpu_trial = gpu_per_trial
    cfg.resources.cpu_trial = cpu_per_trial

    resources_per_trial = {
        "cpu": cfg.resources.cpu_trial,
        "gpu": cfg.resources.gpu_trial
    } if cfg.resources.gpu_trial > 0 else {"cpu": cfg.resources.cpu_trial}

    # Metrics
    metrics_dataset_available = cfg.opt.get('evaluate_metrics', False)
    metrics_dict = get_opt_metric(cfg=cfg, metrics_loader=metrics_dataset_available)
    metric, mode = metrics_dict['metric_key'], metrics_dict['mode']

    # Progress reporter
    progress_reporter = CLIReporter(
        metric_columns=[metric, f'best_{metric}'] +
                       list(cfg.opt.metrics_to_report) +
                       list(cfg.opt.other_reports)
    )

    # Scheduler
    scheduler = ASHAScheduler(
        metric=metric,
        mode=mode,
        max_t=10 ** 18,
        grace_period=50
    )

    sync_config = get_sync_config()

    print("\n" + "=" * 80)
    print("🚀 STARTING RAY TUNE (FINE-TUNING)")
    print("=" * 80)
    print(f"📁 Results directory: {local_dir}")
    print(f"🎯 Optimization metric: {metric} ({mode})")
    print(f"🔢 Number of trials: {args.num_samples}")
    print(f"💾 W&B logging: {'Enabled' if args.wandb else 'Disabled'}")
    print()

    analysis = tune.run(
        trainer,
        scheduler=scheduler,
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

    # Print results
    print("\n" + "=" * 80)
    print("✅ RAY TUNE COMPLETED (FINE-TUNING)")
    print("=" * 80)
    best_config = analysis.get_best_config(metric=metric, mode=mode)
    print(f"🏆 Best configuration:\n{best_config}")

    best_trial = analysis.get_best_trial(metric=metric, mode=mode)
    print(f"\n📊 Best trial: {best_trial.trial_id}")
    print(f"   - {metric}: {best_trial.last_result[metric]:.6f}")
    if 'val_f1_score' in best_trial.last_result:
        print(f"   - F1: {best_trial.last_result['val_f1_score']:.4f}")
    if 'val_roc_auc' in best_trial.last_result:
        print(f"   - ROC-AUC: {best_trial.last_result['val_roc_auc']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tuning with Ray Tune")
    parser.add_argument("--address", default='10.141.1.28:6379', help="Ray cluster address")
    parser.add_argument("--password", default=None, help="Ray cluster password")
    parser.add_argument("--config_file", default='conv_ae2D_CMG_ft',
                        help="Fine-tuning config file")
    parser.add_argument("--trial_per_node", default=1, type=int,
                        help="trial per node")
    parser.add_argument("--n_gpus", default=1, type=int,
                        help="n gpus per trial")
    parser.add_argument("--n_cpus", default=12, type=int,
                        help="n cpus per trial")
    parser.add_argument("--num_samples", default=100, type=int,
                        help="Number of trials")
    parser.add_argument("--wandb", default=0, type=int,
                        help="Enable W&B logging (0/1)")
    parser.add_argument("--project_name", default='conv2D_CMG_ft',
                        help="W&B project name")
    parser.add_argument("--entity", default='robmorelli',
                        help="W&B entity")
    parser.add_argument("--wandb_key", default="56b6f7f0b13c4d89207e51c28ceb90c24201eab5",
                        help="W&B API key")
    parser.add_argument("--debug_mode", default=1, type=int,
                        help="Run single trial for debugging (0/1)")

    args = parser.parse_args()

    # Environment configuration
    os.environ['TUNE_MAX_PENDING_TRIALS_PG'] = "12"

    # ✅ Initialize Ray
    ray.init(address='auto', ignore_reinit_error=True)

    # Run fine-tuning
    main(args)