"""
Main training script with Ray Tune optimization.
"""

import argparse
import os
import shutil
from ray.tune.schedulers import ASHAScheduler
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.tune import CLIReporter
from datetime import datetime
from omegaconf import OmegaConf
from ray import tune
import ray

from utils.load_trainer import get_trainer
from utils.general import extract_config, get_sync_config, trial_dirname_creator
from utils.load_dataset import prepare_shared_configuration
from trainer.utils import get_opt_metric
from config import *


def main(args):
    """Main training function with shared datasets."""

    # Setup
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d_%H-%M-%S")

    # Load config
    cfg_path = os.path.join(config_path, args.config_file + '.yaml')
    ray_config, cfg = extract_config(cfg_path)

    # Freeze config
    print("\n" + "=" * 80)
    print("📌 FREEZING CONFIG")
    print("=" * 80)

    cfg_frozen = OmegaConf.to_container(cfg, resolve=True)
    cfg = OmegaConf.create(cfg_frozen)

    #frozen_config_path = os.path.join('/tmp', f'frozen_config_{args.project_name}_{date_str}.yaml')
    #OmegaConf.save(cfg, frozen_config_path)
    #print(f"✅ Config frozen and saved to: {frozen_config_path}")

    # Prepare shared configuration
    shared_config = prepare_shared_configuration(cfg)
    ray_config['shared_config'] = shared_config


    # Debug mode
    if args.debug_mode:
        print("\n🐛 DEBUG MODE: Running single trial")

        from utils.general import extract_fixed_config
        ray_config_debug, cfg_debug = extract_fixed_config(cfg_path=None, cfg=cfg)
        ray_config_debug['shared_config'] = shared_config

        trainer_test = get_trainer(cfg.model.name)(config=ray_config_debug)

        print("\n" + "=" * 80)
        print("Running debug training steps...")
        print("=" * 80)

        for step in range(15):
            result = trainer_test.step()
            print(f"\nStep {step + 1}/5:")
            print(f"  - Epoch: {result.get('epoch', 'N/A')}")
            print(f"  - Train loss: {result.get('train_loss', 'N/A'):.6f}")
            print(f"  - Val loss: {result.get('val_loss', 'N/A'):.6f}")

            if result.get('val_f1_score'):
                print(f"  - Val F1: {result['val_f1_score']:.4f}")
            if result.get('val_roc_auc'):
                print(f"  - Val ROC-AUC: {result['val_roc_auc']:.4f}")

        print("\n✅ Debug mode completed")
        return

    # Setup Ray Tune
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

    results_dir = os.path.join('./ray_results', f'{args.project_name}_{args.config_file}_{date_str}')

    print("\n" + "=" * 80)
    print("🚀 STARTING RAY TUNE")
    print("=" * 80)
    print(f"📁 Results directory: {results_dir}")
    print(f"🎯 Optimization metric: {metric} ({mode})")
    print(f"🔢 Number of trials: {args.num_samples}")
    print(f"💾 W&B logging: {'Enabled' if args.wandb else 'Disabled'}")
    print()

    analysis = tune.run(
        trainer,
        scheduler=scheduler,
        resources_per_trial=resources_per_trial,
        num_samples=int(args.num_samples),
        local_dir=results_dir,
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
    print("✅ RAY TUNE COMPLETED")
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
    parser = argparse.ArgumentParser(description="Ray Tune hyperparameter optimization")
    parser.add_argument("--address", default=None, help="address of master")
    parser.add_argument("--password", default=None, help="Ray cluster password")
    parser.add_argument("--config_file", "-c", default='conv_ae2D_ref',
                        help="Config file name")
    parser.add_argument("--num_samples", default=100, type=int,
                        help="Number of trials to run")
    parser.add_argument("--wandb", default=0, type=int,
                        help="Enable W&B logging (0/1)")
    parser.add_argument("--project_name", default='fiorire1_2D_zbook',
                        help="W&B project name")
    parser.add_argument("--entity", default='robmorelli',
                        help="W&B entity name")
    parser.add_argument("--wandb_key",
                        default="56b6f7f0b13c4d89207e51c28ceb90c24201eab5",
                        help="W&B API key")
    parser.add_argument("--debug_mode", default=0, type=int,
                        help="Run single trial for debugging (0/1)")

    args = parser.parse_args()

    # Environment configuration
    os.environ['TUNE_MAX_PENDING_TRIALS_PG'] = "12"

    # ✅ Initialize Ray (auto-connect to cluster or start local)
    ray.init(address='auto', ignore_reinit_error=True)

    # Run optimization
    main(args)