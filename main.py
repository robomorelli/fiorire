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
from utils.ray_manager import setup_and_initialize_ray, shutdown_ray
from trainer.utils import get_opt_metric
from config import *


def main(args):
    """Main training function with shared datasets."""

    '''
    # ✅ Setup Ray environment
    if args.address:
        # PBS cluster mode
        setup_and_initialize_ray(
            address=args.address,
            password=args.password
        )
    else:
        # Local mode
        setup_and_initialize_ray(
            address=None,
            object_store_memory_gb=30
        )
    '''

    # Setup
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d_%H-%M-%S")

    # ✅ Create experiment directory FIRST
    results_dir = f'./ray_results/{args.project_name}_{date_str}'
    os.makedirs(results_dir, exist_ok=True)
    print(f"\n📁 Experiment directory: {results_dir}")

    # ✅ SNAPSHOT: Copy config YAML to experiment folder
    original_cfg_path = os.path.join(config_path, args.config_file + '.yaml')
    snapshot_cfg_path = os.path.join(results_dir, f'cfg_{args.config_file}.yaml')
    shutil.copy(original_cfg_path, snapshot_cfg_path)
    print(f"✅ Config snapshot saved to: {snapshot_cfg_path}")

    # ✅ Load config FROM SNAPSHOT (not original)
    #ray_config, cfg = extract_config(snapshot_cfg_path)

    # Freeze config
    print("\n" + "=" * 80)
    print("📌 FREEZING CONFIG")
    print("=" * 80)

    cfg_frozen = OmegaConf.to_container(cfg, resolve=True)
    cfg = OmegaConf.create(cfg_frozen)
    print(f"✅ Config frozen")

    # ✅ Update ray_config to point to SNAPSHOT (not original)
    # This ensures all trials load from the snapshot
    ray_config['opt.config_file_path'] = snapshot_cfg_path
    print(f"✅ Trials will load config from: {snapshot_cfg_path}")

    # ✅ Prepare shared configuration (sequences in Ray Object Store)
    shared_config = prepare_shared_configuration(cfg)
    ray_config['shared_config'] = shared_config

    # Debug mode
    if args.debug_mode:
        print("\n🐛 DEBUG MODE: Running single trial")

        from utils.general import extract_fixed_config
        ray_config_debug, cfg_debug = extract_fixed_config(cfg_path=None, cfg=cfg)
        ray_config_debug['shared_config'] = shared_config
        ray_config_debug['opt.config_file_path'] = snapshot_cfg_path  # ← Use snapshot

        trainer_test = get_trainer(cfg.model.name)(config=ray_config_debug)

        print("\n" + "=" * 80)
        print("Running debug training steps...")
        print("=" * 80)

        for step in range(5):
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
        shutdown_ray()
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

    print("\n" + "=" * 80)
    print("🚀 STARTING RAY TUNE")
    print("=" * 80)
    print(f"📁 Results directory: {results_dir}")
    print(f"📄 Config snapshot: {snapshot_cfg_path}")
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
        config=ray_config,  # ← Contains shared_config + snapshot path
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

    '''
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"\n\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        shutdown_ray()  # ✅ Always cleanup
    '''


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ray Tune hyperparameter optimization")
    parser.add_argument("--address", default=None, help="address of master")
    parser.add_argument("--password", default=None, help="Ray cluster password")
    parser.add_argument("--config_file", "-c", default='conv_ae2D',
                        help="Config file name")
    parser.add_argument("--num_samples", default=100, type=int,
                        help="Number of trials to run")
    parser.add_argument("--wandb", default=0, type=int,
                        help="Enable W&B logging (0/1)")
    parser.add_argument("--project_name", default='fiorire1_2D',
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

    ray.init(address='auto')

    # Run optimization
    main(args)