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
import gc

from config import *

def cleanup_ray():
    """Cleanup Ray environment before starting."""
    print("\n" + "=" * 80)
    print("🧹 CLEANING UP RAY ENVIRONMENT")
    print("=" * 80)

    # Shutdown existing Ray instance
    if ray.is_initialized():
        print("   - Shutting down existing Ray instance...")
        ray.shutdown()

    # Force garbage collection
    print("   - Running garbage collection...")
    gc.collect()

    # Clear Ray temporary directory (optional but helpful)
    import tempfile
    ray_tmp_dir = os.path.join(tempfile.gettempdir(), 'ray')
    if os.path.exists(ray_tmp_dir):
        try:
            # Don't delete if other Ray instances are using it
            print(f"   - Ray temp directory: {ray_tmp_dir}")
        except:
            pass

    print("   ✅ Cleanup complete")


def initialize_ray_with_memory(memory_gb=10, address='auto', password=None):
    """Initialize Ray with increased object store memory (only for local mode)."""
    print("\n" + "=" * 80)
    print("🚀 INITIALIZING RAY")
    print("=" * 80)

    object_store_memory = memory_gb * 1024 ** 3

    print(f"   - Address: {address}")
    if password:
        print(f"   - Using password: {'*' * 8}")

    # ✅ Base kwargs
    init_kwargs = {
        'address': address,
        'ignore_reinit_error': True,
    }

    # ✅ Aggiungi memoria SOLO se NON ti stai connettendo a cluster esistente
    if address in [None, 'local']:
        # Modalità locale - possiamo specificare memoria
        print(f"   - Object store memory: {memory_gb} GB (local mode)")
        init_kwargs['object_store_memory'] = object_store_memory
        init_kwargs['_memory'] = object_store_memory * 2
    elif address == 'auto':
        # Auto mode - potrebbe essere locale o cluster
        # Non specifichiamo memoria - Ray userà quella del cluster se esiste
        print(f"   - Object store memory: using cluster settings (auto mode)")
    else:
        # Address esplicito - sicuramente cluster esistente
        print(f"   - Object store memory: using cluster settings (cluster mode)")

    # ✅ Aggiungi password se fornita
    if password:
        init_kwargs['_redis_password'] = password

    ray.init(**init_kwargs)

    print(f"   ✅ Ray initialized")

    # Mostra info cluster
    try:
        nodes = ray.nodes()
        resources = ray.available_resources()
        print(f"   - Connected nodes: {len(nodes)}")
        print(f"   - Available CPUs: {resources.get('CPU', 0):.0f}")
        print(f"   - Available GPUs: {resources.get('GPU', 0):.0f}")

        if len(nodes) > 1:
            print(f"   ✅ Cluster mode: {len(nodes)} nodes")
        else:
            print(f"   ℹ️  Single node mode")
    except Exception as e:
        print(f"   ⚠️  Could not get cluster info: {e}")


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
    parser.add_argument("--address", default=None, help="Ray cluster address")
    parser.add_argument("--password", default=None, help="Ray cluster password")
    parser.add_argument("--config_file", default='conv_ae1D_MGM_ft',
                        help="Fine-tuning config file")
    parser.add_argument("--trials_per_node", default=1, type=int,
                        help="trial per node")
    parser.add_argument("--n_gpus", default=1, type=int,
                        help="n gpus per trial")
    parser.add_argument("--n_cpus", default=12, type=int,
                        help="n cpus per trial")
    parser.add_argument("--num_samples", default=100, type=int,
                        help="Number of trials")
    parser.add_argument("--wandb", default=0, type=int,
                        help="Enable W&B logging (0/1)")
    parser.add_argument("--project_name", default='conv1D_MGM_ft',
                        help="W&B project name")
    parser.add_argument("--entity", default='robmorelli',
                        help="W&B entity")
    parser.add_argument("--wandb_key", default="56b6f7f0b13c4d89207e51c28ceb90c24201eab5",
                        help="W&B API key")
    parser.add_argument("--debug_mode", default=0
                        , type=int,
                        help="Run single trial for debugging (0/1)")
    parser.add_argument("--ray_memory_gb", default=10, type=int,
                        help="Ray object store memory in GB")

    args = parser.parse_args()

    # Environment configuration
    os.environ['TUNE_MAX_PENDING_TRIALS_PG'] = "6"

    # Cleanup and initialize Ray
    cleanup_ray()
    initialize_ray_with_memory(
        memory_gb=args.ray_memory_gb,
        address=args.address,
        password=args.password
    )

    try:
        main(args)
    finally:
        print("\n🛑 Shutting down Ray...")
        ray.shutdown()
        print("   ✅ Done")

    # Run fine-tuning
    main(args)