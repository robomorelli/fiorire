import argparse
import os
import shutil
import ray
from ray.tune.schedulers import ASHAScheduler
from utils.load_trainer import get_trainer
from utils.general import extract_config, extract_fixed_config, get_sync_config, trial_dirname_creator
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
    # CREATE LOCAL DIR AND COPY YAML (snapshot)
    # =====================================================
    local_dir = os.path.join('./ray_results', f'{args.project_name}_{args.config_file}_{date_str}')
    os.makedirs(local_dir, exist_ok=True)

    # Original config path
    original_cfg_path = os.path.join(config_path, args.config_file + '.yaml')
    # Snapshot config path
    snapshot_cfg_path = os.path.join(local_dir, f'cfg_{args.config_file}.yaml')
    shutil.copy(original_cfg_path, snapshot_cfg_path)
    print(f"✅ YAML snapshot copied to experiment folder: {local_dir}")

    # =====================================================
    # LOAD CONFIG FROM SNAPSHOT (FROZEN)
    # =====================================================
    ray_config, cfg = extract_config(snapshot_cfg_path)
    # Freeze all values so Ray non rilegga riferimenti
    cfg_frozen = OmegaConf.to_container(cfg, resolve=True)
    cfg = OmegaConf.create(cfg_frozen)

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
    # SETUP RAY TUNE
    # =====================================================
    trainer = get_trainer(cfg.model.name)

    # Callbacks
    callbacks = []
    if args.wandb:
        callbacks = [WandbLoggerCallback(
            project=args.project_name,
            entity=args.entity,
            log_config=True,
            api_key=args.wandb_key,
            upload_checkpoints=True
        )]

    # Resources per trial
    resources_per_trial = {"cpu": cfg.resources.cpu_trial}
    if cfg.resources.gpu_trial != 0:
        resources_per_trial["gpu"] = cfg.resources.gpu_trial

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
    print(f"📁 Results will be saved to: {local_dir}\n")

    analysis = ray.tune.run(
        trainer,
        scheduler=sched,
        resources_per_trial=resources_per_trial,
        num_samples=int(args.num_samples),
        local_dir=local_dir,
        name=cfg.opt.exp_name,
        progress_reporter=progress_reporter,
        sync_config=sync_config,
        config=ray_config,  # FROZEN config
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
    parser.add_argument("--config_file", default='conv_ae2D')
    parser.add_argument("--num_samples", default=100)
    parser.add_argument("--wandb", default=0, type=int)
    parser.add_argument("--project_name", default='fiorire1_2D')
    parser.add_argument("--entity", default='robmorelli')
    parser.add_argument("--wandb_key", default="56b6f7f0b13c4d89207e51c28ceb90c24201eab5")
    parser.add_argument("--debug_mode", default=0)
    args = parser.parse_args()

    os.environ['TUNE_MAX_PENDING_TRIALS_PG'] = "12"

    # Initialize Ray
    ray.init(address='auto')

    # Run main
    main(args)
