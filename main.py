import argparse
import ray
from ray.tune.schedulers import ASHAScheduler
from utils.load_trainer import get_trainer
from utils.general import extract_config, extract_fixed_config, get_sync_config, trial_dirname_creator
from datetime import datetime
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.tune import CLIReporter
from trainer.utils import infer_metric_mode, get_opt_metric

from config import *

def main(args):

    # Get date to name the results folder
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d_%H-%M-%S")

    # Set the path to the configuration file
    cfg_path = os.path.join(config_path, args.config_file + '.yaml')
    ray_config, cfg = extract_config(cfg_path)  # Extract fixed config parameters to avoid failure in get_trainer(cfg.model.name)(config=config) * the config is the problem
    # Debug mode: simulate one training iteration to check if the config is correct
    if args.debug_mode:
        ray_config, cfg = extract_fixed_config(cfg_path)
        trainer_test = get_trainer(cfg.model.name)(config=ray_config)
        result = trainer_test.step()  # this simulates one training iteration
        print("Debug mode training result:", result)
        return

    # Set the trainer
    trainer = get_trainer(cfg.model.name)

    if args.wandb:
        callbacks = [WandbLoggerCallback(project=args.project_name, entity=args.entity,  # optional
                                log_config=True,  # logs the config used in each trial
                                api_key=args.wandb_key,upload_checkpoints = True )]
    else:
        callbacks = []


    # Set the resources for each trial
    resources_per_trial = {"cpu":cfg.resources.cpu_trial, "gpu": cfg.resources.gpu_trial} if (
            cfg.resources.gpu_trial != 0) else {"cpu": cfg.resources.cpu_trial}
    metric_loader_path = cfg.opt.metrics_dataset_path
    metrics_dict = get_opt_metric(cfg=cfg, metrics_loader=metric_loader_path)
    metric, mode = metrics_dict['metric_key'], metrics_dict['mode']
    progress_reporter = CLIReporter(
        metric_columns=[metric, f'best_{metric}'] + list(cfg.opt.metrics_to_report) + list(cfg.opt.other_reports))
    sched = ASHAScheduler(metric=metric, mode=mode, max_t = 10 ** 18, grace_period=50)
    sync_config = get_sync_config()
    analysis = tune.run(trainer,
                        scheduler=sched, resources_per_trial=resources_per_trial,
                        num_samples=int(args.num_samples),
                        #local_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "ray_results"),
                        local_dir='./ray_results/{}_{}_{}'.format(args.project_name, cfg.opt.exp_name, date_str),
                        #sync_config=tune.SyncConfig(syncer=None),
                        name="{}".format(cfg.opt.exp_name),
                        progress_reporter=progress_reporter,   # <-- add this here
                        sync_config=sync_config,
                        config=ray_config, callbacks=callbacks,
                        checkpoint_at_end=False,
                        checkpoint_freq=0,
                        keep_checkpoints_num=1,
                        trial_dirname_creator=lambda trial: trial_dirname_creator(trial, max_params=5),
                        stop = {"training_iteration": cfg.opt.max_epochs},)

    print("Best config is:", analysis.get_best_config(metric="val_loss", mode="min"))


if __name__ == "__main__":
    # ip_head and redis_passwords are set by ray cluster shell scripts
    # use the arg parse to call this script from sh script that run the cluster
    # remember to ray start --head on the node you have itneractively
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", default = '10.141.1.28:6379', help="adress of master")
    parser.add_argument("--password", help="password to connect to master")
    parser.add_argument("--config_file", default='conv_ae1D', help="[conv_ae1D, conv_ae2D, lstm]")
    parser.add_argument("--num_samples", default=100, help="the model you want to hpo")
    parser.add_argument("--wandb", default=0, type=int, help="the model you want to hpo")
    parser.add_argument("--project_name", default='fiorire1_1D', help="the model you want to hpo")
    parser.add_argument("--entity", default='robmorelli', help="the model you want to hpo")
    parser.add_argument("--wandb_key", default="56b6f7f0b13c4d89207e51c28ceb90c24201eab5", help="the model you want to hpo")
    parser.add_argument("--debug_mode", default=0, help="the model you want to hpo")
    args = parser.parse_args()

    os.environ['TUNE_MAX_PENDING_TRIALS_PG'] = "12"

    # to test on interactive node
    # first start from the terminal: ray start --head
    # args.address have to be the address of the node otherwise uncomment ray.init(address='auto') line
    #ray.init(address='auto') #
    ray.init(address='auto')
    ###### ISSUE when start on address but ray try to connect to localhost
    ##########No available node types can fulfill resource request
    ########No available node types can fulfill resource request
    ###########No available node types can fulfill resource request node
    ### SOLUTION
    ##### disable wifi

    #ray.init(address='192.168.43.136:6379')
    #ray.init(address='auto', _node_ip_address=args.address.split(":")[0], _redis_password=args.password)


    main(args)

