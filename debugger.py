from utils.load_trainer import get_trainer
from utils.general import make_paths_absolute, extract_config, extract_fixed_config
from ray.tune.schedulers import ASHAScheduler
from config import *  # and whatever else is needed
from omegaconf import OmegaConf
import argparse


def main(args):

    cfg_path = os.path.join(config_path, args.config_file + '.yaml')
    ray_config, cfg = extract_fixed_config(cfg_path)   # Extract fixed config parameters to avoid failure in get_trainer(cfg.model.name)(config=config) * the config is the problem

    trainer_test = get_trainer(cfg.model.name)(config=ray_config)
    trainer = get_trainer(cfg.model.name)

    if cfg.resources.gpu_trial != 0:
        resources_per_trial = {"cpu":cfg.resources.cpu_trial, "gpu": cfg.resources.gpu_trial}
    else:
        resources_per_trial = {"cpu": cfg.resources.cpu_trial}

    sched = ASHAScheduler(metric=cfg.opt.tune_report, mode="min", max_t = 10 ** 18,
                                                        grace_period=50)

    analysis = tune.run(trainer,
                        scheduler=sched,
                        resources_per_trial=resources_per_trial,
                        num_samples=args.num_samples,
                        checkpoint_at_end=True,  # otherwise it fails on multinode?
                        local_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "ray_results"),
                        name="{}/debugger".format(cfg.model.name),
                        config=ray_config)


if __name__ == "__main__":
    # ip_head and redis_passwords are set by ray cluster shell scripts
    # use the arg parse to call this script from sh script that run the cluster
    # remember to ray start --head on the node you have itneractively
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default='conv_ae1D', help="[conv_ae1D, lstm]")
    parser.add_argument("--num_samples", default=1, help="the model you want to hpo")
    args = parser.parse_args()

    main(args)

