import argparse
import ray
from ray.tune.schedulers import ASHAScheduler, HyperBandScheduler
from omegaconf import OmegaConf
from config import *
from utils.load_trainer import get_trainer
from datetime import datetime

def main(args):
    now = datetime.now()
    date = now.strftime("%D:%H:%M:%S")
    print(date)

    print(args.address, args.password)
    cfg = OmegaConf.load(config_path + args.config_file + '.yaml')

    config = {}
    for k, v in cfg.tune_config.items():
        try:
            config[k] = ray_mapper[v.split('(')[0]]([float(s) if '.' in s else int(s) for s in v.split(v.split('(')[0])[1].\
                                                strip("()").strip("[]").split(',')])
            print([float(s) if '.' in s else int(s) for s in v.split(v.split('(')[0])[1].\
                                                strip("()").strip("[]").split(',')])
        except:
            config[k] = ray_mapper[v.split('(')[0]]([s.strip(' ').strip("''") for s in v.split(v.split('(')[0])[1]\
                                                    .strip("()").strip("[]").split(',')])
            print([s.strip(' ').strip("''") for s in v.split(v.split('(')[0])[1].strip("()")\
                  .strip("[]").split(',')])

    trainer = get_trainer(cfg)
    if cfg.resources.gpu_trial != 0:
        resources_per_trial = {"cpu":cfg.resources.cpu_trial, "gpu": cfg.resources.gpu_trial}
    else:
        resources_per_trial = {"cpu": cfg.resources.cpu_trial}

    sched = ASHAScheduler(metric=cfg.opt.tune_report, mode="min", max_t = 10 ** 18,
                                                        grace_period=50)
    analysis = tune.run(trainer,
                        scheduler=sched,
                        resources_per_trial=resources_per_trial,
                        num_samples=4,
                        checkpoint_at_end=True, #otherwise it fails on multinode?
                        local_dir="~/ray_results",
                        name="{}/4_wheels_system_{}_conv_ae1D_sl_40_12M_unscaled_unscaled_overlap_0".format(cfg.model.name,date.replace('/', '-')),

                        config=config)

    print("Best config is:", analysis.get_best_config(metric="val_loss", mode="min"))


if __name__ == "__main__":
    # ip_head and redis_passwords are set by ray cluster shell scripts
    # use the arg parse to call this script from sh script that run the cluster
    # remember to ray start --head on the node you have itneractively
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", default = '192.168.43.136:6379', help="adress of master")
    parser.add_argument("--password", help="password to connect to master")
    #parser.add_argument("--config_path", default='./train_configurations/', help="echo the string you use here")
    parser.add_argument("--config_file", default='lstm_ae', help="the model you want to hpo")
    args = parser.parse_args()

    os.environ['TUNE_MAX_PENDING_TRIALS_PG'] = "12"

    # to test on interactive node
    # first start from the terminal: ray start --head
    # args.address have to be the address of the node otherwis uncomment ray.init(address='auto') line
    ray.init(address='auto') #
    ###### ISSUE when start on address but ray try to connect to localhost
    ##########No available node types can fulfill resource request
    ########No available node types can fulfill resource request
    ###########No available node types can fulfill resource request node
    #à## SOLUTION
    ##### disable wifi

    #ray.init(address='192.168.43.136:6379')
    #ray.init(address='auto', _node_ip_address=args.address.split(":")[0], _redis_password=args.password)

    main(args)

