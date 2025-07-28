from trainer.conv_ae1d_trainer import trainCONVAE1D  # import your trainer class
from config import *  # and whatever else is needed
from omegaconf import OmegaConf
from omegaconf import ListConfig
import argparse


def extract_fixed_config(cfg):
    config = {}
    for k, v in cfg.tune_config.items():
        if isinstance(v, (list, ListConfig)):  # e.g. [0.001, 0.003]
            config[k] = v[0]  # Use first value for testing
        elif isinstance(v, str) and v.startswith("tune.choice"):
            # Handle cases where string parsing is needed
            values = v[v.find("[")+1 : v.find("]")].split(",")
            config[k] = eval(values[0].strip())
        else:
            config[k] = v  # Fallback
    return config

def main(args):
    cfg = OmegaConf.load(os.path.join(config_path, args.config_file + '.yaml'))

    config = extract_fixed_config(cfg)

    #trainer = get_trainer(cfg)
    trial = trainCONVAE1D(config=config)
    trial.setup(config)
    result = trial.train_conv_ae1D()
    print(result)


if __name__ == "__main__":
    # ip_head and redis_passwords are set by ray cluster shell scripts
    # use the arg parse to call this script from sh script that run the cluster
    # remember to ray start --head on the node you have itneractively
    parser = argparse.ArgumentParser()
    #parser.add_argument("--config_path", default='./train_configurations/', help="echo the string you use here")
    parser.add_argument("--config_file", default='conv_ae1D', help="the model you want to hpo")
    args = parser.parse_args()

    main(args)

