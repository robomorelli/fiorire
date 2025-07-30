# utils/load_trainer.py

from utils.trainer_registry import TRAINER_REGISTRY

def get_trainer(cfg_model_name, **kwargs):
    try:
        return TRAINER_REGISTRY[cfg_model_name]
    except KeyError:
        raise ValueError(f"Trainer '{cfg_model_name}' not found in registry.")
