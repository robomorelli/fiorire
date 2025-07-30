# utils/load_trainer.py

from utils.trainer_registry import TRAINER_REGISTRY

def get_trainer(cfg, **kwargs):
    trainer_name = cfg.model.name
    try:
        return TRAINER_REGISTRY[trainer_name]
    except KeyError:
        raise ValueError(f"Trainer '{trainer_name}' not found in registry.")
