# utils/trainer_registry.py

from trainer.trainer import Trainer


TRAINER_REGISTRY = {
    "lstm_ae": Trainer,
    "lstm": Trainer,
    "conv_ae1D": Trainer,
    "conv_ae2D": Trainer,
    # Add new trainers here
}
