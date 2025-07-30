# utils/trainer_registry.py

from trainer.lstm_ae_trainer import trainLSTMAE
from trainer.lstm_trainer import trainLSTM
from trainer.conv_ae1d_trainer import trainCONVAE1D

TRAINER_REGISTRY = {
    "lstm_ae": trainLSTMAE,
    "lstm": trainLSTM,
    "conv_ae1D": trainCONVAE1D,
    # Add new trainers here
}
