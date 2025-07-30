# utils/model_registry.py

from models.lstm_ae import LSTM_AE
from models.lstm import LSTM
from models.conv_ae1D import CONV_AE1D

MODEL_REGISTRY = {
    "lstm_ae": LSTM_AE,
    "lstm": LSTM,
    "conv_ae1D": CONV_AE1D,
    # Add new models here
}