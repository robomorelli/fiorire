from torch import Tensor
import torch.nn.functional as F


def reconstruction_loss(x: Tensor, x_hat: Tensor):
    return F.mse_loss(x_hat, x)


def feature_errors(x: Tensor, x_hat: Tensor):
    return (x_hat - x).pow(2).mean(dim=(0, 1, 3))
