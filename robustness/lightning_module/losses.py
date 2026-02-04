import torch
import torch.nn.functional as F


def reconstruction_loss(x: torch.Tensor, x_hat: torch.Tensor):
    return F.mse_loss(x_hat, x)


def feature_errors(x:torch.Tensor, x_hat: torch.Tensor):
    return (x_hat - x).pow(2).mean(dim=(0, 1, 3))
