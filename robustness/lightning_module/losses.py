from torch import Tensor
from torch import nn
import torch.nn.functional as F


def reconstruction_loss(x: Tensor, x_hat: Tensor, reduction: str = "mean"):
    return F.mse_loss(x_hat, x, reduction=reduction)


def feature_errors(x: Tensor, x_hat: Tensor):
    return (x_hat - x).pow(2).mean(dim=(0, 1, 3))


def regularization_loss(
    encoder: nn.Module,
    x_clean: Tensor,
    x_adv: Tensor,
) -> Tensor:
    """
    Enforces consistency between latent representations.
    """
    z_clean: Tensor = encoder(x_clean).detach()
    z_adv: Tensor = encoder(x_adv)

    return reconstruction_loss(z_adv, z_clean)