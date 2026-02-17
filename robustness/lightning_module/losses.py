from torch import Tensor
from torch import nn
import torch.nn.functional as F


def reconstruction_loss(x: Tensor, x_hat: Tensor, reduction: str = "mean"):
    return F.mse_loss(x_hat, x, reduction=reduction)


def feature_errors(x: Tensor, x_hat: Tensor):
    """
    Returns per-sample per-feature errors, WITHOUT averaging over batch.
    Shape input:  [B, C, F, W]
    Shape output: [B, F]
    """
    err = (x_hat - x).pow(2)          # [B,1,F,T]
    err = err.squeeze(1)              # [B,F,T]
    err = err.median(dim=2).values    # median over time
    return err                        # [B,F]


def regularization_loss(
    encoder: nn.Module,
    x_clean: Tensor,
    x_adv: Tensor,
) -> Tensor:
    """
    Enforces consistency between latent representations.
    """
    z_clean: Tensor = encoder(x_clean)
    z_adv: Tensor = encoder(x_adv)

    return reconstruction_loss(z_adv, z_clean.detach())