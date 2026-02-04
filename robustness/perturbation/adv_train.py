import torch
from torch import Tensor, nn
from robustness.lightning_module.losses import reconstruction_loss


def fgsm_attack(
    model: nn.Module,
    x: Tensor,
    epsilon: float,
) -> Tensor:
    """
    FGSM adversarial attack on reconstruction loss.
    """
    x_adv: Tensor = x.detach().clone().requires_grad_(True)
    x_hat: Tensor = model(x_adv)

    loss: Tensor = reconstruction_loss(x_hat, x_adv)
    grad: Tensor = torch.autograd.grad(loss, x_adv)[0]

    return (x_adv + epsilon * grad.sign()).detach()


def latent_consistency_loss(
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
