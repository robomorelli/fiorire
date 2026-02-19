import torch
from torch import Tensor, nn
from robustness.lightning_module.losses import reconstruction_loss


def pgd_attack(
    model: nn.Module,
    x: Tensor,
    epsilon: float,
    alpha: float,
    steps: int,
):
    x_adv = x.detach().clone().requires_grad_(True)
    x_orig = x.detach()

    for _ in range(steps):
        x_hat = model(x_adv)
        # pgd searches perturbations far from clean manifold
        loss = reconstruction_loss(x_hat, x_orig)

        grad = torch.autograd.grad(loss, x_adv)[0]

        with torch.no_grad():
            x_adv += alpha * grad.sign()
            x_adv = torch.max(
                torch.min(x_adv, x_orig + epsilon),
                x_orig - epsilon,
            )
            x_adv.requires_grad_(True)

    return x_adv.detach()
