import numpy as np
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
    model.eval()
    x_adv = x.detach().clone().requires_grad_(True)
    x_orig = x.detach()

    for _ in range(steps):
        with torch.enable_grad():   # ← FONDAMENTALE
            x_adv.requires_grad_(True)

            x_hat = model(x_adv)
            loss = reconstruction_loss(x_hat, x_orig)

            grad = torch.autograd.grad(
                loss,
                x_adv,
                retain_graph=False,
                create_graph=False,
            )[0]

        with torch.no_grad():
            x_adv = x_adv - alpha * grad.sign()
            x_adv = torch.clamp(x_adv, x_orig - epsilon, x_orig + epsilon)

    return x_adv.detach()


def adaptive_pgd_steps(epsilon, alpha):
    return max(20, int(np.ceil(epsilon / alpha)))
