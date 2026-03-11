import torch
from torch import Tensor, nn


def l1_attack_budget(
    model: nn.Module,
    x: Tensor,
    budget: float = 50.0,
    num_iter: int = 10,
) -> Tensor:
    if x.shape[0] == 0:
        return x

    model.eval()
    x_orig = x.detach()
    x_adv = x_orig.clone()
    loss_fn = nn.MSELoss(reduction="sum")
    l1_norm = torch.sum(x_orig.abs()).clamp(min=1e-8)
    alpha = (budget / 100.0) * l1_norm / (3 * x_orig.numel())

    with torch.no_grad():
        best_loss = loss_fn(model(x_orig), x_orig).item()
    best_x_adv = x_orig.clone()

    for _ in range(num_iter):
        x_adv = x_adv.detach().requires_grad_(True)
        loss = loss_fn(model(x_adv), x_adv)
        grad = torch.autograd.grad(loss, x_adv)[0]

        with torch.no_grad():
            x_adv = x_adv - alpha * grad
            delta = x_adv - x_orig
            change = torch.sum(delta.abs()) * 100 / l1_norm
            if change > budget:
                x_adv = x_orig + delta * (budget / change.item())
            current_loss = loss_fn(model(x_adv), x_adv).item()
            if current_loss < best_loss:
                best_loss = current_loss
                best_x_adv = x_adv.clone()

    return best_x_adv.detach()


def l0_attack_topk(
    model: nn.Module,
    x: Tensor,
    k: int = 10,
    num_iter: int = 10,
) -> Tensor:
    if x.shape[0] == 0:
        return x

    model.eval()
    x_orig = x.detach()
    x_adv = x_orig.clone()
    loss_fn = nn.SmoothL1Loss(reduction="sum")

    x_adv_init = x_orig.requires_grad_(True)
    loss = loss_fn(model(x_adv_init), x_adv_init)
    grad = torch.autograd.grad(loss, x_adv_init)[0]
    top_k_features = grad.abs().sum(dim=(0, 1, 3)).argsort()[-k:]

    selected = x_orig[:, 0, top_k_features, :]
    l1_norm_selected = selected.abs().sum().clamp(min=1e-8)
    alpha = l1_norm_selected / (3 * selected.numel())

    with torch.no_grad():
        best_loss = loss_fn(model(x_orig), x_orig).item()
    best_x_adv = x_orig.clone()

    for _ in range(num_iter):
        x_adv = x_adv.detach().requires_grad_(True)
        loss = loss_fn(model(x_adv), x_adv)
        grad = torch.autograd.grad(loss, x_adv)[0]

        with torch.no_grad():
            x_adv = x_adv.clone()
            x_adv[:, 0, top_k_features, :] -= alpha * grad[:, 0, top_k_features, :]
            current_loss = loss_fn(model(x_adv), x_adv).item()
            if current_loss < best_loss:
                best_loss = current_loss
                best_x_adv = x_adv.clone()

    return best_x_adv.detach()
