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

    # per-sample L1 norm: shape [B, 1, 1, 1]
    l1_norm = x_orig.flatten(1).abs().sum(dim=1, keepdim=True).clamp(min=1e-8)
    l1_norm = l1_norm.reshape(-1, *([1] * (x.dim() - 1)))

    alpha = 0.01 * l1_norm  # per-sample step size

    # per-sample best tracking
    with torch.no_grad():
        init_out = model(x_orig)
    best_score = (init_out - x_orig).pow(2).flatten(1).mean(dim=1)  # [B]
    best_x_adv = x_orig.clone()

    for _ in range(num_iter):
        x_adv = x_adv.detach().requires_grad_(True)
        out = model(x_adv)
        loss = ((out - x_adv.detach()) ** 2).sum()
        grad = torch.autograd.grad(loss, x_adv)[0]

        with torch.no_grad():
            # per-sample gradient normalization
            grad_l1 = grad.flatten(1).abs().sum(dim=1, keepdim=True)
            grad_l1 = grad_l1.reshape(-1, *([1] * (x.dim() - 1))).clamp(min=1e-8)
            grad_normalized = grad / grad_l1
            x_adv = x_adv - alpha * grad_normalized

            # per-sample projection onto L1 ball
            delta = x_adv - x_orig
            change = delta.flatten(1).abs().sum(dim=1, keepdim=True)
            change = change.reshape(-1, *([1] * (x.dim() - 1)))
            change_pct = change * 100.0 / l1_norm
            scale = (budget / change_pct).clamp(max=1.0)
            x_adv = x_orig + delta * scale

            # per-sample best-iterate tracking
            current_score = ((model(x_adv) - x_adv) ** 2).flatten(1).mean(dim=1)  # [B]
            improved = current_score < best_score  # [B]
            best_score = torch.where(improved, current_score, best_score)
            mask = improved.reshape(-1, *([1] * (x.dim() - 1)))
            best_x_adv = torch.where(mask, x_adv, best_x_adv)

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

    # select top-k features by gradient magnitude using detached target
    x_init = x_orig.requires_grad_(True)
    loss_init = ((model(x_init) - x_init.detach()) ** 2).sum()
    grad_init = torch.autograd.grad(loss_init, x_init)[0]
    top_k_features = grad_init.abs().sum(dim=(0, 1, 3)).argsort()[-k:]

    # fixed step size on selected features
    alpha = 0.01 * x_orig[:, 0, top_k_features, :].abs().sum().clamp(min=1e-8)

    with torch.no_grad():
        best_score = (model(x_orig) - x_orig).pow(2).mean().item()
    best_x_adv = x_orig.clone()

    for _ in range(num_iter):
        x_adv = x_adv.detach().requires_grad_(True)
        # detached target
        loss = ((model(x_adv) - x_adv.detach()) ** 2).sum()
        grad = torch.autograd.grad(loss, x_adv)[0]

        with torch.no_grad():
            # normalize and step only on selected features
            grad_sel = grad[:, 0, top_k_features, :]
            grad_sel_normalized = grad_sel / grad_sel.abs().sum().clamp(min=1e-8)
            x_adv = x_adv.clone()
            x_adv[:, 0, top_k_features, :] -= alpha * grad_sel_normalized

            current_score = (model(x_adv) - x_adv).pow(2).mean().item()
            if current_score < best_score:
                best_score = current_score
                best_x_adv = x_adv.clone()

    return best_x_adv.detach()
