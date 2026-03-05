import torch
from torch import Tensor, nn


def l2_attack_budget(
    model: nn.Module,
    x: Tensor,
    budget: float = 50.0,
    num_iter: int = 10,
) -> Tensor:
    """
    L2 adversarial attack con clip hard sul budget L1.
    Restituisce il x_adv con reconstruction error minimo trovato durante il loop
    (strategia best-step), garantendo che l'attacco non peggiori mai lo score.
    budget: max perturbation as % of input L1 norm
    num_iter: più iterazioni = direzione di attacco migliore, non più perturbazione
    """
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

            # clip hard: la perturbazione L1 non supera mai budget%
            delta = x_adv - x_orig
            change = torch.sum(delta.abs()) * 100 / l1_norm
            if change > budget:
                x_adv = x_orig + delta * (budget / change.item())

            # best-step: tieni il punto con reconstruction error minimo
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
    """
    L0 adversarial attack. Perturbs only the top-k most influential features.
    Restituisce il x_adv con reconstruction error minimo trovato durante il loop
    (strategia best-step), garantendo che l'attacco non peggiori mai lo score.
    x: [B, C, F, W] — only anomalous samples
    k: number of features to perturb (selected by gradient magnitude)
    alpha adattivo: proporzionale alla norma L1 delle feature selezionate
    """
    if x.shape[0] == 0:
        return x

    model.eval()
    x_orig = x.detach()
    x_adv = x_orig.clone()
    loss_fn = nn.SmoothL1Loss(reduction="sum")

    # primo gradiente per scegliere le feature
    x_adv_init = x_orig.requires_grad_(True)
    loss = loss_fn(model(x_adv_init), x_adv_init)
    grad = torch.autograd.grad(loss, x_adv_init)[0]
    top_k_features = grad.abs().sum(dim=(0, 1, 3)).argsort()[-k:]  # [k]

    # alpha adattivo: scala con la norma L1 delle sole feature selezionate
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

            # best-step: tieni il punto con reconstruction error minimo
            current_loss = loss_fn(model(x_adv), x_adv).item()
            if current_loss < best_loss:
                best_loss = current_loss
                best_x_adv = x_adv.clone()

    return best_x_adv.detach()
