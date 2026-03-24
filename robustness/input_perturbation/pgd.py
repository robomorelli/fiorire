import torch
from torch import Tensor, nn


def l1_attack_budget(
    model: nn.Module,
    x: Tensor,
    budget: float = 50.0,
    num_iter: int = 10,
    lr: float = 1e-2,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps_adam: float = 1e-8,
) -> Tensor:
    """
    Evasion attack under an L1-budget constraint, optimised with Adam.

    The goal is to *minimise* reconstruction error so the anomaly-detector
    scores the adversarial sample as clean.  Adam replaces the hand-tuned
    per-sample step-size of the original version.
    """
    if x.shape[0] == 0:
        return x

    model.eval()
    x_orig = x.detach()
    B = x_orig.shape[0]
    extra = [1] * (x.dim() - 1)          # for broadcasting over (C,T,F) etc.

    # Per-sample L1 norm used to express the budget as a percentage
    l1_norm = x_orig.flatten(1).abs().sum(dim=1).clamp(min=1e-8)   # [B]
    max_delta_l1 = (budget / 100.0) * l1_norm                       # [B]
    max_delta_l1_bc = max_delta_l1.reshape(B, *extra)

    # Learnable perturbation (starts at zero, i.e. x_orig)
    delta = torch.zeros_like(x_orig, requires_grad=False)

    # Adam moments  (maintained manually so we can project after each step)
    m = torch.zeros_like(delta)
    v = torch.zeros_like(delta)

    # Per-sample best-iterate tracking
    with torch.no_grad():
        best_score = (model(x_orig) - x_orig).pow(2).flatten(1).mean(dim=1)   # [B]
    best_delta = torch.zeros_like(x_orig)

    for t in range(1, num_iter + 1):
        delta_t = delta.detach().requires_grad_(True)
        x_adv = x_orig + delta_t

        loss = ((model(x_adv) - x_adv) ** 2).sum()           # minimise → evasion
        grad = torch.autograd.grad(loss, delta_t)[0].detach()

        with torch.no_grad():
            # ── Adam update ────────────────────────────────────────────────
            m = beta1 * m + (1.0 - beta1) * grad
            v = beta2 * v + (1.0 - beta2) * grad.pow(2)
            m_hat = m / (1.0 - beta1 ** t)
            v_hat = v / (1.0 - beta2 ** t)

            # Scale lr by per-sample l1_norm so the effective step is
            # comparable to the old "alpha = 0.01 * l1_norm" heuristic
            lr_bc = (lr * l1_norm).reshape(B, *extra)
            delta = delta - lr_bc * m_hat / (v_hat.sqrt() + eps_adam)

            # ── Project delta onto the L1 ball ────────────────────────────
            delta_l1 = delta.flatten(1).abs().sum(dim=1).clamp(min=1e-8)
            delta_l1_bc = delta_l1.reshape(B, *extra)
            scale = (max_delta_l1_bc / delta_l1_bc).clamp(max=1.0)
            delta = delta * scale

            # ── Per-sample best-iterate ───────────────────────────────────
            current_score = (
                (model(x_orig + delta) - (x_orig + delta))
                .pow(2).flatten(1).mean(dim=1)
            )                                                              # [B]
            improved = current_score < best_score
            best_score = torch.where(improved, current_score, best_score)
            mask = improved.reshape(B, *extra)
            best_delta = torch.where(mask, delta, best_delta)

    return (x_orig + best_delta).detach()


def l0_attack_topk(
    model: nn.Module,
    x: Tensor,
    k: int = 10,
    num_iter: int = 10,
    lr: float = 1e-2,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps_adam: float = 1e-8,
) -> Tensor:
    """
    Evasion attack that modifies only the top-k features (by gradient magnitude),
    optimised with Adam.

    Two bugs in the original are also fixed here:
      - best_score was a scalar mean across the batch (now tracked per-sample).
      - alpha depended on a single selected-feature slice sum, not per-sample.
    """
    if x.shape[0] == 0:
        return x

    model.eval()
    x_orig = x.detach()
    B = x_orig.shape[0]
    extra = [1] * (x.dim() - 1)

    # ── Select top-k features once from the initial gradient ──────────────
    x_init = x_orig.clone().requires_grad_(True)
    loss_init = ((model(x_init) - x_init) ** 2).sum()
    grad_init = torch.autograd.grad(loss_init, x_init)[0].detach()
    # aggregate over batch and any non-feature dims; adjust slicing to your layout
    top_k_idx = grad_init.abs().sum(dim=(0, 1, 3)).argsort(descending=True)[:k]

    # ── Per-sample best-iterate tracking ──────────────────────────────────
    with torch.no_grad():
        best_score = (model(x_orig) - x_orig).pow(2).flatten(1).mean(dim=1)  # [B]
    best_delta = torch.zeros_like(x_orig)

    # Learnable delta restricted to the k selected features
    delta = torch.zeros_like(x_orig)

    # Adam moments (only the selected-feature slice matters, but storing full
    # tensors keeps the indexing clean)
    m = torch.zeros_like(delta)
    v = torch.zeros_like(delta)

    for t in range(1, num_iter + 1):
        delta_t = delta.detach().requires_grad_(True)
        x_adv = x_orig + delta_t

        loss = ((model(x_adv) - x_adv) ** 2).sum()
        grad = torch.autograd.grad(loss, delta_t)[0].detach()

        with torch.no_grad():
            # Zero out gradient for non-selected features so Adam never
            # accumulates moments for them
            grad_masked = torch.zeros_like(grad)
            grad_masked[:, :, top_k_idx, :] = grad[:, :, top_k_idx, :]

            # ── Adam update ───────────────────────────────────────────────
            m = beta1 * m + (1.0 - beta1) * grad_masked
            v = beta2 * v + (1.0 - beta2) * grad_masked.pow(2)
            m_hat = m / (1.0 - beta1 ** t)
            v_hat = v / (1.0 - beta2 ** t)

            step = lr * m_hat / (v_hat.sqrt() + eps_adam)
            delta = delta - step
            # Enforce the L0 mask: only selected features may be non-zero
            delta_full = torch.zeros_like(delta)
            delta_full[:, :, top_k_idx, :] = delta[:, :, top_k_idx, :]
            delta = delta_full

            # ── Per-sample best-iterate ───────────────────────────────────
            current_score = (
                (model(x_orig + delta) - (x_orig + delta))
                .pow(2).flatten(1).mean(dim=1)
            )                                                             # [B]
            improved = current_score < best_score
            best_score = torch.where(improved, current_score, best_score)
            mask = improved.reshape(B, *extra)
            best_delta = torch.where(mask, delta, best_delta)

    return (x_orig + best_delta).detach()