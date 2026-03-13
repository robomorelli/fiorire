from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm

from robustness.evaluation.metrics import write_metrics_csv

# batched L1 attack
def _l1_attack_budget_batched(
    model: nn.Module,
    X: Tensor,
    budgets: Tensor,
    num_iter: int = 5,
) -> Tensor:
    """
    Batched PGD L1 attack. Each sample in the batch has its own budget.

    Args:
        model:    Raw autoencoder nn.Module.
        X:        Input batch [B, ...].
        budgets:  Per-sample L1 budgets as % of input L1 norm, shape [B].
        num_iter: Number of PGD steps.

    Returns:
        Adversarial batch [B, ...].
    """
    X_orig = X.detach()

    # per-sample l1 norm and step size, shaped for broadcasting [B, 1, 1, ...]
    l1_norms = X_orig.abs().flatten(1).sum(1).clamp(min=1e-8)   # [B]
    view = (-1,) + (1,) * (X.ndim - 1)                          # (B, 1, 1, ...)
    alphas = (budgets / 100.0) * l1_norms / (3 * X_orig[0].numel())
    alphas = alphas.view(view)                                   # [B, 1, 1, ...]
    budgets_v = budgets.view(view)                               # [B, 1, 1, ...]

    loss_fn = nn.MSELoss(reduction="none")
    X_adv = X_orig.clone()

    for _ in range(num_iter):
        X_adv = X_adv.detach().requires_grad_(True)
        # per-element loss, summed over non-batch dims -> [B]
        loss = loss_fn(model(X_adv), X_adv).flatten(1).sum(1)
        loss.sum().backward()

        with torch.no_grad():
            grad = X_adv.grad
            assert grad is not None, "gradient is None — ensure requires_grad=True and loss.backward() was called"
            X_adv = X_adv - alphas * grad
            delta = X_adv - X_orig
            changes = (delta.abs().flatten(1).sum(1) * 100.0 / l1_norms).view(view)
            over = changes > budgets_v
            scale = (budgets_v / changes.clamp(min=1e-8))
            X_adv = torch.where(over, X_orig + delta * scale, X_adv)

    return X_adv.detach()


# batch scoring
def _batch_score(model: nn.Module, X: Tensor) -> Tensor:
    """
    Returns per-sample anomaly score (mean MSE) as a 1-D tensor [B].
    """
    with torch.no_grad():
        X_rec = model(X)
        dims = list(range(1, X.ndim))
        return (X - X_rec).pow(2).mean(dim=dims)


# main function
def compute_perturbation_budget(
    model: nn.Module,
    datamodule,
    threshold: float,
    defense_folder: Path,
    n_samples: int = 100,
    budget_high: float = 200.0,
    n_iter_search: int = 10,
    n_iter_attack: int = 5,
    tol: float = 0.5,
    seed: int = 42,
) -> dict:
    """
    Computes the minimum L1 perturbation budget needed to fool the anomaly
    detector for n_samples anomalous test samples, via batched binary search.

    All samples are processed in parallel at each binary search step:
    a single batched GPU call replaces the sequential per-sample loop.
    Each sample has its own budget tensor, so per-sample alpha scaling is
    preserved exactly as in the original l1_attack_budget.

    Args:
        model:          Raw autoencoder nn.Module (lit_model.model).
        datamodule:     Already set-up DataModule.
        threshold:      tau_95 from the clean test run (p95 of clean scores).
        defense_folder: Output folder for this defense variant (def_off/def_on).
        n_samples:      Number of anomalous samples to evaluate (default: 100).
        budget_high:    Maximum L1 budget to search (% of input L1 norm).
        n_iter_search:  Binary search steps (default: 10, sufficient for tol=0.5).
        n_iter_attack:  PGD steps per evaluation (default: 5).
        tol:            Convergence tolerance on budget in % (default: 0.5).
        seed:           Random seed for sample selection.

    Returns:
        Summary dict with aggregate statistics (also written to CSV).
    """
    device = next(model.parameters()).device
    torch.manual_seed(seed)
    np.random.seed(seed)

    # collect anomalous samples
    anomaly_samples: list[Tensor] = []
    for x_batch, y_batch in datamodule.test_dataloader():
        for i in (y_batch == 1).nonzero(as_tuple=True)[0]:
            anomaly_samples.append(x_batch[i].unsqueeze(0).to(device))
        if len(anomaly_samples) >= n_samples:
            break
    anomaly_samples = anomaly_samples[:n_samples]
    n_total = len(anomaly_samples)

    # stack into a single batch [B, ...]
    X = torch.cat(anomaly_samples, dim=0)  # [B, C, F, W]

    print(
        f"\n[perturbation_budget] {n_total} samples | "
        f"threshold={threshold:.6f} | budget_high={budget_high} | "
        f"n_iter_search={n_iter_search} | n_iter_attack={n_iter_attack}"
    )

    # check feasibility at budget_high for all samples at once
    model.eval()
    budgets_high = torch.full((n_total,), budget_high, device=device)
    X_adv_max = _l1_attack_budget_batched(model, X.clone(), budgets_high, num_iter=n_iter_attack)
    scores_max = _batch_score(model, X_adv_max)                  # [N]
    foolable = scores_max < threshold                            # [N] bool

    n_foolable = int(foolable.sum().item())
    print(f"[perturbation_budget] {n_foolable}/{n_total} samples foolable within budget_high={budget_high}%")

    # binary search bounds — hi[i] is always a sufficient budget for sample i
    lo = torch.zeros(n_total, device=device)
    hi = torch.full((n_total,), budget_high, device=device)

    # batched binary search
    pbar = tqdm(range(n_iter_search), desc="[perturbation_budget] searching", unit="step")
    for _ in pbar:
        # active = foolable and not yet converged
        active = foolable & ((hi - lo) >= tol)
        if not active.any():
            break

        active_idx = active.nonzero(as_tuple=True)[0]            # [n_active]
        mid = (lo + hi) / 2.0                                    # [N]

        # single batched GPU call over all active samples
        X_active = X[active_idx]                                 # [n_active, ...]
        mid_active = mid[active_idx]                             # [n_active]
        X_adv_active = _l1_attack_budget_batched(
            model, X_active.clone(), mid_active, num_iter=n_iter_attack
        )
        scores_active = _batch_score(model, X_adv_active)        # [n_active]

        # scatter scores back and update bounds
        scores_full = torch.zeros(n_total, device=device)
        scores_full[active_idx] = scores_active
        succeeded = scores_full < threshold                      # [N]

        hi = torch.where(active & succeeded, mid, hi)
        lo = torch.where(active & ~succeeded, mid, lo)

        n_active = int(active.sum().item())
        n_converged = n_foolable - int(((hi - lo)[foolable] >= tol).sum().item())
        pbar.set_postfix(active=n_active, converged=f"{n_converged}/{n_foolable}")

    # collect per-sample results
    rows: list[dict] = []
    budgets: list[float] = []

    for i in range(n_total):
        if not foolable[i].item():
            rows.append({"sample_idx": i, "min_budget_pct": float("nan"), "fooled": 0})
        else:
            b = hi[i].item()
            budgets.append(b)
            rows.append({"sample_idx": i, "min_budget_pct": b, "fooled": 1})

    # summary statistics
    arr = np.array(budgets) if budgets else np.array([float("nan")])
    n_fooled = len(budgets)
    summary: dict = {
        "sample_idx": "SUMMARY",
        "n_samples": n_total,
        "n_fooled": n_fooled,
        "n_not_fooled": n_total - n_fooled,
        "fooled_ratio": float(n_fooled / n_total),
        "budget_mean_pct": float(np.nanmean(arr)),
        "budget_median_pct": float(np.nanmedian(arr)),
        "budget_p25_pct": float(np.nanpercentile(arr, 25)),
        "budget_p75_pct": float(np.nanpercentile(arr, 75)),
        "budget_high_pct": budget_high,
        "threshold_tau95": threshold,
    }

    print(
        f"[perturbation_budget] fooled_ratio={summary['fooled_ratio']:.3f} | "
        f"median={summary['budget_median_pct']:.2f}% | "
        f"mean={summary['budget_mean_pct']:.2f}%"
    )

    rows.append(summary)
    out_dir = defense_folder / "perturbation_budget"
    write_metrics_csv(rows, out_dir, filename="perturbation_budget.csv")
    print(f"[perturbation_budget] Saved to {out_dir / 'perturbation_budget.csv'}")

    return summary