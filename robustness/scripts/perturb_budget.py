from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pytorch_lightning as pl

from robustness.evaluation.metrics import write_metrics_csv


def _l1_attack_budget_batched(
    model: nn.Module,
    X: Tensor,
    budgets: Tensor,
    num_iter: int = 5,
) -> Tensor:
    """
    Batched PGD L1 attack with best-iterate tracking.
    Each sample in the batch has its own budget.

    The attack minimizes the reconstruction MSE of X_adv (i.e. makes the
    perturbed sample look normal to the autoencoder) by descending the loss
    ||model(X_adv) - X_adv||^2 w.r.t. X_adv, with the target detached so
    that gradients flow only through model(X_adv).

    Best-iterate tracking returns the X_adv that achieved the lowest anomaly
    score across all iterations, making the attack robust to overshooting.

    Args:
        model:    Raw autoencoder nn.Module.
        X:        Input batch [B, ...].
        budgets:  Per-sample L1 budgets as % of input L1 norm, shape [B].
        num_iter: Number of PGD steps.

    Returns:
        Adversarial batch [B, ...] — best iterate per sample.
    """
    X_orig = X.detach()

    l1_norms = X_orig.abs().flatten(1).sum(1).clamp(min=1e-8)  # [B]
    view = (-1,) + (1,) * (X.ndim - 1)                         # (B, 1, 1, ...)
    budgets_v = budgets.view(view)                              # [B, 1, 1, ...]

    # fixed step size: fraction of L1 norm, independent of budget and num_iter
    # this decouples step size from the projection constraint (budget)
    step = (0.01 * l1_norms).view(view)                        # [B, 1, 1, ...]

    X_adv = X_orig.clone()

    # best-iterate tracking: keep the X_adv with the lowest anomaly score seen
    with torch.no_grad():
        best_scores = _batch_score(model, X_orig)              # [B]
    X_best = X_orig.clone()

    for _ in range(num_iter):
        X_adv = X_adv.detach().requires_grad_(True)

        # loss: how well does the model reconstruct X_adv?
        # target is detached so gradient flows only through model(X_adv)
        rec = model(X_adv)
        loss = ((rec - X_adv.detach()) ** 2).flatten(1).sum(1)
        loss.sum().backward()

        with torch.no_grad():
            grad = X_adv.grad
            assert grad is not None, "gradient is None"

            # normalize gradient to unit L1 norm per sample for stable steps
            grad_norms = grad.abs().flatten(1).sum(1).clamp(min=1e-8).view(view)
            grad_normalized = grad / grad_norms

            # gradient descent: move X_adv toward lower reconstruction error
            X_adv = X_adv - step * grad_normalized

            # project onto L1 ball of radius budgets_v around X_orig
            delta = X_adv - X_orig
            changes = (delta.abs().flatten(1).sum(1) * 100.0 / l1_norms).view(view)
            over = changes > budgets_v
            scale = budgets_v / changes.clamp(min=1e-8)
            X_adv = torch.where(over, X_orig + delta * scale, X_adv)

            # update best iterate per sample
            current_scores = _batch_score(model, X_adv)        # [B]
            improved = current_scores < best_scores             # [B] bool
            improved_v = improved.view(view)
            X_best = torch.where(improved_v, X_adv, X_best)
            best_scores = torch.where(improved, current_scores, best_scores)

    return X_best.detach()


def _batch_score(model: nn.Module, X: Tensor) -> Tensor:
    """Returns per-sample anomaly score (mean MSE) as a 1-D tensor [B]."""
    with torch.no_grad():
        X_rec = model(X)
        dims = list(range(1, X.ndim))
        return (X - X_rec).pow(2).mean(dim=dims)


def _plot_boundary_anomalies(
    model: nn.Module,
    samples: Tensor,
    indices: list[int],
    budgets: list[float],
    out_dir: Path,
    n_plot: int = 5,
) -> None:
    """
    For each of the first n_plot boundary anomalies, produces two plots:
      1. Line plot — one subplot per feature, original vs reconstruction overlaid.
      2. Heatmap — [features x time] side by side: original | reconstruction | residual.

    Args:
        model:    Raw autoencoder nn.Module (eval mode, on device).
        samples:  Tensor of shape [K, C, F, T] — boundary anomaly samples on CPU.
        indices:  List of sample indices (for titles).
        budgets:  List of min_budget_pct values (for titles).
        out_dir:  Directory where plots are saved.
        n_plot:   Number of samples to plot (default: 5).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    n_plot = min(n_plot, len(samples))

    for k in range(n_plot):
        x = samples[k]                        # [C, F, T] or [F, T]
        x_dev = x.unsqueeze(0).to(device)     # [1, ...]

        with torch.no_grad():
            x_rec = model(x_dev).squeeze(0).cpu()

        # collapse channel dim if present: [C, F, T] -> [F, T]
        if x.ndim == 3:
            x_2d = x.mean(0)
            x_rec_2d = x_rec.mean(0)
        else:
            x_2d = x
            x_rec_2d = x_rec

        x_np = x_2d.numpy()
        x_rec_np = x_rec_2d.numpy()
        residual = np.abs(x_np - x_rec_np)

        n_features = x_np.shape[0]
        sample_title = f"sample_idx={indices[k]} | min_budget={budgets[k]:.3f}%"

        # Plot 1: line plot
        fig, axes_arr = plt.subplots(n_features, 1, figsize=(12, 2 * n_features), sharex=True, squeeze=False)
        axes = axes_arr[:, 0].tolist()
        fig.suptitle(f"Line plot — {sample_title}", fontsize=11)

        for f_idx, ax in enumerate(axes):
            ax.plot(x_np[f_idx], label="original", linewidth=1.2)
            ax.plot(x_rec_np[f_idx], label="reconstruction", linewidth=1.2, linestyle="--")
            ax.set_ylabel(f"feat {f_idx}", fontsize=8)
            ax.tick_params(labelsize=7)
            if f_idx == 0:
                ax.legend(fontsize=8, loc="upper right")

        axes[-1].set_xlabel("time step")
        plt.tight_layout()
        fig.savefig(out_dir / f"boundary_{k:02d}_lineplot.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Plot 2: heatmap
        vmin = min(x_np.min(), x_rec_np.min())
        vmax = max(x_np.max(), x_rec_np.max())

        fig = plt.figure(figsize=(15, max(3, n_features * 0.4 + 2)))
        fig.suptitle(f"Heatmap — {sample_title}", fontsize=11)
        gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

        ax_orig = fig.add_subplot(gs[0])
        ax_rec  = fig.add_subplot(gs[1])
        ax_res  = fig.add_subplot(gs[2])

        feature_ticks = list(range(n_features))
        feature_labels = [str(i + 1) for i in range(n_features)]

        im0 = ax_orig.imshow(x_np, aspect="auto", origin="lower", vmin=vmin, vmax=vmax, cmap="viridis")
        ax_orig.set_title("Original", fontsize=10)
        ax_orig.set_xlabel("time step")
        ax_orig.set_ylabel("feature")
        ax_orig.set_yticks(feature_ticks)
        ax_orig.set_yticklabels(feature_labels, fontsize=7)
        plt.colorbar(im0, ax=ax_orig, fraction=0.046, pad=0.04)

        im1 = ax_rec.imshow(x_rec_np, aspect="auto", origin="lower", vmin=vmin, vmax=vmax, cmap="viridis")
        ax_rec.set_title("Reconstruction", fontsize=10)
        ax_rec.set_xlabel("time step")
        ax_rec.set_yticks(feature_ticks)
        ax_rec.set_yticklabels(feature_labels, fontsize=7)
        plt.colorbar(im1, ax=ax_rec, fraction=0.046, pad=0.04)

        im2 = ax_res.imshow(residual, aspect="auto", origin="lower", cmap="Reds")
        ax_res.set_title("|Original − Reconstruction|", fontsize=10)
        ax_res.set_xlabel("time step")
        ax_res.set_yticks(feature_ticks)
        ax_res.set_yticklabels(feature_labels, fontsize=7)
        plt.colorbar(im2, ax=ax_res, fraction=0.046, pad=0.04)

        plt.tight_layout()
        fig.savefig(out_dir / f"boundary_{k:02d}_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"[perturbation_budget] Saved {n_plot * 2} plots to {out_dir}")


def compute_perturbation_budget(
    model: nn.Module,
    datamodule: pl.LightningDataModule,
    threshold: float,
    defense_folder: Path,
    n_samples: int = 100,
    budget_high: float = 200.0,
    n_iter_search: int = 10,
    n_iter_attack: int = 50,
    tol: float = 0.5,
    boundary_tol: float = 1.0,
    n_plot: int = 5,
    seed: int = 42,
) -> dict:
    """
    Computes the minimum L1 perturbation budget needed to fool the anomaly
    detector for n_samples anomalous test samples, via batched binary search.

    The feasibility check has been removed. Instead, all samples are passed
    to the binary search directly. Samples whose hi never moves below
    budget_high - tol are reported as not foolable.

    Args:
        model:          Raw autoencoder nn.Module (lit_model.model).
        datamodule:     Already set-up DataModule.
        threshold:      tau_95 from the clean test run (p95 of clean scores).
        defense_folder: Output folder for this defense variant (def_off/def_on).
        n_samples:      Number of anomalous samples to evaluate (default: 100).
        budget_high:    Maximum L1 budget to search (% of input L1 norm).
        n_iter_search:  Binary search steps (default: 10, sufficient for tol=0.5).
        n_iter_attack:  PGD steps per evaluation (default: 50).
        tol:            Convergence tolerance on budget in % (default: 0.5).
        boundary_tol:   Max budget to classify a sample as near-boundary (default: 1.0%).
        n_plot:         Number of boundary anomalies to plot (default: 5).
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

    X = torch.cat(anomaly_samples, dim=0)  # [N, ...]

    print(
        f"\n[perturbation_budget] {n_total} samples | "
        f"threshold={threshold:.6f} | budget_high={budget_high} | "
        f"n_iter_search={n_iter_search} | n_iter_attack={n_iter_attack}"
    )

    model.eval()

    # no feasibility check — run binary search on all samples
    # samples whose hi never moves down are identified as not foolable at the end
    print(f"[perturbation_budget] skipping feasibility check, running binary search on all {n_total} samples")

    lo = torch.zeros(n_total, device=device)
    hi = torch.full((n_total,), budget_high, device=device)
    active_mask = torch.ones(n_total, dtype=torch.bool, device=device)

    pbar = tqdm(range(n_iter_search), desc="[perturbation_budget] searching", unit="step")
    for _ in pbar:
        active = active_mask & ((hi - lo) >= tol)
        if not active.any():
            break

        active_idx = active.nonzero(as_tuple=True)[0]
        mid = (lo + hi) / 2.0

        X_active = X[active_idx]
        mid_active = mid[active_idx]
        X_adv_active = _l1_attack_budget_batched(
            model, X_active.clone(), mid_active, num_iter=n_iter_attack
        )
        scores_active = _batch_score(model, X_adv_active)

        scores_full = torch.full((n_total,), float("inf"), device=device)
        scores_full[active_idx] = scores_active
        succeeded = scores_full < threshold

        hi = torch.where(active & succeeded, mid, hi)
        lo = torch.where(active & ~succeeded, mid, lo)

        n_active = int(active.sum().item())
        n_converged = int((active_mask & ((hi - lo) < tol)).sum().item())
        pbar.set_postfix(active=n_active, converged=f"{n_converged}/{n_total}")

    # collect per-sample results
    # a sample is foolable iff hi moved strictly below budget_high
    rows: list[dict] = []
    budgets: list[float] = []

    for i in range(n_total):
        b = hi[i].item()
        if b >= budget_high - tol:
            # hi never moved down: attack never succeeded at any probed budget
            rows.append({"sample_idx": i, "min_budget_pct": float("nan"), "fooled": 0})
        else:
            budgets.append(b)
            rows.append({"sample_idx": i, "min_budget_pct": b, "fooled": 1})

    n_fooled = len(budgets)
    print(f"[perturbation_budget] {n_fooled}/{n_total} samples foolable within budget_high={budget_high}%")

    # near-boundary anomalies
    boundary_indices: list[int] = []
    boundary_samples: list[Tensor] = []
    boundary_budgets: list[float] = []

    for i in range(n_total):
        b = hi[i].item()
        if b < budget_high - tol and b <= boundary_tol:
            boundary_indices.append(i)
            boundary_samples.append(X[i].cpu())
            boundary_budgets.append(b)

    out_dir = defense_folder / "perturbation_budget"

    if boundary_indices:
        print(
            f"\n[perturbation_budget] {len(boundary_indices)} near-boundary anomalies "
            f"(min_budget <= {boundary_tol}%): sample indices {boundary_indices}"
        )
        boundary_out = out_dir / "boundary_anomalies"
        boundary_out.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "indices": boundary_indices,
                "budgets": boundary_budgets,
                "samples": torch.stack(boundary_samples),
            },
            boundary_out / "boundary_anomalies.pt",
        )
        print(f"[perturbation_budget] Checkpoint saved to {boundary_out / 'boundary_anomalies.pt'}")

        _plot_boundary_anomalies(
            model=model,
            samples=torch.stack(boundary_samples),
            indices=boundary_indices,
            budgets=boundary_budgets,
            out_dir=boundary_out / "plots",
            n_plot=n_plot,
        )
    else:
        print(f"\n[perturbation_budget] No near-boundary anomalies found (boundary_tol={boundary_tol}%)")

    # summary statistics
    arr = np.array(budgets) if budgets else np.array([float("nan")])
    summary: dict = {
        "sample_idx": "SUMMARY",
        "min_budget_pct": float("nan"),
        "fooled": "",
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
        "n_boundary": len(boundary_indices),
        "boundary_tol_pct": boundary_tol,
    }

    print(
        f"[perturbation_budget] fooled_ratio={summary['fooled_ratio']:.3f} | "
        f"median={summary['budget_median_pct']:.2f}% | "
        f"mean={summary['budget_mean_pct']:.2f}% | "
        f"n_boundary={summary['n_boundary']}"
    )

    rows.append(summary)
    write_metrics_csv(rows, out_dir, filename="perturbation_budget.csv")
    print(f"[perturbation_budget] Saved to {out_dir / 'perturbation_budget.csv'}")

    return summary