from pathlib import Path
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from torch import nn

from robustness.input_perturbation.pgd import pgd_attack, adaptive_pgd_steps
from robustness.input_perturbation.real import (
    gaussian_noise,
    dropout_noise,
    impulse_noise,
)


def perturbation_dict(model: nn.Module, cfg: DictConfig) -> dict[str, tuple]:
    """
    Returns a dict:
        name -> (params, perturb_builder)
    where:
        perturb_builder(p) -> Callable[[Tensor], Tensor]
    """
    return {
        "adversarial": (
            cfg["curves"]["pgd_epsilons"],
            lambda eps: lambda x: pgd_attack(
                model,
                x,
                eps,
                alpha=cfg["defense"]["alpha"],
                steps=adaptive_pgd_steps(eps, cfg["defense"]["alpha"]),
            ),
        ),
        "gaussian": (
            cfg["curves"]["gaussian_stds"],
            lambda std: lambda x: gaussian_noise(x, std),
        ),
        "dropout": (
            cfg["curves"]["dropout_probs"],
            lambda p: lambda x: dropout_noise(x, p),
        ),
        "impulse": (
            cfg["curves"]["impulse_stds"],
            lambda std: lambda x: impulse_noise(x, std),
        ),
    }


def plot_robustness_curves(
    clean_metrics: dict[str, float],
    anom_metrics: dict[str, float],
    results: dict[str, dict[float, dict[str, float]]],
    out_dir: str,
):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    metrics = list(clean_metrics.keys())

    for perturb_type, values in results.items():
        if not values:
            continue

        x = sorted(values.keys())

        for metric in metrics:
            fig, ax = plt.subplots(figsize=(7, 4))

            # baseline orizzontali
            ax.axhline(clean_metrics[metric], linestyle="--", color="black", label="clean baseline")
            ax.axhline(anom_metrics[metric], linestyle=":", color="red", label="anom baseline")

            # curva perturbazione
            y = [values[v][metric] for v in x]
            ax.plot(x, y, marker="o", color="steelblue", label=perturb_type)

            ax.set_xlabel("Perturbation intensity")
            ax.set_ylabel(metric)
            ax.set_title(f"{perturb_type} — {metric}")
            ax.legend()
            ax.grid(True)
            fig.tight_layout()
            fig.savefig(out_path / f"{perturb_type}_{metric}.png")
            plt.close(fig)
