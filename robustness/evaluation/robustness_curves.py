from pathlib import Path
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from torch import nn

from robustness.input_perturbation.adv_train_utils import pgd_attack
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
            cfg["curves"]["adversarial_epsilons"],
            lambda steps: lambda alpha: lambda eps: lambda x: pgd_attack(
                model, x, eps, alpha, steps
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
    results: dict[str, dict[float, dict[str, float]]],
    out_dir: str,
):
    """
    results = {
        "adversarial": {epsilon: {metric: value}},
        "gaussian": {std: {...}},
        "dropout": {p: {...}},
        "impulse": {std: {...}},
    }
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    metrics = list(clean_metrics.keys())

    for metric in metrics:
        plt.figure(figsize=(6, 4))

        # clean baseline
        plt.axhline(
            clean_metrics[metric],
            linestyle="--",
            color="black",
            label="clean baseline",
        )

        for perturb_type, values in results.items():
            if not values:
                continue
            x = sorted(values.keys())
            y = [values[v][metric] for v in x]
            plt.plot(x, y, marker="o", label=perturb_type)

        plt.xlabel("Perturbation intensity")
        plt.ylabel(metric)
        plt.title(f"Robustness curve {metric}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_path / f"{metric}_robustness.png")
        plt.close()
