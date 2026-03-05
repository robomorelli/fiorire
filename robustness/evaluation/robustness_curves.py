from pathlib import Path
import matplotlib.pyplot as plt


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
            ax.axhline(anom_metrics[metric], linestyle=":", color="red", label="perturbed baseline")

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
