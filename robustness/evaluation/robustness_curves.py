from pathlib import Path
import matplotlib.pyplot as plt
from typing import Mapping


def strip_prefix(metrics: Mapping[str, float]) -> dict[str, float]:
    """
    Rimuove prefix e mode lasciando solo il nome della metrica.
    'def_off/clean_rec_error'     -> 'rec_error'
    'def_off/perturbed_roc_auc'   -> 'roc_auc'
    """
    result = {}
    for k, v in metrics.items():
        # rimuovi la parte prima dell'ultimo slash: 'def_off/clean_rec_error' -> 'clean_rec_error'
        after_slash = k.split("/")[-1]
        # rimuovi il mode (prima parola prima di _): 'clean_rec_error' -> 'rec_error'
        metric_name = after_slash.split("_", 1)[1] if "_" in after_slash else after_slash
        result[metric_name] = v
    return result

def plot_robustness_curves(
    clean_metrics: Mapping[str, float],
    anom_metrics: Mapping[str, float],
    results: dict[str, dict[float, dict[str, float]]],
    out_dir: str,
):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    clean_metrics_short = strip_prefix(clean_metrics)
    anom_metrics_short  = strip_prefix(anom_metrics)
    # strip prefix anche sui risultati sweep
    results_short = {
        perturb_type: {p: strip_prefix(m) for p, m in values.items()}
        for perturb_type, values in results.items()
    }

    metrics = list(clean_metrics_short.keys())

    for perturb_type, values in results_short.items():
        if not values:
            continue
        x = sorted(values.keys())
        for metric in metrics:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.axhline(clean_metrics_short[metric], linestyle="--", color="black", label="clean baseline")
            ax.axhline(anom_metrics_short[metric], linestyle=":", color="red", label="perturbed baseline")
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
