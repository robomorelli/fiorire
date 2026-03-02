from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import fire
from sklearn.metrics import auc

def plot_compare(exp_dir: str):
    """
    Plotta PR e ROC AUC di tutti i run nella stessa cartella exp_dir.
    Cerca dentro clean/ e anom/ ogni run per i file curves.npz.
    """
    exp_path = Path(exp_dir)
    run_dirs = [d for d in exp_path.iterdir() if d.is_dir()]

    if not run_dirs:
        raise ValueError("No run directories found")

    metrics = ["pr_auc", "roc_auc"]

    for metric in metrics:
        plt.figure(figsize=(6, 4))

        for run_dir in run_dirs:
            for subdir_name in ["clean", "anom"]:
                subdir = run_dir / subdir_name
                npz_file = subdir / "curves.npz"
                if not npz_file.exists():
                    print(f"{subdir}: curves.npz not found, skipping")
                    continue

                data = np.load(npz_file, allow_pickle=True)

                # prendi la curva e calcola AUC
                if metric == "roc_auc" and "fpr" in data and "tpr" in data:
                    x = data["fpr"]
                    y = data["tpr"]
                    auc_val = auc(x, y)
                elif metric == "pr_auc" and "precision" in data and "recall" in data:
                    x = data["recall"]
                    y = data["precision"]
                    auc_val = auc(x, y)
                else:
                    print(f"{subdir}: required data for {metric} not found, skipping")
                    continue

                # plot singolo, senza doppio plot
                plt.plot(x, y, label=f"{run_dir.name}/{subdir_name} (AUC={auc_val:.3f})")

        plt.xlabel("FPR / Recall")  # usa asse corretto in base alla curva
        plt.ylabel("TPR / Precision")
        plt.title(f"{metric.replace('_', ' ').upper()} robustness comparison")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(exp_path / f"{metric}_compare.png")
        plt.close()

    print("Saved comparison plots for:", ", ".join(metrics))


if __name__ == "__main__":
    fire.Fire(plot_compare)