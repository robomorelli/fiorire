from pathlib import Path
from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt
import fire



def compute_auc(x, y):
    order = np.argsort(x)
    return float(np.trapezoid(y[order], x[order]))


def load_npz(path):
    data = np.load(path, allow_pickle=True)
    return dict(data)


def make_label(npz_path: Path) -> str:
    parts = npz_path.parts

    def_idx = next(i for i, p in enumerate(parts) if p in ("def_off", "def_on"))

    name_exp = parts[def_idx - 2]
    run_name = parts[def_idx - 1].replace("test_", "")
    def_flag = parts[def_idx]

    # tutto quello che c'è tra def_flag e curves.npz
    suffix_parts = parts[def_idx + 1 : -1]  # esclude "curves.npz"
    suffix = "/".join(suffix_parts)

    return f"{name_exp}/{run_name}/{def_flag}/{suffix}"


def collect_npz(exp_paths: list[Path]) -> list[tuple[str, Path]]:
    """Trova tutti i curves.npz nelle exp_paths e restituisce (label, path)."""
    entries = []
    for exp_path in exp_paths:
        for npz in sorted(exp_path.rglob("curves.npz")):
            label = make_label(npz)
            entries.append((label, npz))
    return entries


def plot_compare(*exp_dirs: str, out_dir: str = ""):
    """
    Trova tutti i curves.npz nelle cartelle fornite e li plotta a confronto.

    Uso:
        python script.py /path/AOC /path/MGM
        python script.py /path/AOC --defense_flags=def_off  # filtra per path
        python script.py /path/AOC --out_dir=/path/output
    """
    exp_paths = [Path(d) for d in exp_dirs]
    for p in exp_paths:
        if not p.exists():
            raise ValueError(f"Directory non trovata: {p}")

    out_path = Path(out_dir) if out_dir else exp_paths[0] / "comparison_plots"
    out_path.mkdir(parents=True, exist_ok=True)

    entries = collect_npz(exp_paths)

    if not entries:
        print("Nessun curves.npz trovato.")
        return

    print(f"Trovati {len(entries)} curves.npz:")
    for label, _ in entries:
        print(f"  {label}")

    colors = plt.cm.tab10.colors  # type: ignore

    fig_roc, ax_roc = plt.subplots(figsize=(10, 6))
    fig_pr,  ax_pr  = plt.subplots(figsize=(10, 6))

    for i, (label, npz_path) in enumerate(entries):
        data = load_npz(npz_path)
        color = colors[i % len(colors)]
        # distingui clean (linea piena), perturbed (tratteggio), sweep (punto-linea)
        if "clean" in label:
            ls = "-"
        elif "perturbed" in label and "perturbations" not in label:
            ls = "--"
        else:
            ls = "-."

        if "fpr" in data and "tpr" in data:
            auc_val = compute_auc(data["fpr"], data["tpr"])
            ax_roc.plot(data["fpr"], data["tpr"], color=color, linestyle=ls,
                        label=f"{label} ({auc_val:.3f})")

        if "precision" in data and "recall" in data:
            auc_val = compute_auc(data["recall"], data["precision"])
            ax_pr.plot(data["recall"], data["precision"], color=color, linestyle=ls,
                       label=f"{label} ({auc_val:.3f})")

    for ax, title, xlabel, ylabel, fname in [
        (ax_roc, "ROC comparison", "FPR", "TPR", "roc_compare.png"),
        (ax_pr,  "PR comparison",  "Recall", "Precision", "pr_compare.png"),
    ]:
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, 1)      # float, float invece di lista
        ax.set_ylim(0, 1)
        ax.grid(True)
        ax.legend(fontsize=6)
        fig = ax.get_figure()  # get_figure() invece di ax.figure
        assert fig is not None
        assert isinstance(fig, Figure)
        fig.tight_layout()
        fig.savefig(out_path / fname, dpi=150)



if __name__ == "__main__":
    fire.Fire(plot_compare)