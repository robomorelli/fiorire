from pathlib import Path
from typing import Any, Mapping, Optional
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score, precision_recall_curve, auc, roc_curve
from numpy.typing import NDArray
from torch import Tensor

from robustness.evaluation.write_csv import write_metrics_csv


def robustness_delta(clean: Mapping[str, Any], attacked: Mapping[str, Any]) -> dict[str, Any]:
    """Calcola delta tra metriche clean e attacked."""
    keys = ["roc_auc", "pr_auc", "recall_at_fpr5", "score_separation_d"]
    result: dict[str, Any] = {}
    for k in keys:
        # le chiavi nei risultati del trainer hanno prefissi tipo "def_off/clean_roc_auc"
        # strip_prefix non è disponibile qui, quindi cerchiamo per suffisso
        clean_val = find_metric(clean, k)
        attacked_val = find_metric(attacked, k)
        if clean_val is not None and attacked_val is not None:
            result[f"delta_{k}"] = clean_val - attacked_val
    return result
 
 
def find_metric(metrics: Mapping[str, Any], name: str) -> float | None:
    """Cerca una metrica per suffisso nel dict (gestisce prefissi tipo 'def_off/clean_roc_auc')."""
    for k, v in metrics.items():
        if k.endswith(name) and isinstance(v, float):
            return v
    return 

def save_robustness_summary(
    defense_folder: Path,
    suffix: str,
    clean_metrics: Mapping[str, Any],
    anom_metrics: Mapping[str, Any],
    delta_baseline: dict[str, Any],
    results: dict[str, dict[Any, Any]],
    sweep_deltas: dict[str, dict[Any, Any]],
) -> None:
    rows: list[dict[str, Any]] = []
 
    # riga baseline (clean vs mixed perturbed)
    row_baseline: dict[str, Any] = {
        "suffix": suffix,
        "perturb_type": "baseline",
        "param": "mixed",
    }
    for k, v in clean_metrics.items():
        if isinstance(v, float):
            row_baseline[f"clean_{k}"] = v
    for k, v in anom_metrics.items():
        if isinstance(v, float):
            row_baseline[f"attacked_{k}"] = v
    row_baseline.update(delta_baseline)
    rows.append(row_baseline)
 
    # una riga per ogni punto dello sweep
    for perturb_key, param_dict in sweep_deltas.items():
        for p, deltas in param_dict.items():
            row: dict[str, Any] = {
                "suffix": suffix,
                "perturb_type": perturb_key,
                "param": p,
            }
            for k, v in results[perturb_key][p].items():
                if isinstance(v, float):
                    row[f"attacked_{k}"] = v
            row.update(deltas)
            rows.append(row)
 
    write_metrics_csv(rows, defense_folder, filename="robustness_summary.csv")


def _cohens_d(anom_scores: NDArray, clean_scores: NDArray) -> float:
    na, nc = len(anom_scores), len(clean_scores)
    pooled_var = (
        (na - 1) * anom_scores.std(ddof=1) ** 2
        + (nc - 1) * clean_scores.std(ddof=1) ** 2
    ) / (na + nc - 2 + 1e-8)
    return float(
        (anom_scores.mean() - clean_scores.mean()) / (np.sqrt(pooled_var) + 1e-8)
    )

def _recall_at_target_fpr(fpr: NDArray, tpr: NDArray, target_fpr: float = 0.05) -> float:
    idx = np.searchsorted(fpr, target_fpr)  # primo indice dove fpr >= target
    idx = min(idx, len(tpr) - 1)
    return float(tpr[idx])


def compute_metrics(
    x: Optional[Tensor],
    x_rec: Optional[Tensor],
    labels: Optional[NDArray],
    scores: Optional[NDArray],
    metric_types: list[str],
    return_curves: bool = False,
    attacked_mask: Optional[NDArray] = None,
    clean_threshold: Optional[float] = None,
    target_fpr: float = 0.05,
) -> dict[str, Any]:

    metrics: dict[str, Any] = {}

    # reconstruction
    if x is not None and x_rec is not None:
        y_true_np = x.detach().float().cpu().numpy().flatten()
        y_pred_np = x_rec.detach().float().cpu().numpy().flatten()

        if "mse/mae" in metric_types:
            mse = mean_squared_error(y_true_np, y_pred_np)
            mae = mean_absolute_error(y_true_np, y_pred_np)
            metrics["mse/mae"] = float(mse / (mae + 1e-8))
            metrics["anomaly_score"] = float(mse)
            metrics["mae"] = float(mae)

    
    # detection
    if labels is not None and scores is not None:
        labels = labels.astype(np.int32).flatten()
        scores = scores.astype(np.float32).flatten()

        # ROC
        roc_auc = roc_auc_score(labels, scores)
        metrics["roc_auc"] = float(roc_auc)

        fpr, tpr, roc_thresholds = roc_curve(labels, scores)

        # PR
        precision, recall, pr_thresholds = precision_recall_curve(labels, scores)
        pr_auc = auc(recall, precision)
        metrics["pr_auc"] = float(pr_auc)

        if return_curves:
            metrics["_roc_curve"] = {
                "fpr": fpr,
                "tpr": tpr,
                "thresholds": roc_thresholds,
            }
            metrics["_pr_curve"] = {
                "precision": precision,
                "recall": recall,
                "thresholds": pr_thresholds,
            }

        # Recall@FPR5
        metrics["recall_at_fpr5"] = _recall_at_target_fpr(fpr, tpr, target_fpr)

        # Score separation
        anom_scores = scores[labels == 1]
        clean_scores = scores[labels == 0]
        if len(anom_scores) > 0 and len(clean_scores) > 0:
            metrics["score_separation_d"] = _cohens_d(anom_scores, clean_scores)
            metrics["score_delta_mean"] = float(anom_scores.mean() - clean_scores.mean())

        # p95 clean — sempre calcolato, serve come soglia per ASR nei run successivi
        if len(clean_scores) > 0:
            metrics["score_p95_clean"] = float(np.percentile(clean_scores, 95))

        # Metriche che richiedono attacked_mask
        if attacked_mask is not None:
            attacked_mask = attacked_mask.astype(bool).flatten()
            attacked_anom_mask = attacked_mask & (labels == 1)
            clean_mask = labels == 0

            # Robust ROC/PR AUC — utile solo se non tutti gli anomali sono attaccati
            robust_mask = attacked_anom_mask | clean_mask
            if robust_mask.sum() > 0 and len(np.unique(labels[robust_mask])) == 2:
                metrics["robust_roc_auc"] = float(
                    roc_auc_score(labels[robust_mask], scores[robust_mask])
                )
                r_prec, r_rec, _ = precision_recall_curve(
                    labels[robust_mask], scores[robust_mask]
                )
                metrics["robust_pr_auc"] = float(auc(r_rec, r_prec))

            # ASR
            if clean_threshold is not None and attacked_anom_mask.sum() > 0:
                fooled = scores[attacked_anom_mask] < clean_threshold
                metrics["asr"] = float(fooled.sum() / attacked_anom_mask.sum())

    return metrics
