from typing import Any, Optional
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score, precision_recall_curve, auc, roc_curve
from numpy.typing import NDArray
from torch import Tensor


def compute_metrics(
    x: Optional[Tensor],
    x_rec: Optional[Tensor],
    labels: Optional[NDArray],
    scores: Optional[NDArray],
    metric_types: list[str],
    return_curves: bool = False,
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

    return metrics
