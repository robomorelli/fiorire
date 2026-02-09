from typing import Optional
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score, precision_recall_curve, auc
from numpy.typing import NDArray
from torch import Tensor


def compute_metrics(
    x: Optional[Tensor],
    x_rec: Optional[Tensor],
    labels: Optional[NDArray],
    scores: Optional[NDArray],
    metric_types: list[str],
) -> dict[str, float]:
    """
    Calcola tutte le metriche:
    - reconstruction (x vs x_rec), se fornito
    - detection (labels vs scores), se fornito
    Restituisce un dict unico con tutte le metriche.
    """
    metrics: dict[str, float] = {}

    # ======== RECONSTRUCTION ========
    if x is not None and x_rec is not None:
        y_true_np = x.detach().float().cpu().numpy().flatten()
        y_pred_np = x_rec.detach().float().cpu().numpy().flatten()

        if "mse" in metric_types:
            metrics["mse"] = float(mean_squared_error(y_true_np, y_pred_np))

        if "mae" in metric_types:
            metrics["mae"] = float(mean_absolute_error(y_true_np, y_pred_np))

        if "mse/mae" in metric_types:
            mse = mean_squared_error(y_true_np, y_pred_np)
            mae = mean_absolute_error(y_true_np, y_pred_np)
            metrics["mse/mae"] = float(mse / (mae + 1e-8))

    # ======== DETECTION ========
    if labels is not None and scores is not None:
        labels = labels.astype(np.int32).flatten()
        scores = scores.astype(np.float32).flatten()
        try:
            metrics["roc_auc"] = float(roc_auc_score(labels, scores))
        except ValueError:
            metrics["roc_auc"] = float("nan")

        precision, recall, _ = precision_recall_curve(labels, scores)
        metrics["pr_auc"] = float(auc(recall, precision))

    return metrics
