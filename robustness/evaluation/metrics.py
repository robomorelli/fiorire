import torch
from sklearn.metrics import precision_recall_curve, roc_auc_score, auc, mean_squared_error, mean_absolute_error

def compute_metrics(y_true: torch.Tensor, y_pred: torch.Tensor, metric_types: list) -> dict[str, float]:
    """
    Compute requested metrics:
    - mse/mae ratio
    - mae
    - mse
    - pr_auc
    - roc_auc
    """
    metrics = {}
    y_true_np = y_true.detach().cpu().numpy().flatten()
    y_pred_np = y_pred.detach().cpu().numpy().flatten()

    if "mse" in metric_types:
        mse_val = mean_squared_error(y_true_np, y_pred_np)
        metrics["mse"] = mse_val
    if "mae" in metric_types:
        mae_val = mean_absolute_error(y_true_np, y_pred_np)
        metrics["mae"] = mae_val
    if "mse/mae" in metric_types:
        mse_val = mean_squared_error(y_true_np, y_pred_np)
        mae_val = mean_absolute_error(y_true_np, y_pred_np)
        metrics["mse/mae"] = mse_val / (mae_val + 1e-8)
    if "pr_auc" in metric_types:
        precision, recall, _ = precision_recall_curve(y_true_np, y_pred_np)
        metrics["pr_auc"] = auc(recall, precision)
    if "roc_auc" in metric_types:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true_np, y_pred_np)
        except ValueError:
            metrics["roc_auc"] = float("nan")  # caso single-class

    return metrics


