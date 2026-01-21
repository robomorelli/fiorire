"""
Compute and display comprehensive reconstruction metrics.
"""

import torch
import yaml
import argparse
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

from utils.load_model import get_model


def compute_metrics_from_checkpoint(checkpoint_path, config):
    """Compute comprehensive metrics including ROC, AUC, F1."""

    # Load checkpoint
    print(f"\n📦 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    cfg = checkpoint['cfg']

    # Get metric dataset path
    metric_dataset_path = config.get('metric_dataset_path') or checkpoint.get('metric_dataset_path')
    if not metric_dataset_path:
        raise ValueError("metric_dataset_path not found!")

    # Load dataset
    print(f"\n📊 Loading metric dataset: {metric_dataset_path}")
    saved_dict = torch.load(metric_dataset_path, map_location='cpu')
    metric_dataset = saved_dict['dataset']

    metric_loader = DataLoader(
        metric_dataset,
        batch_size=cfg.opt.get('batch_size', 32),
        shuffle=False
    )

    # Load model
    print(f"\n🤖 Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(cfg).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Compute errors
    print(f"\n🔍 Computing reconstruction errors...")
    all_errors = []
    all_labels = []

    use_error = cfg.opt.get('use_error', 'abs')

    with torch.no_grad():
        for batch in metric_loader:
            x, target, mask, *_ = batch
            x = x.to(device)
            target = target.to(device)

            # Reconstruction
            recon = model(x)

            # Compute error
            if use_error == 'abs':
                err = torch.abs(target - recon)
            else:
                err = (target - recon) ** 2

            # Mean error per timestep per feature
            err_mean = err.mean(dim=1)  # [B, T]

            # Labels (1 if anomaly, 0 if normal)
            labels = (mask.view(mask.size(0), -1).sum(dim=1) > 0).int()

            all_errors.append(err_mean.cpu())
            all_labels.append(labels.cpu())

    # Concatenate
    all_errors = torch.cat(all_errors, dim=0)  # [N, T]
    all_labels = torch.cat(all_labels, dim=0)  # [N]

    # Flatten for ROC computation
    flat_errors = all_errors.flatten().numpy()
    flat_labels = all_labels.repeat_interleave(all_errors.shape[1]).numpy()

    # Compute ROC
    print(f"\n📈 Computing ROC/AUC...")
    fpr, tpr, thresholds = roc_curve(flat_labels, flat_errors)
    roc_auc = auc(fpr, tpr)

    # Find optimal threshold (Youden's index)
    youden_index = tpr - fpr
    optimal_idx = np.argmax(youden_index)
    optimal_threshold = thresholds[optimal_idx]
    optimal_tpr = tpr[optimal_idx]
    optimal_fpr = fpr[optimal_idx]

    # Compute F1 at optimal threshold
    n_pos = flat_labels.sum()
    n_neg = len(flat_labels) - n_pos

    precision = (optimal_tpr * n_pos) / (optimal_tpr * n_pos + optimal_fpr * n_neg + 1e-12)
    recall = optimal_tpr
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-12)

    # Print results
    print(f"\n{'=' * 60}")
    print(f"RECONSTRUCTION METRICS")
    print(f"{'=' * 60}")
    print(f"ROC AUC:           {roc_auc:.4f}")
    print(f"Optimal Threshold: {optimal_threshold:.6f}")
    print(f"TPR (Recall):      {optimal_tpr:.4f}")
    print(f"FPR:               {optimal_fpr:.4f}")
    print(f"Precision:         {precision:.4f}")
    print(f"F1 Score:          {f1_score:.4f}")
    print(f"{'=' * 60}\n")

    # Compute error statistics
    normal_mask = all_labels == 0
    anom_mask = all_labels == 1

    normal_errors = all_errors[normal_mask].flatten().numpy()
    anom_errors = all_errors[anom_mask].flatten().numpy()

    print(f"ERROR STATISTICS:")
    print(f"{'=' * 60}")
    print(f"Normal sequences:")
    print(f"  Mean:   {normal_errors.mean():.6f}")
    print(f"  Std:    {normal_errors.std():.6f}")
    print(f"  Median: {np.median(normal_errors):.6f}")
    print(f"  Q95:    {np.quantile(normal_errors, 0.95):.6f}")
    print(f"\nAnomalous sequences:")
    print(f"  Mean:   {anom_errors.mean():.6f}")
    print(f"  Std:    {anom_errors.std():.6f}")
    print(f"  Median: {np.median(anom_errors):.6f}")
    print(f"  Q95:    {np.quantile(anom_errors, 0.95):.6f}")
    print(f"{'=' * 60}\n")

    # Create plots
    output_dir = Path(checkpoint_path).parent / "metrics"
    output_dir.mkdir(exist_ok=True)

    # ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.scatter([optimal_fpr], [optimal_tpr], color='red', s=100, zorder=5,
                label=f'Optimal (TPR={optimal_tpr:.3f}, FPR={optimal_fpr:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Anomaly Detection')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_curve.png', dpi=150)
    print(f"✓ ROC curve saved: {output_dir / 'roc_curve.png'}")

    # Error distributions
    plt.figure(figsize=(10, 6))
    plt.hist(normal_errors, bins=50, alpha=0.6, label='Normal', density=True, color='blue')
    plt.hist(anom_errors, bins=50, alpha=0.6, label='Anomalous', density=True, color='red')
    plt.axvline(optimal_threshold, color='green', linestyle='--', linewidth=2,
                label=f'Threshold={optimal_threshold:.4f}')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Density')
    plt.title('Error Distribution - Normal vs Anomalous')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'error_distribution.png', dpi=150)
    print(f"✓ Error distribution saved: {output_dir / 'error_distribution.png'}")

    plt.close('all')

    return {
        'roc_auc': roc_auc,
        'optimal_threshold': optimal_threshold,
        'optimal_tpr': optimal_tpr,
        'optimal_fpr': optimal_fpr,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }


def main():
    parser = argparse.ArgumentParser(description='Compute comprehensive metrics from checkpoint')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config YAML file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    checkpoint_path = config['checkpoint_path']

    metrics = compute_metrics_from_checkpoint(checkpoint_path, config)

    print(f"\n✅ Metrics computation complete!")


if __name__ == "__main__":
    main()