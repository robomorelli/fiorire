from pathlib import Path
from typing import Any
import csv


def write_test_metrics_csv(test_metrics_epoch: list[dict], out_dir: Path):
    """Scrive le metriche di test batch-wise + aggregato in CSV"""
    if len(test_metrics_epoch) == 0:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "metrics.csv"

    fieldnames = list(test_metrics_epoch[0].keys())

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for m in test_metrics_epoch:
            writer.writerow(m)
