from pathlib import Path
from typing import Any
import csv


def write_metrics_csv(
    rows: list[dict[str, Any]],
    out_dir: Path,
    filename: str = "metrics.csv",
):
    if not rows:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / filename

    fieldnames = rows[0].keys()

    write_header = not csv_path.exists()

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if write_header:
            writer.writeheader()

        for row in rows:
            writer.writerow(row)

