from pathlib import Path
import pandas as pd
import torch

def write_metrics_csv(rows: list[dict], out_dir: Path, filename: str = "metrics.csv"):
    if not rows:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / filename

    def convert(v):
        if isinstance(v, torch.Tensor):
            return v.item()
        return v

    rows = [{k: convert(v) for k, v in row.items()} for row in rows]
    df_new = pd.DataFrame(rows)

    if csv_path.exists():
        df_existing = pd.read_csv(csv_path)
        df = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df = df_new

    df.to_csv(csv_path, index=False)
