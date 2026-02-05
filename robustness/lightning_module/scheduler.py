from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Dict


def build_scheduler(
    optimizer: Optimizer,
    cfg_opt: dict,
    monitor: str = "val_loss",
) -> Dict[str, object]:
    """
    Restituisce un dict pronto da passare a Lightning come lr_scheduler.
    """
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=cfg_opt["lr_patience"],
        factor=cfg_opt["lr_factor"],
        min_lr=cfg_opt["lr_min"],
    )

    return {
        "scheduler": scheduler,
        "monitor": monitor,
        "interval": "epoch",
        "frequency": 1,
    }
