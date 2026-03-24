import torch
import warnings
from typing import Optional
from torch import Tensor
from models.conv_ae2D import Encoder, Decoder

from robustness.lightning_module.losses import reconstruction_loss


def approximate_projection(
    encoder: Encoder,
    decoder: Decoder,
    x: Tensor,
    alpha: float,
    num_iter: int,
    loss_fn: Optional[torch.nn.Module] = None,
):
    if loss_fn is None:
        loss_fn = torch.nn.MSELoss(reduction="sum")
 
    z = encoder(x).detach()
    z.requires_grad_(True)
 
    # Adam converges in far fewer iterations than vanilla gradient accumulation
    # making it the right choice when num_iter is small (e.g. 10 vs original 1000)
    optimizer = torch.optim.Adam([z], lr=alpha)
 
    x_rec = decoder(z)
    for _ in range(num_iter):
        optimizer.zero_grad()
        x_rec = decoder(z)
        loss = loss_fn(x_rec, x)
        loss.backward()
        optimizer.step()
 
    return x_rec.detach(), z.detach()


def apply_feature_weighting(
    rec_err_feat: Tensor,
    train_feat_median: Optional[Tensor],
    epsilon: float = 1e-5,
    warn: bool = True,
    batch_idx: Optional[int] = None,
):
    """
    Args:
        rec_err_feat: [B, F] or [B, F, W] per-feature reconstruction error
    Returns:
        rec_err: [B] or [B, W] weighted reconstruction error
    """
    if train_feat_median is None:
        if warn and (batch_idx is None or batch_idx == 0):
            warnings.warn(
                "Train feature errors not found: "
                "feature weighting DISABLED during test."
            )
        return rec_err_feat.sum(dim=1)
    weights = 1.0 / (epsilon + train_feat_median.to(rec_err_feat.device))  # [F]
    weights = weights / weights.sum() * rec_err_feat.shape[1]              # [F]
    # works for both [B, F] and [B, F, W] since dim=1 is always F
    return (rec_err_feat * weights.unsqueeze(0).unsqueeze(-1) if rec_err_feat.ndim == 3
            else rec_err_feat * weights).sum(dim=1)


def reconstruct_and_weight(
    encoder: Encoder,
    decoder: Decoder,
    x: Tensor,
    alpha: float,
    num_iter: int,
    train_feat_median: Optional[Tensor],
    label_granularity: str = "sequence",   # <-- aggiunto
    epsilon: float = 1e-4,
    use_feature_weighting: bool = True,
) -> tuple[Tensor, Tensor]:
    """
    Returns:
        x_rec: [B, C, F, W]
        rec_err: [B] if label_granularity=="sequence", [B, W] if "timestamp"
    """
    x_rec, _ = approximate_projection(encoder, decoder, x, alpha, num_iter)
    # sequence → dimensions=[2] → [B, F]
    # timestamp → dimensions=[]  → [B, F, W]  (no reduction on W)
    dimensions = [2] if label_granularity == "sequence" else []
    rec_err_feat = reconstruction_loss(x, x_rec, dimensions=dimensions)
    if use_feature_weighting:
        rec_err = apply_feature_weighting(rec_err_feat, train_feat_median, epsilon)
    else:
        rec_err = rec_err_feat.mean(dim=1)
    return x_rec, rec_err
