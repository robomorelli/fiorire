import torch
import warnings
from typing import Optional

from models.conv_ae2D import Encoder, Decoder


def approximate_projection(
    encoder: Encoder,
    decoder: Decoder,
    x: torch.Tensor,
    alpha: float,
    num_iter: int,
    loss_fn: Optional[torch.nn.Module] = None,
):
    """
    Approximate projection in latent space.

    Args:
        encoder: callable, maps x -> z
        decoder: callable, maps z -> x_hat
        x: input batch [B, W, F]
        alpha: step size
        num_iter: number of projection steps
        loss_fn: reconstruction loss (default: SmoothL1 sum)

    Returns:
        x_rec: reconstructed batch after projection
        z: optimized latent codes
    """
    if loss_fn is None:
        loss_fn = torch.nn.SmoothL1Loss(reduction="sum")

    with torch.no_grad():
        z = encoder(x)

    z = z.detach().clone().requires_grad_(True)

    for _ in range(num_iter):
        x_rec = decoder(z)
        loss = loss_fn(x_rec, x)

        loss.backward()
        z.data -= alpha * z.grad
        z.grad.zero_()

    return x_rec.detach(), z.detach()


def apply_feature_weighting(
    rec_err_feat: torch.Tensor,
    train_feat_median: Optional[torch.Tensor],
    epsilon: float = 1e-4,
    warn: bool = True,
    batch_idx: Optional[int] = None,
):
    """
    Apply feature weighting to reconstruction errors.

    Args:
        rec_err_feat: [B, F] per-feature reconstruction error
        train_feat_median: [F] median train feature errors or None
        epsilon: numerical stability term
        warn: whether to emit warning if weighting disabled
        batch_idx: for emitting warning only once

    Returns:
        rec_err: [B] weighted reconstruction error
    """
    if train_feat_median is None:
        if warn and (batch_idx is None or batch_idx == 0):
            warnings.warn(
                "Train feature errors not found: "
                "feature weighting DISABLED during test."
            )
        return rec_err_feat.sum(dim=1)

    weights = 1.0 / (epsilon + train_feat_median.to(rec_err_feat.device))
    return (rec_err_feat * weights).sum(dim=1)
