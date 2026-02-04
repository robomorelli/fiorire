import torch
import warnings
from typing import Optional
from torch import Tensor

from robustness.input_perturbation.defenses import (
    approximate_projection,
    apply_feature_weighting,
)
from models.conv_ae2D import Encoder, Decoder


def approximate_projection(
    encoder: Encoder,
    decoder: Decoder,
    x: Tensor,
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
        z: Tensor = encoder(x)

    z = z.detach().clone().requires_grad_(True)

    for _ in range(num_iter):
        x_rec = decoder(z)
        loss = loss_fn(x_rec, x)

        loss.backward()

        assert z.grad is not None
        grad = z.grad

        with torch.no_grad():
            z -= alpha * grad
            grad.zero_()

    return x_rec.detach(), z.detach()


def apply_feature_weighting(
    rec_err_feat: Tensor,
    train_feat_median: Optional[Tensor],
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


def reconstruct_and_weight(
    encoder: Encoder,
    decoder: Decoder,
    x: Tensor,
    alpha: float,
    num_iter: int,
    train_feat_median: Optional[Tensor],
    epsilon: float = 1e-4,
) -> tuple[Tensor, Tensor]:
    """
    Apply approximate projection + feature weighting in one shot.

    Args:
        encoder: torch module, maps x -> latent z
        decoder: torch module, maps latent z -> reconstructed x_hat
        x: input batch [B, C, F, W]
        alpha: step size for projection
        num_iter: number of projection steps
        train_feat_median: [F] median of training feature errors (or None)
        epsilon: small constant for numerical stability in weighting

    Returns:
        x_rec: reconstructed batch [B, C, F, W]
        rec_err: weighted reconstruction error [B]
    """
    x_rec, _ = approximate_projection(encoder, decoder, x, alpha, num_iter)
    rec_err_feat = (x_rec - x).pow(2).mean(dim=(1, 3))
    rec_err = apply_feature_weighting(rec_err_feat, train_feat_median, epsilon)
    return x_rec, rec_err
