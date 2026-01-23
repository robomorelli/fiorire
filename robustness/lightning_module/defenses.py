import numpy as np
import torch.nn as nn
import torch


def approximate_projection(
    decoder,
    z_init,
    target_x,
    num_iter=100,
    alpha=1e-2,
    loss_fn=None
):
    """
    Approximate projection defense.
    Args:
        decoder: trained decoder (frozen)
        z_init: latent encodings (requires_grad=False)
        target_x: attacked samples
    """
    if loss_fn is None:
        loss_fn = nn.SmoothL1Loss(reduction="sum")

    decoder.eval()

    z = z_init.detach().clone()
    z.requires_grad_(True)

    for _ in range(num_iter):
        x_rec = decoder(z)
        loss = loss_fn(x_rec, target_x)

        loss.backward()

        with torch.no_grad():
            z -= alpha * z.grad
            z.grad.zero_()

    return z.detach(), loss.item()


def compute_feature_weights(train_errors, epsilon=1e-3):
    """
    train_errors: np.ndarray (N, F) o (N, T, F)
    """
    if train_errors.ndim == 3:
        train_errors = train_errors.reshape(-1, train_errors.shape[-1])

    weights = 1.0 / (epsilon + np.median(train_errors, axis=0))
    return weights


def apply_feature_weighting(test_errors, weights):
    """
    test_errors: np.ndarray (N, F) o (N, T, F)
    """
    if test_errors.ndim == 3:
        test_errors = test_errors.reshape(test_errors.shape[0], -1)

    weighted_errors = np.dot(test_errors, weights)
    return weighted_errors