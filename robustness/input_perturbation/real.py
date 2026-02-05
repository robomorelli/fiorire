import torch


def gaussian_noise(x: torch.Tensor, std: float) -> torch.Tensor:
    """
    Additive Gaussian noise.
    """
    return x + std * torch.randn_like(x)


def dropout_noise(x: torch.Tensor, p: float) -> torch.Tensor:
    """
    Feature-wise dropout (set to zero with probability p).
    """
    mask = torch.rand_like(x) > p
    return x * mask


def impulse_noise(x: torch.Tensor, std: float) -> torch.Tensor:
    """
    Impulsive noise (Gaussian, but conceptually spike-like).
    """
    return x + std * torch.randn_like(x)


def random_real_perturbation(
    x: torch.Tensor,
    params: dict,
) -> torch.Tensor:
    """
    Apply a random real perturbation to each sample in the batch.
    Each sample independently chooses between:
    - gaussian
    - dropout
    - impulse
    """
    x_pert = x.clone()

    B = x.size(0)
    device = x.device

    # 0 = gaussian, 1 = dropout, 2 = impulse
    choices = torch.randint(0, 3, (B,), device=device)

    for i in range(B):
        if choices[i] == 0:
            x_pert[i] = gaussian_noise(x_pert[i], params["gaussian_std"])
        elif choices[i] == 1:
            x_pert[i] = dropout_noise(x_pert[i], params["dropout_prob"])
        else:
            x_pert[i] = impulse_noise(x_pert[i], params["impulse_std"])

    return x_pert
