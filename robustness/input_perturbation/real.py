import torch

def random_adversarial_attack(
    model: torch.nn.Module,
    x: torch.Tensor,
    num_vectors: int = 50,
    noise_std: float = 0.01,
) -> torch.Tensor:
    loss_fn = torch.nn.SmoothL1Loss(reduction="none")
    x_adv = x.detach().clone()
    B = x_adv.shape[0]
    extra_dims = tuple(range(1, x_adv.dim()))

    with torch.no_grad():
        x_rec = model(x_adv)
        current_losses = loss_fn(x_rec, x_adv).mean(dim=extra_dims)  # (B,)

        for _ in range(num_vectors):
            noise = torch.randn_like(x_adv) * noise_std
            x_pert = x_adv + noise
            x_rec_pert = model(x_pert)
            pert_losses = loss_fn(x_rec_pert, x_pert).mean(dim=extra_dims)

            improved = pert_losses < current_losses
            improved_exp = improved.view(B, *([1] * (x_adv.dim() - 1)))
            current_losses = torch.where(improved, pert_losses, current_losses)
            x_adv = torch.where(improved_exp, x_pert, x_adv)

    return x_adv

