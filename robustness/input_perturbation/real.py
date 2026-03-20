import torch

def random_adversarial_attack(
    model: torch.nn.Module,
    x: torch.Tensor,
    num_vectors: int = 50,
    noise_std: float = 0.01,
) -> torch.Tensor:
    x_adv = x.detach().clone()
    B = x_adv.shape[0]
    extra_dims = tuple(range(1, x_adv.dim()))

    with torch.no_grad():
        # use MSE consistent with the anomaly detector score
        current_scores = (model(x_adv) - x_adv).pow(2).mean(dim=extra_dims)  # [B]

        for _ in range(num_vectors):
            noise = torch.randn_like(x_adv) * noise_std
            x_pert = x_adv + noise
            pert_scores = (model(x_pert) - x_pert).pow(2).mean(dim=extra_dims)  # [B]

            improved = pert_scores < current_scores
            improved_exp = improved.view(B, *([1] * (x_adv.dim() - 1)))
            current_scores = torch.where(improved, pert_scores, current_scores)
            x_adv = torch.where(improved_exp, x_pert, x_adv)

    return x_adv