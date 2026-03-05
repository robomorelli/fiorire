import math
import numpy as np
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F


def reconstruction_loss(x: Tensor, x_hat: Tensor, dimensions: list[int], reduction: str = "mean"):
    err = (x_hat - x).pow(2)          # [B,1,F,W]
    err = err.squeeze(1)              # [B,F,W]
    if (np.array(dimensions)>2).any():
        raise ValueError("Too much dimensions to compute mean. Data shape: [B, F, W].")
    err = err.mean(dim=dimensions)    # media su dimensions 0,1,2 → [B,...]
    return err


def feature_errors(x: Tensor, x_hat: Tensor):
    """
    Returns per-sample per-feature errors, WITHOUT averaging over batch.
    Shape input:  [B, C, F, W]
    Shape output: [B, F]
    """
    err = (x_hat - x).pow(2)          # [B,1,F,T]
    err = err.squeeze(1)              # [B,F,T]
    err = err.median(dim=2).values    # median over time
    return err                        # [B,F]


class LipschitzEMAController:
    """
    Controller per aggiornare dinamicamente il coefficiente lambda_init
    basato sulla norma Lipschitz stimata con EMA.

    Attributes:
        lambda_init (float): Valore corrente di lambda per la regolarizzazione latente.
        target_norm (float): Norma target desiderata.
        ema_decay (float): Fattore di decadimento dell'EMA.
        lr (float): Learning rate per l'aggiornamento esponenziale di lambda.
        lambda_min (float): Valore minimo di lambda_init.
        lambda_max (float): Valore massimo di lambda_init.
        ema_norm (Optional[Tensor]): EMA corrente della norma calcolata.
    """

    def __init__(
        self,
        lambda_init: float = 1e-4,
        target_norm: float = 1.0,
        ema_decay: float = 0.95,
        lr: float = 0.01,
        lambda_min: float = 1e-6,
        lambda_max: float = 1e-2,
        ema_norm: float | None = None
    ) -> None:
        self.lambda_init = lambda_init
        self.target_norm = target_norm
        self.ema_decay = ema_decay
        self.lr = lr
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.ema_norm = ema_norm

    def update(self, norm: Tensor) -> tuple[float, float]:
        norm = norm.detach()
        if self.ema_norm is None:
            self.ema_norm = norm.item()
        else:
            self.ema_norm = self.ema_decay * self.ema_norm + (1 - self.ema_decay) * norm.item()

        error = self.ema_norm - self.target_norm
        self.lambda_init *= math.exp(self.lr * error)
        self.lambda_init = max(self.lambda_min, min(self.lambda_init, self.lambda_max))

        return self.lambda_init, self.ema_norm 
    
class RatioEMAController:
    """
    Controlla lambda dinamicamente per portare il ratio
    (lambda * jac_loss / recon_loss) verso target_ratio.
    
    Invece di fissare una target norm dello jacobiano, fissa direttamente
    il contributo relativo della regolarizzazione rispetto alla recon loss.
    """

    def __init__(
        self,
        lambda_init: float = 1e-4,
        target_ratio: float = 0.1,   # es. jac_contrib = 10% di recon_loss
        ema_decay: float = 0.95,
        lr: float = 0.01,
        lambda_min: float = 1e-6,
        lambda_max: float = 1.0,     # range più ampio: il ratio scala recon_loss
    ) -> None:
        self.lambda_init = lambda_init
        self.target_ratio = target_ratio
        self.ema_decay = ema_decay
        self.lr = lr
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.ema_ratio: float | None = None

    def update(self, jac_loss: Tensor, recon_loss: Tensor) -> tuple[float, float]:
        """
        Args:
            jac_loss:   ||J||^2 (scalar tensor)
            recon_loss: reconstruction loss (scalar tensor)
        Returns:
            (lambda_init, ema_ratio)
        """
        current_ratio = (self.lambda_init * jac_loss / (recon_loss + 1e-8)).item()

        if self.ema_ratio is None:
            self.ema_ratio = current_ratio
        else:
            self.ema_ratio = self.ema_decay * self.ema_ratio + (1 - self.ema_decay) * current_ratio

        # errore sul ratio (in log space per stabilità)
        error = math.log(self.ema_ratio + 1e-8) - math.log(self.target_ratio + 1e-8)
        self.lambda_init *= math.exp(-self.lr * error)
        self.lambda_init = max(self.lambda_min, min(self.lambda_init, self.lambda_max))

        return self.lambda_init, self.ema_ratio


def compute_jacobian_norm(encoder: nn.Module, x: Tensor) -> Tensor:
    """
    Calcola la norma media della Jacobiana dell'encoder rispetto all'input x.
    Questa funzione è utile per misurare la sensibilità locale del modello.

    Args:
        encoder (torch.nn.Module): Encoder del modello.
        x (Tensor): Input batch (B, C, H, W) con requires_grad=True.

    Returns:
        Tensor: Norma media della Jacobiana per il batch (scalar tensor).
    """
    z = encoder(x)
    
    # v ~ N(0,1) normalizzato per-sample come nel paper
    v = torch.randn_like(z)
    v = v / (v.flatten(1).norm(dim=1, keepdim=True).reshape(-1, *([1]*(v.dim()-1))) + 1e-8)
    
    JTv = torch.autograd.grad(
        outputs=z,
        inputs=x,
        grad_outputs=v,
        retain_graph=True,
        create_graph=True,
    )[0]
    
    # stima norma di Frobenius: n_out * ||J^T v||^2, media sul batch
    n_out = z.flatten(1).shape[1]
    frob_sq = n_out * JTv.flatten(1).norm(dim=1).pow(2)
    norm = frob_sq.mean().sqrt()
    return norm
