import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from robustness.dataset.data_types import Config
from models.conv_ae2D import CONV_AE2D
from scheduler import build_scheduler
from metrics import compute_metrics
from defenses import approximate_projection, apply_feature_weighting

class LitAutoEncoder(pl.LightningModule):
    def __init__(self, cfg: Config):
        super().__init__()

        # salva config nel checkpoint
        self.save_hyperparameters(cfg)

        self.cfg = cfg
        self.model = CONV_AE2D(cfg)
        self.lr = cfg.opt.lr
        self.epsilon = cfg.defense.epsilon
        self.lambda_latent = cfg.defense.lambda_latent
        self.p_adv = cfg.defense.p_adv

        # buffer per errori di training
        self.train_feat_errors = []
        # inizializzati via checkpoint
        self.train_feat_median = None

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x: torch.Tensor = batch                      # [B, 1, F, W]
        B = x.size(0)

        epsilon = self.cfg.defense.epsilon
        lambda_latent = self.cfg.defense.lambda_latent
        p_adv = self.cfg.defense.p_adv

        # split the batch
        n_adv = int(p_adv * B)
        n_clean = B - n_adv
        x_clean = x[:n_clean]
        x_adv_src = x[n_clean:]

        # clean
        x_hat_clean = self(x_clean)
        recon_loss = F.mse_loss(x_hat_clean, x_clean)
        # feature-wise statistics
        feat_err = (x_hat_clean - x_clean).pow(2).mean(dim=(0, 1, 3))
        self.train_feat_errors.append(feat_err.detach().cpu())

        # FGSM adversarial generation
        if n_adv > 0:
            x_adv = x_adv_src.detach().clone()
            x_adv.requires_grad_(True)

            x_hat_adv = self(x_adv)
            adv_recon_loss = F.mse_loss(x_hat_adv, x_adv)

            grad_x = torch.autograd.grad(
                adv_recon_loss,
                x_adv,
                retain_graph=False,
                create_graph=False,
            )[0]

            x_adv = x_adv + epsilon * grad_x.sign()
            x_adv = x_adv.detach()   # IMPORTANT

            # latent consistency
            z_clean = self.model.encoder(x_adv_src).detach()
            z_adv = self.model.encoder(x_adv)

            latent_loss = F.mse_loss(z_adv, z_clean)
        else:
            latent_loss = torch.tensor(0.0, device=self.device)
        
        # total loss
        loss = recon_loss + lambda_latent * latent_loss

        self.log_dict(
            {
                "train_loss": loss,
                "train_recon_loss": recon_loss,
                "train_latent_loss": latent_loss,
            },
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        return loss
    
    def on_train_epoch_end(self):
        if len(self.train_feat_errors) == 0:
            return

        self.train_feat_median = torch.stack(
            self.train_feat_errors
        ).median(dim=0).values

        # reset per epoca successiva
        self.train_feat_errors.clear()

    def validation_step(self, batch, batch_idx):
        x = batch
        x_hat = self(x)
        metrics = {}
        if self.cfg.metrics.compute_validation:
            metrics = compute_metrics(x, x_hat, self.cfg.metrics.types)
        loss = F.mse_loss(x_hat, x)
        self.log("val_loss", loss, prog_bar=True)
        for k, v in metrics.items():
            self.log(f"val_{k}", v, prog_bar=True)
        return metrics
    
    def test_step(self, batch, batch_idx):
        x: torch.Tensor = batch
        perturb = self.cfg.metrics.perturb_test

        if perturb:
            B = x.size(0)
            n_adv = int(self.cfg.metrics.perturb_fraction * B)

            x_adv = x[:n_adv].detach().clone()
            x_real = x[n_adv:].detach().clone()

            # --- adversarial perturbation ---
            x_adv.requires_grad_(True)
            x_hat_adv = self(x_adv)
            adv_loss = F.mse_loss(x_hat_adv, x_adv)
            grad_x = torch.autograd.grad(adv_loss, x_adv)[0]
            x_adv = x_adv + self.epsilon * grad_x.sign()
            x_adv = x_adv.detach()

            # --- real perturbation (50% batch) ---
            real_params = self.cfg.metrics.real_noise_params
            perturb_types = ["gaussian", "dropout", "impulse"]
            # per ogni sample scegli un tipo casuale
            for i in range(x_real.size(0)):
                choice_idx = int(torch.randint(0, len(perturb_types), (1,)).item())
                choice = perturb_types[choice_idx]
                if choice == "gaussian":
                    x_real[i] += real_params.gaussian_std * torch.randn_like(x_real[i])
                elif choice == "dropout":
                    mask = torch.rand_like(x_real[i]) < real_params.dropout_prob
                    x_real[i][mask] = 0.0
                elif choice == "impulse":
                    x_real[i] += real_params.impulse_std * torch.randn_like(x_real[i])

            # ricombina il batch perturbato
            x = torch.cat([x_adv, x_real], dim=0)

        # --- reconstruction + feature weighting ---
        x_rec, _ = approximate_projection(
            encoder=self.model.encoder,
            decoder=self.model.decoder,
            x=x,
            alpha=self.cfg.defense.alpha,
            num_iter=self.cfg.defense.num_iter,
        )

        rec_err_feat = (x_rec - x).pow(2).mean(dim=(1, 3))
        rec_err = apply_feature_weighting(rec_err_feat, self.train_feat_median, epsilon=1e-4, batch_idx=batch_idx)

        # log reconstruction error separatamente
        self.log("test_rec_error", rec_err.mean(), prog_bar=True, on_epoch=True)

        # --- metriche ---
        metrics = {}
        if self.cfg.metrics.compute_test:
            metrics = compute_metrics(x, x_rec, self.cfg.metrics.types)
            for k, v in metrics.items():
                self.log(f"test_{k}", v, prog_bar=True)

        return metrics


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        # restituisce direttamente il dict pronto per Lightning
        scheduler_dict = build_scheduler(optimizer, self.cfg.opt)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_dict,
        }
    
    # called when a lightning checkpoint is saved or loaded
    def on_save_checkpoint(self, checkpoint):
        if hasattr(self, "train_feat_median"):
            checkpoint["train_feat_median"] = self.train_feat_median

    def on_load_checkpoint(self, checkpoint):
        self.train_feat_median = checkpoint.get("train_feat_median", None)
