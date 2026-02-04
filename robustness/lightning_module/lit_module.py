import pytorch_lightning as pl
import torch

from robustness.dataset.data_types import Config
from models.conv_ae2D import CONV_AE2D
from robustness.evaluation.robustness_curves import (
    plot_robustness_curves,
    build_robustness_curves,
)
from robustness.lightning_module.losses import reconstruction_loss, feature_errors
from scheduler import build_scheduler
from robustness.evaluation.metrics import compute_metrics
from robustness.perturbation.defenses import reconstruct_and_weight
from robustness.perturbation.adv_train import fgsm_attack, latent_consistency_loss
from robustness.perturbation.real import random_real_perturbation


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

        self._clean_metrics = {}
        self._robustness_results = {
            "adversarial": {},
            "gaussian": {},
            "dropout": {},
            "impulse": {},
        }

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x: torch.Tensor = batch  # [B, 1, F, W]
        B = x.size(0)

        # split the batch
        n_adv = int(self.p_adv * B)
        n_clean = B - n_adv
        x_clean = x[:n_clean]
        x_adv_src = x[n_clean:]

        # clean
        x_hat_clean = self(x_clean)
        recon_loss = reconstruction_loss(x_clean, x_hat_clean)
        # feature-wise statistics
        feat_err = feature_errors(x_clean, x_hat_clean)
        self.train_feat_errors.append(feat_err.detach().cpu())

        # FGSM adversarial generation
        if n_adv > 0:
            x_adv = fgsm_attack(self, x_adv_src, self.epsilon)
            x_adv = x_adv.detach()
            latent_loss = latent_consistency_loss(self.model.encoder, x_adv_src, x_adv)
        else:
            latent_loss = torch.tensor(0.0, device=self.device)

        # total loss
        loss = recon_loss + self.lambda_latent * latent_loss

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

        self.train_feat_median = (
            torch.stack(self.train_feat_errors).median(dim=0).values
        )

        # reset per epoca successiva
        self.train_feat_errors.clear()

    def validation_step(self, batch, batch_idx):
        x = batch
        x_hat = self(x)

        loss = reconstruction_loss(x, x_hat)
        self.log("val_loss", loss, prog_bar=True)

        if self.cfg.metrics.compute_validation:
            metrics = compute_metrics(x, x_hat, self.cfg.metrics.types)
            for k, v in metrics.items():
                self.log(f"val_{k}", v, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        x: torch.Tensor = batch
        perturb = self.cfg.metrics.perturb_test
        epsilon_test = self.cfg.metrics.epsilon

        if perturb:
            B = x.size(0)
            n_adv = int(self.cfg.metrics.perturb_fraction * B)

            x_real = x[n_adv:].detach().clone()

            x_adv = fgsm_attack(self, x[:n_adv].detach().clone(), epsilon_test)
            x_real = random_real_perturbation(
                x[n_adv:], self.cfg.metrics.real_noise_params
            )

            # ricombina il batch perturbato
            x = torch.cat([x_adv, x_real], dim=0)

        x_rec, rec_err = reconstruct_and_weight(
            self.model.encoder,
            self.model.decoder,
            x,
            self.cfg.defense.alpha,
            self.cfg.defense.num_iter,
            self.train_feat_median,
        )
        # log reconstruction error separatamente
        self.log("test_rec_error", rec_err.mean(), prog_bar=True, on_epoch=True)

        metrics = {}
        if self.cfg.metrics.compute_test:
            metrics = compute_metrics(x, x_rec, self.cfg.metrics.types)
            for k, v in metrics.items():
                self.log(f"test_{k}", v, prog_bar=True)

        return metrics

    def on_test_start(self):
        assert self.trainer.test_dataloaders is not None
        self._test_dataloader = self.trainer.test_dataloaders[0]

    def on_test_epoch_end(self):
        if not self.cfg.curves.enabled:
            return

        dataloader = self._test_dataloader

        # clean baseline
        self._clean_metrics = self._evaluate_on_loader(dataloader)

        curves = build_robustness_curves(self, self.cfg)
        for name, (params, perturb_builder) in curves.items():
            for p in params:
                self._robustness_results[name][p] = self._evaluate_on_loader(
                    dataloader,
                    perturb_builder(p),
                )

        plot_robustness_curves(
            clean_metrics=self._clean_metrics,
            results=self._robustness_results,
            out_dir=f"{self.cfg.trainer.out_dir}/robustness",
        )

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

    def _evaluate_on_loader(self, dataloader, perturb_fn=None, requires_grad=False):
        self.eval()
        metrics_acc = []

        for batch in dataloader:
            x = batch.to(self.device)

            # applica perturbazione se presente
            if perturb_fn is not None:
                if requires_grad:
                    x = perturb_fn(x)
                else:
                    with torch.no_grad():
                        x = perturb_fn(x)

            # ricostruzione + feature weighting
            with torch.no_grad():
                x_rec, rec_err = reconstruct_and_weight(
                    self.model.encoder,
                    self.model.decoder,
                    x,
                    self.cfg.defense.alpha,
                    self.cfg.defense.num_iter,
                    self.train_feat_median,
                )

            # metriche
            metrics = compute_metrics(x, x_rec, self.cfg.metrics.types)
            metrics["rec_error"] = rec_err.mean().item()  # weighted error media batch
            metrics_acc.append(metrics)

        # media batch
        return {
            k: sum(m[k] for m in metrics_acc) / len(metrics_acc) for k in metrics_acc[0]
        }
