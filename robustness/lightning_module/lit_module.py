from pathlib import Path
import pytorch_lightning as pl
import torch
from typing import Literal

from robustness.dataset.data_types import Config
from models.conv_ae2D import CONV_AE2D
from robustness.evaluation.robustness_curves import (
    plot_robustness_curves,
    build_robustness_curves,
)
from robustness.lightning_module.losses import (
    reconstruction_loss,
    feature_errors,
    regularization_loss,
)
from scheduler import build_scheduler
from robustness.evaluation.metrics import compute_metrics
from robustness.evaluation.write_csv import write_test_metrics_csv
from robustness.input_perturbation.defenses import reconstruct_and_weight
from robustness.input_perturbation.adv_train_utils import pgd_attack
from robustness.input_perturbation.real import random_real_perturbation


class LitAutoEncoder(pl.LightningModule):
    def __init__(self, cfg: Config):
        super().__init__()

        # salva config nel checkpoint
        self.save_hyperparameters(cfg)

        self.cfg = cfg
        self.model = CONV_AE2D(cfg)
        self.lr = cfg.opt.lr
        self.epsilon_train = cfg.defense.epsilon
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

        self.test_mode: Literal["clean", "anom"]
        self._test_metrics_epoch = []  # buffer per le metriche di test

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x: torch.Tensor = batch  # [B, 1, F, W]
        B = x.size(0)

        if self.cfg.defense.adv_training:
            # split the batch
            n_adv = int(self.p_adv * B)
            x_clean = x[:-n_adv]
            x_adv_src = x[-n_adv:]
        else:
            # training normale, tutto "clean"
            n_adv = 0
            x_clean = x

        # clean
        x_hat_clean = self(x_clean)
        recon_loss = reconstruction_loss(x_clean, x_hat_clean)
        # feature-wise statistics
        feat_err = feature_errors(x_clean, x_hat_clean)
        self.train_feat_errors.append(feat_err.detach().cpu())

        latent_loss = torch.tensor(0.0, device=self.device)
        if n_adv > 0:
            x_adv = pgd_attack(
                self,
                x_adv_src,
                epsilon=self.epsilon_train,
                alpha=self.epsilon_train / self.cfg.defense.pgd_steps,
                steps=self.cfg.defense.pgd_steps,
            )
            latent_loss = regularization_loss(self.model.encoder, x_adv_src, x_adv)

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

            x_adv = pgd_attack(
                self,
                x[:n_adv].detach().clone(),
                epsilon=epsilon_test,
                alpha=epsilon_test / self.cfg.defense.pgd_steps,
                steps=self.cfg.defense.pgd_steps,
            )
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
        self.log(
            f"{self.test_mode}_rec_error",
            rec_err.mean(),
            prog_bar=True,
            on_epoch=True,
        )

        metrics = {}
        if self.cfg.metrics.compute_test:
            metrics = compute_metrics(x, x_rec, self.cfg.metrics.types)
            for k, v in metrics.items():
                self.log(f"test_{k}", v, prog_bar=True)

        metrics["anomaly_score"] = rec_err.mean().item()
        metrics["batch_idx"] = batch_idx

        self._test_metrics_epoch.append(metrics)

        return metrics

    def on_test_start(self):
        assert self.trainer.test_dataloaders is not None
        self._test_dataloader = self.trainer.test_dataloaders[0]

    def on_test_epoch_end(self):
        if not self.trainer.is_global_zero:
            return

        out_dir = self._test_out_dir()
        write_test_metrics_csv(self._test_metrics_epoch, out_dir)

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

        out_dir = self._test_out_dir() / "curves"
        out_dir.mkdir(parents=True, exist_ok=True)

        plot_robustness_curves(
            clean_metrics=self._clean_metrics,
            results=self._robustness_results,
            out_dir=str(out_dir),
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

    def _test_out_dir(self) -> Path:
        return Path(self.cfg.trainer.out_dir) / self.test_mode

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
