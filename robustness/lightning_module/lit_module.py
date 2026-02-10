from pathlib import Path
from omegaconf import DictConfig
import pytorch_lightning as pl
import torch
from torch import Tensor
from typing import Literal
import numpy as np

from models.conv_ae2D import CONV_AE2D
from robustness.evaluation.robustness_curves import (
    plot_robustness_curves,
    perturbation_dict,
)
from robustness.lightning_module.losses import (
    reconstruction_loss,
    feature_errors,
    regularization_loss,
)
from robustness.lightning_module.scheduler import build_scheduler
from robustness.evaluation.metrics import compute_metrics
from robustness.evaluation.write_csv import write_test_metrics_csv
from robustness.input_perturbation.defenses import reconstruct_and_weight
from robustness.input_perturbation.adv_train_utils import pgd_attack
from robustness.input_perturbation.real import random_real_perturbation


class LitAutoEncoder(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.save_hyperparameters(cfg)
        self.cfg = cfg
        self.model = CONV_AE2D(cfg)
        self.lr = cfg["opt"]["lr"]
        self.epsilon_train = cfg["defense"]["epsilon"]
        self.lambda_latent = cfg["defense"]["lambda_latent"]
        self.p_adv = cfg["defense"]["p_adv"]

        # training buffers
        self.train_feat_errors = []
        self.train_feat_median = None

        # test buffers
        self.test_mode: Literal["clean", "anom"]
        self._test_scores = []
        self._test_labels = []
        self._test_metrics_epoch = []

        # robustness
        self._clean_metrics = {}
        self._robustness_results = {
            "adversarial": {},
            "gaussian": {},
            "dropout": {},
            "impulse": {},
        }

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx):
        x, _ = batch
        B = x.size(0)

        if self.cfg["defense"]["adv_training"]:
            n_adv = int(self.p_adv * B)
            x_clean = x[:-n_adv]
            x_adv_src = x[-n_adv:]
        else:
            n_adv = 0
            x_clean = x

        x_hat_clean = self(x_clean)
        recon_loss = reconstruction_loss(x_clean, x_hat_clean)
        feat_err = feature_errors(x_clean, x_hat_clean)
        self.train_feat_errors.append(feat_err.detach().cpu())

        latent_loss = torch.tensor(0.0, device=self.device)
        if n_adv > 0:
            x_adv = pgd_attack(
                self,
                x_adv_src,
                epsilon=self.epsilon_train,
                alpha=self.epsilon_train / self.cfg["defense"]["pgd_steps"],
                steps=self.cfg["defense"]["pgd_steps"],
            )
            latent_loss = regularization_loss(self.model.encoder, x_adv_src, x_adv)

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
            sync_dist=True,
        )
        return loss

    def on_train_epoch_end(self):
        if not self.train_feat_errors:
            return
        self.train_feat_median = (
            torch.stack(self.train_feat_errors).median(dim=0).values
        )
        self.train_feat_errors.clear()

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx):
        x, _ = batch
        x_rec = self(x)

        loss = reconstruction_loss(x, x_rec)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

        if self.cfg["metrics"]["compute_validation"]:
            metrics = compute_metrics(
                x=x,
                x_rec=x_rec,
                labels=None,
                scores=None,
                metric_types=self.cfg["metrics"].types,
            )
            for k, v in metrics.items():
                self.log(f"val_{k}", v, prog_bar=True, sync_dist=True)

        return loss

    def test_step(self, batch: tuple[Tensor, Tensor], batch_idx):
        x, y = batch
        perturb = self.cfg["metrics"]["perturb_test"]
        epsilon_test = self.cfg["metrics"]["epsilon"]

        if perturb:
            B = x.size(0)
            n_adv = int(self.cfg["metrics"]["perturb_fraction"] * B)
            x_adv = pgd_attack(
                self,
                x[:n_adv].detach().clone(),
                epsilon=epsilon_test,
                alpha=epsilon_test / self.cfg["defense"]["pgd_steps"],
                steps=self.cfg["defense"]["pgd_steps"],
            )
            x_real = random_real_perturbation(
                x[n_adv:], self.cfg["metrics"]["real_noise_params"]
            )
            x = torch.cat([x_adv, x_real], dim=0)

        x_rec, rec_err = reconstruct_and_weight(
            self.model.encoder,
            self.model.decoder,
            x,
            self.cfg["defense"]["alpha"],
            self.cfg["defense"]["num_iter"],
            self.train_feat_median,
        )

        # log reconstruction error per batch
        self.log(
            f"{self.test_mode}_rec_error",
            rec_err.mean(),
            prog_bar=True,
            on_epoch=True,
            sync_dist=True,
        )

        # accumulo buffer per detection metrics
        self._test_scores.append(rec_err.detach().cpu().numpy())
        self._test_labels.append(y.detach().cpu().numpy())

        # batch-wise (solo anomaly score)
        self._test_metrics_epoch.append(
            {
                "batch_idx": batch_idx,
                "anomaly_score": rec_err.mean().item(),
            }
        )

    def on_test_start(self):
        self._test_scores.clear()
        self._test_labels.clear()
        self._test_metrics_epoch.clear()
        assert self.trainer.test_dataloaders is not None
        self._test_dataloader = self.trainer.test_dataloaders[0]

    def on_test_epoch_end(self):
        if not self.trainer.is_global_zero:
            return

        out_dir = self._test_out_dir()

        # compute global detection metrics
        scores = np.concatenate(self._test_scores, axis=0)
        labels = np.concatenate(self._test_labels, axis=0)

        # ricostruzione media batch-wise
        anomaly_score_mean = float(
            np.mean([m["anomaly_score"] for m in self._test_metrics_epoch])
        )

        # calcola tutte le metriche (detection + reconstruction)
        all_metrics = compute_metrics(
            x=None,  # non serve ricostruzione globale, già nel rec_error
            x_rec=None,
            labels=labels,
            scores=scores,
            metric_types=self.cfg["metrics"].types,
        )
        # aggiungi il rec_error medio
        all_metrics["anomaly_score"] = anomaly_score_mean

        # log globale
        for k, v in all_metrics.items():
            self.log(f"{self.test_mode}_{k}", v)

        # scrive CSV batch-wise + aggregate epoca
        csv_data = self._test_metrics_epoch.copy()
        csv_data.append({"batch_idx": "ALL", **all_metrics})
        write_test_metrics_csv(csv_data, out_dir)

        # robustness curves (solo se non clean)
        if self.test_mode != "clean" and self.cfg["curves"]["enabled"]:
            curves = perturbation_dict(self, self.cfg)
            self._robustness_results = {}
            for name, (params, perturb_builder) in curves.items():
                self._robustness_results[name] = {}
                for p in params:
                    self._robustness_results[name][p] = self._evaluate_on_loader(
                        self._test_dataloader,
                        perturb_fn=perturb_builder(p),
                    )

            out_dir_curves = out_dir / "curves"
            out_dir_curves.mkdir(parents=True, exist_ok=True)
            plot_robustness_curves(
                clean_metrics=self._clean_metrics,
                results=self._robustness_results,
                out_dir=str(out_dir_curves),
            )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler_dict = build_scheduler(optimizer, self.cfg["opt"])
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}

    def on_save_checkpoint(self, checkpoint):
        if hasattr(self, "train_feat_median"):
            checkpoint["train_feat_median"] = self.train_feat_median

    def on_load_checkpoint(self, checkpoint):
        self.train_feat_median = checkpoint.get("train_feat_median", None)

    def _test_out_dir(self) -> Path:
        return Path(self.cfg["trainer"]["out_dir"]) / self.test_mode

    def _evaluate_on_loader(self, dataloader, perturb_fn=None, requires_grad=False):
        """
        Calcola tutte le metriche (reconstruction + detection) batch-wise.
        Restituisce la media su tutti i batch.
        """
        self.eval()
        metrics_acc = []
        all_scores = []
        all_labels = []

        for batch in dataloader:
            batch: tuple[Tensor, Tensor]
            x, y = batch
            x = x.to(self.device)

            if perturb_fn is not None:
                if requires_grad:
                    x = perturb_fn(x)
                else:
                    with torch.no_grad():
                        x = perturb_fn(x)

            with torch.no_grad():
                x_rec, rec_err = reconstruct_and_weight(
                    self.model.encoder,
                    self.model.decoder,
                    x,
                    self.cfg["defense"]["alpha"],
                    self.cfg["defense"]["num_iter"],
                    self.train_feat_median,
                )

            # accumula scores e labels
            all_scores.append(rec_err.detach().cpu().numpy())
            all_labels.append(y.detach().cpu().numpy())

            # calcola metriche di ricostruzione batch-wise
            metrics = compute_metrics(
                x=x,
                x_rec=x_rec,
                labels=None,
                scores=None,
                metric_types=self.cfg["metrics"].types,
            )
            metrics["rec_error"] = rec_err.mean().item()
            metrics_acc.append(metrics)

        # concatena per detection globale
        all_scores = np.concatenate(all_scores, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        # calcola detection metrics globali
        detection_metrics = compute_metrics(
            x=None,
            x_rec=None,
            labels=all_labels,
            scores=all_scores,
            metric_types=self.cfg["metrics"].types,
        )

        # media batch-wise delle ricostruzioni
        recon_metrics = {
            k: sum(m[k] for m in metrics_acc) / len(metrics_acc) for k in metrics_acc[0]
        }

        # unisci tutto
        recon_metrics.update(detection_metrics)

        return recon_metrics
