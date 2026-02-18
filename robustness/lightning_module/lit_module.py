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
    LipschitzEMAController,
    compute_jacobian_norm
)
from robustness.lightning_module.scheduler import build_scheduler
from robustness.evaluation.metrics import compute_metrics
from robustness.evaluation.write_csv import write_metrics_csv
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
        # self.epsilon_train = cfg["defense"]["epsilon"]
        self.lambda_latent = cfg["defense"]["lambda_latent"]

        # training buffers
        self.train_feat_errors = []
        self.train_feat_median = None
        self._epoch_train_loss = []
        self._epoch_recon_loss = []
        self._epoch_latent_loss = []
        self.train_loss = 0
        # self.train_latent_loss = 0
        self.train_recon_loss = 0
        self.lipschitz_ctrl = LipschitzEMAController(
            init_lambda=cfg["defense"]["lambda_latent"],
            target_norm=cfg["defense"]["lipschitz_target"],
            ema_decay=cfg["defense"]["lipschitz_ema"],
            lr=cfg["defense"]["lipschitz_lr"],
            min_lambda=cfg["defense"]["lambda_latent_min"],
            max_lambda=cfg["defense"]["lambda_latent_max"]
        )
        self.current_lambda = cfg["defense"]["lambda_latent"]
        self.current_lipschitz = 0

        # validation epoch buffers
        self._val_scores = []
        self._val_labels = []

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
        x = x.requires_grad_(True)

        # clean forward
        x_hat = self(x)
        recon_loss = reconstruction_loss(x, x_hat)
        # latent_loss = torch.tensor(0.0, device=self.device)

        # feature meadian errors
        feat_err = feature_errors(x, x_hat)
        self.train_feat_errors.append(feat_err.detach())

        lipschitz_norm = torch.tensor(0.0, device=self.device)
        jac_loss = torch.tensor(0.0, device=self.device)

        warmup_epochs = self.cfg["defense"]["warmup_epochs"]

        if (
            self.cfg["defense"]["adv_training"]
            and self.current_epoch >= warmup_epochs
        ):
            # x_adv = pgd_attack(
            #     self,
            #     x,
            #     epsilon=self.epsilon_train,
            #     alpha=self.epsilon_train / self.cfg["defense"]["pgd_steps"],
            #     steps=self.cfg["defense"]["pgd_steps"],
            # )
            # # stop gradient on clean side
            # latent_loss = regularization_loss(self.model.encoder, x, x_adv)

            lipschitz_norm = compute_jacobian_norm(
                self.model.encoder,
                x,
            )

            jac_loss = lipschitz_norm ** 2

            # adaptive lambda update
            self.current_lambda, self.current_lipschitz = self.lipschitz_ctrl.update(lipschitz_norm)

        # loss = recon_loss + self.lambda_latent * latent_loss
        loss = recon_loss + self.current_lambda * jac_loss

        self._epoch_train_loss.append(loss.detach())
        self._epoch_recon_loss.append(recon_loss.detach())
        # self._epoch_latent_loss.append(latent_loss.detach())

        return loss

    def on_train_epoch_end(self):
        if not self.train_feat_errors:
            return
        all_feat_err = torch.cat(self.train_feat_errors, dim=0)  # [N,F]
        self.train_feat_median = all_feat_err.median(dim=0).values
        self.train_feat_errors.clear()

        # epoch logging
        if not self._epoch_train_loss:
            return
        self.train_loss = torch.stack(self._epoch_train_loss).mean()
        self.train_recon_loss = torch.stack(self._epoch_recon_loss).mean()
        # self.train_latent_loss = float(torch.stack(self._epoch_latent_loss).mean())

        self.log("train_loss", self.train_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_recon_loss", self.train_recon_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        # self.log("train_latent_loss", self.train_latent_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_lambda", torch.tensor(self.current_lambda, device=self.device),
                 on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_lipschitz_norm", torch.tensor(self.current_lipschitz, device=self.device),
                 on_epoch=True, prog_bar=True, sync_dist=True)

        self._epoch_train_loss.clear()
        self._epoch_recon_loss.clear()
        # self._epoch_latent_loss.clear()

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx):
        x, y = batch
        x_rec = self(x)

        loss = reconstruction_loss(x, x_rec, reduction = "none")
        rec_err = loss.view(x.size(0), -1).mean(dim=1)  # shape: (B,)

        self._val_scores.append(rec_err.detach())
        self._val_labels.append(y.detach())

        return loss

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking or not self._val_scores:
            return

        all_scores = torch.cat(self._val_scores, dim=0).cpu().numpy()
        all_labels = torch.cat(self._val_labels, dim=0).cpu().numpy()

        epoch_val_loss = torch.tensor(all_scores.mean(), device=self.device)
        # log per epoca
        self.log("val_loss", epoch_val_loss, on_epoch=True, prog_bar=True, sync_dist=True)

        metrics = compute_metrics(
            x=None,
            x_rec=None,
            labels=all_labels,
            scores=all_scores,
            metric_types=self.cfg["metrics"].types,
        )
        # log tutte le altre metriche
        for k, v in metrics.items():
            self.log(f"val_{k}", v, on_epoch=True, sync_dist=True)

        # scriviamo CSV SOLO su rank 0
        if self.trainer.is_global_zero:
            csv_dir = (
                Path(self.cfg["trainer"]["out_dir"])
                / self.cfg["trainer"]["name_exp"]
                / self.cfg["trainer"]["run_name"]
            )
            row = {
                "epoch": self.current_epoch,
                "train_loss": self.train_loss,
                "train_recon_loss": self.train_recon_loss,
                "train_lipschitz_norm": torch.tensor(self.current_lipschitz, device=self.device),
                "train_lambda": torch.tensor(self.current_lambda, device=self.device),
                # "train_latent_loss": self.train_latent_loss,
                "val_loss": float(epoch_val_loss),
                **metrics,
            }
            write_metrics_csv([row], csv_dir, filename="metrics_train.csv")

        # cleanup
        self._val_scores.clear()
        self._val_labels.clear()

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
        write_metrics_csv(csv_data, out_dir)

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
