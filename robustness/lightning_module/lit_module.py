from pathlib import Path
from omegaconf import DictConfig
import pytorch_lightning as pl
import torch
from torch import Tensor
from typing import cast
import numpy as np

from models.conv_ae2D import CONV_AE2D
from robustness.lightning_module.losses import (
    reconstruction_loss,
    feature_errors,
    LipschitzEMAController,
    compute_jacobian_norm,
)
from robustness.lightning_module.scheduler import build_scheduler
from robustness.evaluation.metrics import compute_metrics
from robustness.evaluation.write_csv import write_metrics_csv
from robustness.input_perturbation.defenses import reconstruct_and_weight
from robustness.input_perturbation.pgd import adaptive_pgd_steps, pgd_attack
from robustness.input_perturbation.real import random_real_perturbation


class LitAutoEncoder(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.save_hyperparameters(cfg)
        self.cfg = cfg
        self.model = CONV_AE2D(cfg)
        self.lr = cfg["opt"]["lr"]
        self.lambda_latent = cfg["defense"]["lambda_latent"]

        # training buffers
        self.train_feat_errors = []
        self.train_feat_median = None
        self._epoch_train_loss = []
        self._epoch_recon_loss = []
        self._epoch_jac_loss = []
        self._epoch_ratio = []
        self._epoch_lipschitz_norm = []
        self._epoch_lambda = []
        self.train_loss = 0.0
        self.train_recon_loss = 0.0
        self.train_jac_loss = 0.0
        self.train_ratio = 0.0
        self.train_lipschitz_norm = 0.0
        self.train_lambda = 0.0
        self.lipschitz_ctrl = LipschitzEMAController(
            init_lambda=cfg["defense"]["lambda_latent"],
            target_norm=cfg["defense"]["lipschitz_target"],
            ema_decay=cfg["defense"].get("ema_decay", 0.0),
            lr=cfg["defense"]["lipschitz_lr"],
            min_lambda=cfg["defense"]["lambda_latent_min"],
            max_lambda=cfg["defense"]["lambda_latent_max"],
        )
        self.current_lambda = cfg["defense"]["lambda_latent"]
        self.current_lipschitz = 0.0

        # validation epoch buffers
        self._val_scores = []
        self._val_labels = []
        self._val_lipschitz_norm: list[Tensor] = []
        self.val_lipschitz_norm = None

        # test buffers
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

        # feature meadian errors
        feat_err = feature_errors(x, x_hat)
        self.train_feat_errors.append(feat_err.detach())

        jac_loss = torch.tensor(0.0, device=self.device)
        lipschitz_norm = compute_jacobian_norm(
            self.model.encoder,
            x,
        )
        warmup_epochs = self.cfg["defense"]["warmup_epochs"]

        if (
            self.cfg["defense"]["regularization"]
            and self.current_epoch >= warmup_epochs
        ):
            jac_loss = lipschitz_norm**2

        jac_contrib = self.current_lambda * jac_loss
        ratio = jac_contrib / recon_loss
        loss = recon_loss + self.current_lambda * jac_loss

        self._epoch_train_loss.append(loss.detach())
        self._epoch_recon_loss.append(recon_loss.detach())
        self._epoch_jac_loss.append(jac_contrib.detach())
        self._epoch_ratio.append(ratio.detach())
        self._epoch_lipschitz_norm.append(lipschitz_norm.detach())
        self._epoch_lambda.append(torch.tensor(self.current_lambda, device=self.device))

        return loss

    def on_train_epoch_end(self):
        if not self.train_feat_errors:
            return
        all_feat_err = torch.cat(self.train_feat_errors, dim=0)  # [N,F]
        self.train_feat_median = all_feat_err.median(dim=0).values
        self.train_feat_errors.clear()

        avg_norm = torch.stack(self._epoch_lipschitz_norm).mean()

        if self.cfg["defense"]["regularization"]:
            self.current_lambda, self.current_lipschitz = self.lipschitz_ctrl.update(
                avg_norm
            )
        else:
            self.current_lipschitz = avg_norm

        # epoch logging
        if not self._epoch_train_loss:
            return
        self.train_loss = torch.stack(self._epoch_train_loss).mean()
        self.train_recon_loss = torch.stack(self._epoch_recon_loss).mean()
        self.train_jac_loss = torch.stack(self._epoch_jac_loss).mean()
        self.train_ratio = torch.stack(self._epoch_ratio).mean()
        self.train_lipschitz_norm = avg_norm
        self.train_lambda = torch.stack(self._epoch_lambda).mean()

        self.log(
            "train_loss", self.train_loss, on_epoch=True, prog_bar=True, sync_dist=True
        )
        self.log(
            "train_recon_loss", self.train_recon_loss, on_epoch=True, sync_dist=True
        )
        self.log("train_jac_loss", self.train_jac_loss, on_epoch=True, sync_dist=True)
        self.log(
            "train_ratio",
            self.train_ratio,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "train_lipschitz_norm",
            self.train_lipschitz_norm,
            on_epoch=True,
            sync_dist=True,
        )
        self.log("train_lambda", self.train_lambda, on_epoch=True, sync_dist=True)

        self._epoch_train_loss.clear()
        self._epoch_recon_loss.clear()
        self._epoch_jac_loss.clear()
        self._epoch_ratio.clear()
        self._epoch_lipschitz_norm.clear()
        self._epoch_lambda.clear()

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx):
        x, y = batch
        x = x.requires_grad_(True)

        x_rec = self(x)

        loss = reconstruction_loss(x, x_rec, reduction="none")
        rec_err = loss.view(x.size(0), -1).mean(dim=1)  # shape: (B,)

        self._val_scores.append(rec_err.detach())
        self._val_labels.append(y.detach())

        # compute Lipschitz only at target epoch
        target_epoch = self.cfg["defense"]["save_lipschitz"]

        if (
            not self.cfg["defense"]["regularization"]
            and self.current_epoch == target_epoch
        ):
            with torch.enable_grad():
                x = x.requires_grad_(True)

                lipschitz_norm = compute_jacobian_norm(
                    self.model.encoder,
                    x,
                )

            self._val_lipschitz_norm.append(lipschitz_norm.detach())

        return loss

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking or not self._val_scores:
            return

        all_scores = torch.cat(self._val_scores, dim=0).cpu().numpy()
        all_labels = torch.cat(self._val_labels, dim=0).cpu().numpy()

        # print(
        #     f"VAL Clean scores - mean: {all_scores[all_labels==0].mean():.4f}, std: {all_scores[all_labels==0].std():.4f}"
        # )
        # print(
        #     f"VAL Anomaly scores - mean: {all_scores[all_labels==1].mean():.4f}, std: {all_scores[all_labels==1].std():.4f}"
        # )
        # print(
        #     f"VAL Clean percentiles: 25%={np.percentile(all_scores[all_labels==0],25):.4f}, 50%={np.percentile(all_scores[all_labels==0],50):.4f}, 75%={np.percentile(all_scores[all_labels==0],75):.4f}"
        # )
        # print(
        #     f"VAL Anomaly percentiles: 25%={np.percentile(all_scores[all_labels==1],25):.4f}, 50%={np.percentile(all_scores[all_labels==1],50):.4f}, 75%={np.percentile(all_scores[all_labels==1],75):.4f}"
        # )

        epoch_val_loss = torch.tensor(all_scores.mean(), device=self.device)
        # log per epoca
        self.log(
            "val_loss", epoch_val_loss, on_epoch=True, prog_bar=True, sync_dist=True
        )

        target_epoch = self.cfg["defense"]["save_lipschitz"]
        if (
            not self.cfg["defense"]["regularization"]
            and self.current_epoch == target_epoch
            and self._val_lipschitz_norm
        ):
            local_mean = torch.stack(self._val_lipschitz_norm).mean()

            # media tra GPU
            gathered = cast(Tensor, self.all_gather(local_mean))
            global_mean = gathered.mean()
            self.val_lipschitz_norm = global_mean

            self._val_lipschitz_norm.clear()

        metrics = compute_metrics(
            x=None,
            x_rec=None,
            labels=all_labels,
            scores=all_scores,
            metric_types=self.cfg["metrics"].types,
            return_curves=self.cfg["metrics"]["return_curves"],
        )
        metrics.pop("_roc_curve", None)
        metrics.pop("_pr_curve", None)
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
                "train_jac_loss": self.train_jac_loss,
                "train_ratio": self.train_ratio,
                "train_lambda": float(self.current_lambda),
                "val_loss": float(epoch_val_loss),
                **metrics,
            }
            write_metrics_csv([row], csv_dir, filename="metrics.csv")

        # cleanup
        self._val_scores.clear()
        self._val_labels.clear()

    def test_step(self, batch: tuple[Tensor, Tensor], batch_idx):
        x, y = batch
        apply_defense = self.cfg["defense"]["apply_defense"]
        perturb = self.cfg["metrics"]["perturb_test"]
        epsilon = self.cfg["metrics"]["pgd_epsilon"]
        need_grad = perturb or apply_defense

        with torch.enable_grad() if need_grad else torch.no_grad():
            if perturb:
                anomaly_mask = y == 1
                anomaly_idx = anomaly_mask.nonzero(as_tuple=True)[0]

                if len(anomaly_idx) > 0:

                    # shuffle indici anomali
                    perm = torch.randperm(len(anomaly_idx), device=x.device)
                    anomaly_idx = anomaly_idx[perm]

                    n_adv = int(
                        self.cfg["metrics"]["perturb_fraction"] * len(anomaly_idx)
                    )

                    adv_idx = anomaly_idx[:n_adv]
                    real_idx = anomaly_idx[n_adv:]

                    x = x.clone()

                    # PGD adversarial (minimizza rec loss)
                    if len(adv_idx) > 0:
                        alpha = self.cfg["defense"]["alpha"]  # fisso
                        steps = adaptive_pgd_steps(epsilon, alpha)

                        x_adv = pgd_attack(
                            self,
                            x[adv_idx].detach().clone(),
                            epsilon=epsilon,
                            alpha=alpha,
                            steps=steps,
                        )
                        x[adv_idx] = x_adv

                    # Perturbazioni reali
                    if len(real_idx) > 0:
                        x_real = random_real_perturbation(
                            x[real_idx],
                            self.cfg["metrics"]["real_noise_params"],
                        )
                        x[real_idx] = x_real

            if apply_defense:
                x_rec, rec_err = reconstruct_and_weight(
                    self.model.encoder,
                    self.model.decoder,
                    x,
                    self.cfg["defense"]["alpha"],
                    self.cfg["defense"]["num_iter"],
                    self.train_feat_median,
                )
            else:
                # normal test
                x_rec = self.model.decoder(self.model.encoder(x))
                rec_err = ((x_rec - x) ** 2).mean(dim=(1, 2, 3))  # MSE puro per sample

        # log reconstruction error per batch
        self.log(
            f"{self.cfg['metrics']['test_mode']}_rec_error",
            rec_err.mean(),
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
        self._test_dataloader = self.trainer.test_dataloaders

    def on_test_epoch_end(self):
        if not self.trainer.is_global_zero:
            return

        out_dir = self._test_out_dir()

        # compute global detection metrics
        scores = np.concatenate(self._test_scores, axis=0)
        labels = np.concatenate(self._test_labels, axis=0)
        # in on_test_epoch_end, dopo aver concatenato scores e labels
        anom_scores = scores[labels == 1]
        print(
            f"Anomaly scores percentiles: 25%={np.percentile(anom_scores,25):.4f}, 50%={np.percentile(anom_scores,50):.4f}, 75%={np.percentile(anom_scores,75):.4f}"
        )
        clean_scores = scores[labels == 0]
        print(
            f"Clean scores percentiles: 25%={np.percentile(clean_scores,25):.4f}, 50%={np.percentile(clean_scores,50):.4f}, 75%={np.percentile(clean_scores,75):.4f}"
        )
        # print("Clean scores - mean:", scores[labels==0].mean(), "std:", scores[labels==0].std())
        # print("Anomaly scores - mean:", scores[labels==1].mean(), "std:", scores[labels==1].std())
        all_metrics = compute_metrics(
            x=None,
            x_rec=None,
            labels=labels,
            scores=scores,
            metric_types=self.cfg["metrics"].types,
            return_curves=self.cfg["metrics"]["return_curves"],
        )
        all_metrics["anomaly_score"] = float(scores.mean())

        # rimuovi curve prima del log
        roc_data = all_metrics.pop("_roc_curve", None)
        pr_data = all_metrics.pop("_pr_curve", None)
        # log globale
        for k, v in all_metrics.items():
            self.log(f"{self.cfg['metrics']['test_mode']}_{k}", v, sync_dist=True)

        # scrive CSV batch-wise + aggregate epoca
        csv_data = self._test_metrics_epoch.copy()
        csv_data.append({"batch_idx": "ALL", **all_metrics})

        # se è una perturbation, il nome del CSV include il parametro
        test_mode_parts = self.cfg["metrics"]["test_mode"].split("/")
        if "perturbations" in test_mode_parts:
            param = test_mode_parts[-1]  # prendi sempre l'ultimo pezzo
            out_dir.mkdir(parents=True, exist_ok=True)
            write_metrics_csv(csv_data, out_dir, filename=f"{param}_metrics.csv")
        else:
            out_dir.mkdir(parents=True, exist_ok=True)
            write_metrics_csv(csv_data, out_dir, filename="metrics.csv")

        # salva curve separatamente
        if roc_data is not None and pr_data is not None:
            np.savez(
                out_dir / "curves.npz",
                fpr=roc_data["fpr"],
                tpr=roc_data["tpr"],
                roc_thresholds=roc_data["thresholds"],
                precision=pr_data["precision"],
                recall=pr_data["recall"],
                pr_thresholds=pr_data["thresholds"],
            )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler_dict = build_scheduler(optimizer, self.cfg["opt"])
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}

    def on_save_checkpoint(self, checkpoint):
        if hasattr(self, "train_feat_median"):
            checkpoint["train_feat_median"] = self.train_feat_median
        if self.val_lipschitz_norm is not None:
            checkpoint["val_lipschitz_norm"] = float(self.val_lipschitz_norm)

    def on_load_checkpoint(self, checkpoint):
        self.train_feat_median = checkpoint.get("train_feat_median", None)

    def _test_out_dir(self) -> Path:
        base = (
            Path(self.cfg["trainer"]["out_dir"])
            / self.cfg["trainer"]["name_exp"]
            / self.cfg["trainer"]["run_name"]
        )

        suffix = self.cfg["metrics"].get("defense_suffix", "")
        if suffix:
            base = base / suffix

        test_mode = self.cfg["metrics"]["test_mode"]

        if "perturbations/" in test_mode:
            parts = test_mode.split("/")
            perturb_type = parts[2]  # suffix/perturbations/<tipo>/<param>
            return base / "perturbations" / perturb_type
        else:
            return base / test_mode.split("/")[-1]

    def set_test_configuration(
        self,
        test_mode: str,
        perturb: bool,
        apply_defense: bool = False,
        pgd_epsilon: float = 0.0,
        gaussian_std: float = 0.0,
        dropout_prob: float = 0.0,
        impulse_std: float = 0.0,
    ):
        """
        Aggiorna dinamicamente la configurazione di test
        senza ricreare il modello.
        """

        self.cfg["metrics"]["test_mode"] = test_mode
        self.cfg["metrics"]["perturb_test"] = perturb
        self.cfg["defense"]["apply_defense"] = apply_defense

        # reset
        self.cfg["metrics"]["pgd_epsilon"] = 0.0
        self.cfg["metrics"]["real_noise_params"]["gaussian_std"] = 0.0
        self.cfg["metrics"]["real_noise_params"]["dropout_prob"] = 0.0
        self.cfg["metrics"]["real_noise_params"]["impulse_std"] = 0.0

        # set specifico
        self.cfg["metrics"]["pgd_epsilon"] = float(pgd_epsilon)
        self.cfg["metrics"]["real_noise_params"]["gaussian_std"] = float(gaussian_std)
        self.cfg["metrics"]["real_noise_params"]["dropout_prob"] = float(dropout_prob)
        self.cfg["metrics"]["real_noise_params"]["impulse_std"] = float(impulse_std)

        print(f"\n Test mode set to: {test_mode}")
