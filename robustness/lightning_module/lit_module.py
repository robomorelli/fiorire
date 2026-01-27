import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import warnings

from robustness.dataset.data_types import Config
from models.conv_ae2D import CONV_AE2D
from scheduler import build_scheduler

class LitAutoEncoder(pl.LightningModule):
    def __init__(self, cfg: Config):
        super().__init__()

        # salva config nel checkpoint
        self.save_hyperparameters(cfg)

        self.cfg = cfg
        self.model = CONV_AE2D(cfg)
        self.lr = cfg.opt.lr

        # buffer per errori di training
        self.train_feat_errors = []
        # inizializzati via checkpoint
        self.train_feat_median = None

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch
        x_hat = self(x)

        # errore per feature: mean su batch e tempo
        # x: [B, W, F]
        feat_err = (x_hat - x).pow(2).mean(dim=(0, 1))  # [F]
        self.train_feat_errors.append(feat_err.detach().cpu())

        loss = feat_err.mean()
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        return loss
    
    def on_train_end(self):
        if len(self.train_feat_errors) == 0:
            return

        # [N_steps, F] → [F]
        self.train_feat_median = torch.stack(
            self.train_feat_errors
        ).median(dim=0).values

    def validation_step(self, batch, batch_idx):
        x = batch
        x_hat = self(x)
        loss = F.mse_loss(x_hat, x)
        self.log("val_loss", loss, prog_bar=True)
    
    def test_step(self, batch, batch_idx):
        x = batch

        # ---------- encode ----------
        with torch.no_grad():
            enc = self.model.encoder(x)

        enc = enc.detach().clone().requires_grad_(True)

        loss_fn = torch.nn.SmoothL1Loss(reduction="sum")
        alpha = self.cfg.defense.alpha
        num_iter = self.cfg.defense.num_iter

        # ---------- approximate projection ----------
        for _ in range(num_iter):
            x_rec = self.model.decoder(enc)
            loss = loss_fn(x_rec, x)

            loss.backward()
            enc.data -= alpha * enc.grad
            enc.grad.zero_()

        # ---------- reconstruction error ----------
        # per-feature error: [B, F]
        rec_err_feat = (x_rec - x).pow(2).mean(dim=1)

        # ---------- feature weighting ----------
        if self.train_feat_median is not None:
            weights = 1.0 / (
                1e-4 + self.train_feat_median.to(rec_err_feat.device)
            )
            rec_err = (rec_err_feat * weights).sum(dim=1)
        else:
            if batch_idx == 0:
                warnings.warn(
                    "Train feature errors not found: "
                    "feature weighting DISABLED during test."
                )
            rec_err = rec_err_feat.sum(dim=1)

        self.log(
            "test_rec_error",
            rec_err.mean(),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

        return rec_err

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        # restituisce direttamente il dict pronto per Lightning
        scheduler_dict = build_scheduler(optimizer, self.cfg.opt)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_dict,
        }
    
    def on_save_checkpoint(self, checkpoint):
        if hasattr(self, "train_feat_median"):
            checkpoint["train_feat_median"] = self.train_feat_median

    def on_load_checkpoint(self, checkpoint):
        self.train_feat_median = checkpoint.get("train_feat_median", None)
