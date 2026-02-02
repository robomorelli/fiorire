import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from robustness.dataset.data_types import Config
from models.conv_ae2D import CONV_AE2D
from scheduler import build_scheduler
from defenses import approximate_projection, apply_feature_weighting

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
        feat_err = (x_hat - x).pow(2).mean(dim=(0, 1, 3))  # → [F]
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

        x_rec, _ = approximate_projection(
            encoder=self.model.encoder,
            decoder=self.model.decoder,
            x=x,
            alpha=self.cfg.defense.alpha,
            num_iter=self.cfg.defense.num_iter,
        )

        rec_err_feat = (x_rec - x).pow(2).mean(dim=(1, 3))

        rec_err = apply_feature_weighting(
            rec_err_feat,
            self.train_feat_median,
            epsilon=1e-4,
            batch_idx=batch_idx,
        )

        self.log(
            "test_rec_error",
            rec_err.mean(),
            prog_bar=True,
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
    
    # called when a lightning checkpoint is saved or loaded
    def on_save_checkpoint(self, checkpoint):
        if hasattr(self, "train_feat_median"):
            checkpoint["train_feat_median"] = self.train_feat_median

    def on_load_checkpoint(self, checkpoint):
        self.train_feat_median = checkpoint.get("train_feat_median", None)
