import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from robustness.dataset.data_types import Config
from models.conv_ae2D import CONV_AE2D

class LitAutoEncoder(pl.LightningModule):
    def __init__(self, cfg: Config):
        super().__init__()

        # salva config nel checkpoint
        self.save_hyperparameters(cfg)

        self.cfg = cfg
        self.model = CONV_AE2D(cfg)
        self.lr = cfg.opt.lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch
        x_hat = self(x)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        x_hat = self(x)
        loss = F.mse_loss(x_hat, x)
        self.log("val_loss", loss, prog_bar=True)
    
    # def test_step()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
