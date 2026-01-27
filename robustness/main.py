from pytorch_lightning import Trainer
from omegaconf import OmegaConf
from pathlib import Path
import fire
import torch

from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from robustness.lightning_module.lit_module import LitAutoEncoder
from robustness.utils import inject_training_hparams_from_ckpt
from robustness.dataset.data_types import Config
from robustness.dataset.data_module import DataModule


def main(config_path: str | Path, mode: str = "train"):
    cfg_default = OmegaConf.structured(Config)   # crea config tipizzata
    yaml_cfg = OmegaConf.load(config_path)
    cfg_merged = OmegaConf.merge(cfg_default, yaml_cfg)

    # qui cfg_merged è ancora DictConfig/structuredConfig ma compatibile
    cfg: Config = OmegaConf.to_object(cfg_merged)  #type: ignore

    ckpt = cfg.opt.checkpoint_path
    ckpt_dict = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = inject_training_hparams_from_ckpt(cfg, ckpt_dict)

    datamodule = DataModule(cfg, mode=mode)
    datamodule.setup()
    model = LitAutoEncoder(cfg)

    if mode == "train":
        print("Training: architettura da checkpoint, pesi random")

        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=cfg.opt.es_patience,
            mode="min",
            verbose=True,
        )
        checkpoint_cb = ModelCheckpoint(
            dirpath=cfg.trainer.out_dir,
            filename="best-{epoch:03d}-{val_loss:.4f}",
            monitor="val_loss",      # metrica da ottimizzare
            mode="min",
            save_top_k=1,            # salva SOLO il migliore
            save_last=False,         # opzionale
        )

        trainer = Trainer(
            accelerator=cfg.trainer.accelerator,
            devices=cfg.trainer.devices,
            max_epochs=cfg.trainer.epochs,
            precision=cfg.trainer.precision,
            default_root_dir=cfg.trainer.out_dir,
            callbacks=[early_stopping, checkpoint_cb],
        )

        trainer.fit(model, datamodule=datamodule)

    elif mode == "test":
        print("Test: carico modello CON pesi")

        # carichiamo a mano il modello perché non è un checkpoint Lightning
        # non possiamo usare: LitAutoEncoder.load_from_checkpoint(...)
        model.load_state_dict(ckpt_dict["model_state_dict"], strict=True)
        model.eval()

        trainer = Trainer(
            accelerator=cfg.trainer.accelerator,
            devices=cfg.trainer.devices,
        )

        trainer.test(model, datamodule=datamodule)

    else:
        raise ValueError("mode deve essere 'train' o 'test'")



if __name__ == "__main__":
    fire.Fire(main)
