from pytorch_lightning import Trainer
from omegaconf import OmegaConf
from pathlib import Path
import fire

from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from robustness.lightning_module.lit_module import LitAutoEncoder
from robustness.dataset.data_types import Config
from robustness.dataset.data_module import DataModule


def main(config_path: str | Path, mode: str = "train"):
    cfg_default = OmegaConf.structured(Config)  # crea config tipizzata
    yaml_cfg = OmegaConf.load(config_path)
    cfg_merged = OmegaConf.merge(cfg_default, yaml_cfg)

    # qui cfg_merged è ancora DictConfig/structuredConfig ma compatibile
    cfg: Config = OmegaConf.to_object(cfg_merged)  # type: ignore

    datamodule = DataModule(cfg, mode=mode, test_mode = None)
    datamodule.setup()
    model = LitAutoEncoder(cfg)

    if mode == "train":
        print("Training from scratch (Lightning checkpoint will be saved)")

        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=cfg.opt.es_patience,
            mode="min",
            verbose=True,
        )
        checkpoint_cb = ModelCheckpoint(
            dirpath=cfg.trainer.out_dir,
            filename="best-{epoch:03d}-{val_loss:.4f}",
            monitor="val_loss",  # metrica da ottimizzare
            mode="min",
            save_top_k=1,  # salva SOLO il migliore
        )

        trainer = Trainer(
            accelerator=cfg.trainer.accelerator,
            devices=cfg.trainer.devices,
            strategy=cfg.trainer.strategy,
            max_epochs=cfg.trainer.epochs,
            precision=cfg.trainer.precision,
            callbacks=[early_stopping, checkpoint_cb],
        )

        trainer.fit(model, datamodule=datamodule)

    elif mode == "test":
        print("Test: carico modello da checkpoint Lightning")

        model = LitAutoEncoder.load_from_checkpoint(
            cfg.opt.checkpoint_path,
            cfg=cfg,
            strict=True,
        )

        trainer = Trainer(
            accelerator=cfg.trainer.accelerator,
            devices=cfg.trainer.devices,
            strategy=cfg.trainer.strategy,
        )

        print("Running CLEAN test")
        datamodule_clean = DataModule(cfg, mode="test", test_mode="clean")
        trainer.test(model, datamodule=datamodule_clean)

        print("Running ANOMALOUS test")
        datamodule_anom = DataModule(cfg, mode="test", test_mode="anom")
        trainer.test(model, datamodule=datamodule_anom)

    else:
        raise ValueError("mode deve essere 'train' o 'test'")


if __name__ == "__main__":
    fire.Fire(main)
