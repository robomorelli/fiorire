from typing import cast
from pytorch_lightning import Trainer
from pathlib import Path
import fire
import torch
from omegaconf import DictConfig, OmegaConf

from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from robustness.lightning_module.lit_module import LitAutoEncoder
from robustness.dataset.data_module import DataModule

torch.set_float32_matmul_precision("medium")


def main(config_path: str | Path, name: str, mode: str = "train"):
    cfg = OmegaConf.load(config_path)
    cfg = cast(DictConfig, cfg)

    loggers = [
        TensorBoardLogger(
            save_dir=cfg["trainer"]["out_dir"],
            name=name,
            version=cfg["trainer"]["run_name"],
        )
    ]

    datamodule = DataModule(cfg, mode=mode, test_mode=None)
    datamodule.setup()
    model = LitAutoEncoder(cfg)

    if mode == "train":
        print("Training from scratch (Lightning checkpoint will be saved)")

        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=cfg["opt"]["es_patience"],
            mode="min",
            verbose=True,
        )
        checkpoint_cb = ModelCheckpoint(
            dirpath=None,
            filename="best-{epoch:03d}-{val_loss:.4f}",
            monitor="val_loss",  # metrica da ottimizzare
            mode="min",
            save_top_k=1,  # salva SOLO il migliore
        )

        trainer = Trainer(
            accelerator=cfg["trainer"]["accelerator"],
            devices=cfg["trainer"]["devices"],
            strategy=cfg["trainer"]["strategy"],
            max_epochs=cfg["trainer"]["epochs"],
            precision=cfg["trainer"]["precision"],  # type: ignore
            accumulate_grad_batches=cfg["trainer"]["accumulate_grad_batches"],
            callbacks=[early_stopping, checkpoint_cb],
            logger=loggers,
        )

        print(torch.cuda.mem_get_info())
        trainer.fit(model, datamodule=datamodule)

    elif mode == "test":
        print("Test: carico modello da checkpoint Lightning")

        model = LitAutoEncoder.load_from_checkpoint(
            cfg["opt"]["checkpoint_path"],
            cfg=cfg,
            strict=True,
        )

        trainer = Trainer(
            accelerator=cfg["trainer"]["accelerator"],
            devices=cfg["trainer"]["devices"],
            strategy=cfg["trainer"]["strategy"],
            accumulate_grad_batches=cfg["trainer"]["accumulate_grad_batches"],
            logger=loggers,
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
