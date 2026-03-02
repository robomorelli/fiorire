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


def main(config_path: str | Path, mode: str = "train"):
    cfg = OmegaConf.load(config_path)
    cfg = cast(DictConfig, cfg)
    cfg["metrics"]["test_mode"] = "anom" if cfg["metrics"]["perturb_test"] else "clean"

    loggers = [
        TensorBoardLogger(
            save_dir=cfg["trainer"]["out_dir"],
            name=cfg["trainer"]["name_exp"],
            version=cfg["trainer"]["run_name"],
        )
    ]

    datamodule = DataModule(cfg, mode=mode)
    datamodule.setup()
    model = LitAutoEncoder(cfg)

    if mode == "train":
        print("Training from scratch (Lightning checkpoint will be saved)")

        early_stopping = EarlyStopping(
            monitor=cfg["opt"]["monitor_metric"],
            patience=cfg["opt"]["es_patience"],
            mode=cfg["opt"]["monitor_mode"],
            verbose=True,
        )
        monitor_metric = cfg["opt"]["monitor_metric"]
        filename=f"best-{{epoch:03d}}-{{{monitor_metric}:.4f}}"
        checkpoint_cb = ModelCheckpoint(
            dirpath=None,
            filename=filename,
            monitor=monitor_metric,  # metrica da ottimizzare
            mode=cfg["opt"]["monitor_mode"],
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

        trainer = Trainer(
            accelerator=cfg["trainer"]["accelerator"],
            devices=cfg["trainer"]["devices"],
            strategy=cfg["trainer"]["strategy"],
            accumulate_grad_batches=cfg["trainer"]["accumulate_grad_batches"],
            logger=loggers,
            inference_mode=False,
        )

        print(f"Running {cfg["metrics"]["test_mode"]} test")
        datamodule_anom = DataModule(cfg, mode="test")
        model = LitAutoEncoder.load_from_checkpoint(
            cfg["defense"]["checkpoint_path"],
            cfg=cfg,
            strict=True,
            weights_only=False,
        )
        trainer.test(model, datamodule=datamodule_anom)

    else:
        raise ValueError("mode deve essere 'train' o 'test'")


if __name__ == "__main__":
    fire.Fire(main)
