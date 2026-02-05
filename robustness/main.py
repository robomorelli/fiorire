from pytorch_lightning import Trainer
from pathlib import Path
import fire
import yaml

from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

from robustness.lightning_module.lit_module import LitAutoEncoder
from robustness.dataset.data_module import DataModule


def main(config_path: str | Path, mode: str = "train"):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg: dict = yaml.safe_load(f)

    loggers = [
        TensorBoardLogger(
            save_dir="lightning_logs",
            name="ae_robust",
            version=cfg["trainer"]["run_name"],
        ),
        CSVLogger(
            save_dir="lightning_logs",
            name="ae_robust",
            version=cfg["trainer"]["run_name"],
        ),
    ]

    datamodule = DataModule(cfg, mode=mode, test_mode=None)
    datamodule.setup()

    cfg["model"]["aux_channels"] = datamodule.n_features
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
            dirpath=cfg["trainer"]["out_dir"],
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
            precision=cfg["trainer"]["precision"], # type: ignore
            callbacks=[early_stopping, checkpoint_cb],
            logger=loggers,
        )

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
