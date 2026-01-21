import pytorch_lightning as pl
from omegaconf import OmegaConf
from pathlib import Path
import fire
import torch

from robustness.lightning_module.lit_module import LitAutoEncoder
from robustness.dataset.data_module import DataModule



def main(config_path: str | Path, mode: str = "train"):
    """
    Avvia training o test a seconda del parametro 'mode'.
    mode = "train" -> addestra il modello da zero
    mode = "test"  -> carica checkpoint (.pt o .ckpt) e valuta sul validation set
    """
    cfg = OmegaConf.load(config_path)

    # Inizializza DataModule
    datamodule = DataModule(cfg)
    datamodule.setup()  # prepara i dataset

    if mode == "train":
        print("Training from scratch")
        model = LitAutoEncoder(cfg)

        trainer = pl.Trainer(
            accelerator=cfg.trainer.accelerator,
            devices=cfg.trainer.devices,
            max_epochs=cfg.trainer.epochs,
            precision=cfg.trainer.precision,
            default_root_dir=cfg.trainer.out_dir,
            deterministic=True,
            log_every_n_steps=10
        )

        trainer.fit(model, datamodule=datamodule)


    elif mode == "test":
        print("Modalità test: carico checkpoint e valuto sul validation set")
        model = LitAutoEncoder(cfg)

        # Carico checkpoint: supporta sia .pt (state_dict) sia .ckpt Lightning
        checkpoint_path = cfg.opt.get("checkpoint_path")
        if checkpoint_path is None:
            raise ValueError("Per test serve un checkpoint nel config.yaml: opt.checkpoint_path")

        if checkpoint_path.endswith(".ckpt"):
            # Lightning checkpoint
            model = LitAutoEncoder.load_from_checkpoint(checkpoint_path)
        elif checkpoint_path.endswith(".pt"):
            # PyTorch state_dict
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(state_dict)
        else:
            raise ValueError("Checkpoint deve essere .ckpt o .pt")

        model.eval()  # importantissimo per inference

        trainer = pl.Trainer(
            accelerator=cfg.trainer.accelerator,
            devices=cfg.trainer.devices,
        )

        # usa validation loader per test
        trainer.validate(model, dataloaders=datamodule.val_dataloader())

    else:
        raise ValueError("mode deve essere 'train' oppure 'test'")


if __name__ == "__main__":
    fire.Fire(main)
