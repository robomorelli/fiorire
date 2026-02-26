import copy
from pathlib import Path
from typing import cast
import pandas as pd
from omegaconf import OmegaConf, DictConfig
from fire import Fire

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

# importa il tuo main
from robustness.run import main  # <-- cambia nome file se diverso
from robustness.lightning_module.lit_module import LitAutoEncoder
from robustness.dataset.data_module import DataModule
# importa la funzione di plotting che hai già
from robustness.evaluation.robustness_curves import plot_robustness_curves


def _load_all_metrics(csv_file: Path) -> dict:
    df = pd.read_csv(csv_file)
    row = df[df["batch_idx"] == "ALL"]
    if row.empty:
        raise ValueError(f"{csv_file} non contiene riga ALL")
    return row.iloc[0].to_dict()


def run_and_plot(config_path: str | Path):

    base_cfg = OmegaConf.load(config_path)
    base_cfg = cast(DictConfig, base_cfg)

    if not base_cfg["curves"]["enabled"]:
        raise ValueError("curves.enabled è False nel config")

    base_run_name = base_cfg["trainer"]["run_name"]
    out_root = (
        Path(base_cfg["trainer"]["out_dir"])
        / base_cfg["trainer"]["name_exp"]
    )

    datamodule = DataModule(base_cfg, mode="test")
    model = LitAutoEncoder.load_from_checkpoint(
        base_cfg["defense"]["checkpoint_path"],
        strict=True,
        weights_only=False,
    )
    model.cfg = cast(DictConfig, OmegaConf.merge(model.cfg, base_cfg))
    trainer = Trainer(
        accelerator=base_cfg["trainer"]["accelerator"],
        devices=base_cfg["trainer"]["devices"],
        strategy=base_cfg["trainer"]["strategy"],
        logger=False,
        inference_mode=False,
    )
    print("Model loaded.\n")

    print("Running CLEAN test")
    model.set_test_configuration(
        test_mode="clean",
        perturb=False,
    )
    trainer.test(model, datamodule=datamodule)
    clean_csv = (
        out_root
        / base_run_name
        / "clean"
        / "metrics.csv"
    )
    clean_metrics = _load_all_metrics(clean_csv)

    print("Running ANOM test")
    model.set_test_configuration(
        test_mode="anom",
        perturb=True,
    )
    trainer.test(model, datamodule=datamodule)
    anom_csv = (
        out_root
        / base_run_name
        / "anom"
        / "metrics.csv"
    )
    anom_metrics = _load_all_metrics(anom_csv)

    # Perturbations
    perturbation_map = {
        "adversarial": base_cfg["curves"]["adversarial_epsilons"],
        "gaussian": base_cfg["curves"]["gaussian_stds"],
        "dropout": base_cfg["curves"]["dropout_probs"],
        "impulse": base_cfg["curves"]["impulse_stds"],
    }

    results = {}
    for perturb_type, param_list in perturbation_map.items():
        results[perturb_type] = {}
        for p in param_list:
            print(f"Running {perturb_type} | param={p}")

            if perturb_type == "adversarial":
                model.set_test_configuration(
                    test_mode=f"{perturb_type}_{p}",
                    perturb=True,
                    epsilon=float(p),
                )

            elif perturb_type == "gaussian":
                model.set_test_configuration(
                    test_mode=f"{perturb_type}_{p}",
                    perturb=True,
                    gaussian_std=float(p),
                )

            elif perturb_type == "dropout":
                model.set_test_configuration(
                    test_mode=f"{perturb_type}_{p}",
                    perturb=True,
                    dropout_prob=float(p),
                )

            elif perturb_type == "impulse":
                model.set_test_configuration(
                    test_mode=f"{perturb_type}_{p}",
                    perturb=True,
                    impulse_std=float(p),
                )

            trainer.test(model, datamodule=datamodule)

            csv_path = (
                out_root
                / base_run_name
                / f"{perturb_type}_{p}"
                / "metrics.csv"
            )

            metrics_dict = _load_all_metrics(csv_path)
            results[perturb_type][float(p)] = metrics_dict

    # Ploting curves
    print("\nPlotting robustness curves")
    plot_robustness_curves(
        clean_metrics=clean_metrics,
        anom_metrics=anom_metrics,
        results=results,
        out_dir=str(out_root / base_run_name / "robustness_curves"),
    )
    print("Robustness curves saved.")

if __name__ == "__main__":
    Fire(run_and_plot)