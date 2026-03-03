from pathlib import Path
from typing import cast
import pandas as pd
from omegaconf import OmegaConf, DictConfig
from fire import Fire
import torch
from pytorch_lightning import Trainer

from robustness.lightning_module.lit_module import LitAutoEncoder
from robustness.dataset.data_module import DataModule
from robustness.evaluation.robustness_curves import plot_robustness_curves

torch.set_float32_matmul_precision('medium')


def _load_all_metrics(csv_file: Path) -> dict:
    df = pd.read_csv(csv_file)
    # tieni solo la riga aggregata
    row = df[df["batch_idx"] == "ALL"]
    if row.empty:
        raise ValueError(f"{csv_file} non contiene riga ALL")
    # rimuovi batch_idx prima di plottare
    row = row.drop(columns=["batch_idx"])
    return row.iloc[0].to_dict()


def run_and_plot(config_path: str | Path):
    base_cfg = OmegaConf.load(config_path)
    base_cfg = cast(DictConfig, base_cfg)

    if not base_cfg["curves"]["enabled"]:
        raise ValueError("curves.enabled è False nel config")

    base_run_name = base_cfg["trainer"]["run_name"]
    out_root = Path(base_cfg["trainer"]["out_dir"]) / base_cfg["trainer"]["name_exp"]

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

    # loop su defense off/on
    for defense_flag in [False, True]:
        suffix = "def_on" if defense_flag else "def_off"
        print(f"\nRunning tests with defense = {defense_flag} ({suffix})")

        defense_folder = out_root / base_run_name / suffix
        defense_folder.mkdir(parents=True, exist_ok=True)
        model.cfg["defense"]["apply_defense"] = defense_flag
        model.cfg["metrics"]["defense_suffix"] = suffix

        # clean & perturbed folders
        clean_dir = defense_folder / "clean"
        clean_dir.mkdir(parents=True, exist_ok=True)
        anom_dir = defense_folder / "perturbed"
        anom_dir.mkdir(parents=True, exist_ok=True)

        print("Running CLEAN test")
        model.set_test_configuration(
            test_mode="clean",
            perturb=False, 
            apply_defense=defense_flag
        )
        model.cfg["metrics"]["test_mode"] = f"{suffix}/clean"
        trainer.test(model, datamodule=datamodule)
        clean_metrics = _load_all_metrics(clean_dir / "metrics.csv")

        print("Running PERTURBED test")
        model.set_test_configuration(
            test_mode="perturbed",
            perturb=True,
            apply_defense=defense_flag,
            pgd_epsilon=base_cfg["metrics"]["pgd_epsilon"],
            gaussian_std=base_cfg["metrics"]["real_noise_params"]["gaussian_std"],
            dropout_prob=base_cfg["metrics"]["real_noise_params"]["dropout_prob"],
            impulse_std=base_cfg["metrics"]["real_noise_params"]["impulse_std"]
        )
        model.cfg["metrics"]["test_mode"] = f"{suffix}/perturbed"
        trainer.test(model, datamodule=datamodule)
        anom_metrics = _load_all_metrics(anom_dir / "metrics.csv")

        # perturbations for plotting univariate curves
        perturbation_map = {
            "pgd_epsilon": base_cfg["curves"]["pgd_epsilons"],
            "gaussian_std": base_cfg["curves"]["gaussian_stds"],
            "dropout_prob": base_cfg["curves"]["dropout_probs"],
            "impulse_std": base_cfg["curves"]["impulse_stds"],
        }

        perturb_base = defense_folder / "perturbations"
        perturb_base.mkdir(exist_ok=True)

        for perturb_type, param_list in perturbation_map.items():
            perturb_type_dir = perturb_base / perturb_type
            perturb_type_dir.mkdir(exist_ok=True)

            for p in param_list:
                print(f"Running {perturb_type}={p}")
                test_mode_name = f"{perturb_type}_{p}"

                # setta i parametri corretti
                kwargs = {perturb_type: float(p)}
                model.set_test_configuration(
                    test_mode=test_mode_name,
                    perturb=True,
                    apply_defense=defense_flag,
                    **kwargs
                )

                # test
                model.cfg["metrics"]["test_mode"] = f"{suffix}/perturbations/{perturb_type}/{p}"
                trainer.test(model, datamodule=datamodule)

        # read all the csv of perturbations
        results = {}
        for perturb_type, param_list in perturbation_map.items():
            results[perturb_type] = {}
            perturb_type_dir = perturb_base / perturb_type
            for csv_file in perturb_type_dir.glob("*_metrics.csv"):
                # estrae il parametro dal nome del file
                param = float(csv_file.stem.split("_")[0])
                results[perturb_type][param] = _load_all_metrics(csv_file)

        print("RESULTS STRUCTURE:", results)

        print("\nPlotting robustness curves")
        plot_robustness_curves(
            clean_metrics=clean_metrics,
            anom_metrics=anom_metrics,
            results=results,
            out_dir=str(defense_folder / "robustness_curves"),
        )
        print(f"Robustness curves saved ({suffix})")


if __name__ == "__main__":
    Fire(run_and_plot)