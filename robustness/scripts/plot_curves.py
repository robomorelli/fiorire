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
    model.model = torch.compile(model.model)  # type: ignore
    trainer = Trainer(
        accelerator=base_cfg["trainer"]["accelerator"],
        devices=base_cfg["trainer"]["devices"],
        strategy=base_cfg["trainer"]["strategy"],
        logger=False,
        inference_mode=False,
    )
    print("Model loaded.\n")

    for defense_flag in [False, True]:
        suffix = "def_on" if defense_flag else "def_off"
        print(f"\nRunning tests with defense = {defense_flag} ({suffix})")

        defense_folder = out_root / base_run_name / suffix
        defense_folder.mkdir(parents=True, exist_ok=True)

        # ── Clean test ────────────────────────────────────────────────────────
        print("Running CLEAN test")
        model.set_test_configuration(
            test_mode=f"{suffix}/clean",
            perturb=False,
            apply_defense=defense_flag,
            defense_suffix=suffix,
        )
        trainer.test(model, datamodule=datamodule)
        clean_metrics = _load_all_metrics(defense_folder / "clean" / "metrics.csv")

        # ── Perturbed test ────────────────────────────────────────────────────────
        print("Running PERTURBED test")
        attack_cfg = base_cfg["attack"]
        real_p = base_cfg["attack"]["real_noise"]

        # Ricava il parametro specifico dell'attacco dal config
        l2_budget = float(attack_cfg["budget"]) if attack_cfg["type"] == "l2" else None
        l0_k      = int(attack_cfg["k"])        if attack_cfg["type"] == "l0" else None

        model.set_test_configuration(
            test_mode=f"{suffix}/perturbed",
            perturb=True,
            apply_defense=defense_flag,
            defense_suffix=suffix,
            attack_type=attack_cfg["type"],
            l2_budget=l2_budget,
            l0_k=l0_k,
            gaussian_std=real_p["gaussian_std"],
            dropout_prob=real_p["dropout_prob"],
            impulse_std=real_p["impulse_std"],
        )
        trainer.test(model, datamodule=datamodule)
        anom_metrics = _load_all_metrics(defense_folder / "perturbed" / "metrics.csv")

        # ── Univariate perturbation sweep for robustness curves ───────────────
        curves_cfg = base_cfg["curves"]

        # Map: (perturb_type_key, set_test_configuration kwarg, param_list)
        perturbation_sweep = [
            # adversarial attacks
            ("l2_budget", dict(attack_type="l2"), curves_cfg["attacks"]["l2_budget"]),
            ("l0_k",      dict(attack_type="l0"), curves_cfg["attacks"]["l0_k"]),
            # real noise
            ("gaussian_std", {}, curves_cfg["noise"]["gaussian_std"]),
            ("dropout_prob", {}, curves_cfg["noise"]["dropout_prob"]),
            ("impulse_std",  {}, curves_cfg["noise"]["impulse_std"]),
        ]

        perturb_base = defense_folder / "perturbations"
        perturb_base.mkdir(exist_ok=True)

        for perturb_key, extra_kwargs, param_list in perturbation_sweep:
            perturb_type_dir = perturb_base / perturb_key
            perturb_type_dir.mkdir(exist_ok=True)

            for p in param_list:
                print(f"Running {perturb_key}={p}")
                # Build the per-sweep kwarg (l2_budget, l0_k, or noise param)
                sweep_kwarg = {perturb_key: p}
                model.set_test_configuration(
                    test_mode=f"{suffix}/perturbations/{perturb_key}/{p}",
                    perturb=True,
                    apply_defense=defense_flag,
                    defense_suffix=suffix,
                    **extra_kwargs,
                    **sweep_kwarg,
                )
                trainer.test(model, datamodule=datamodule)

        # ── Collect results from CSVs ─────────────────────────────────────────
        results = {}
        for perturb_key, _, _ in perturbation_sweep:
            results[perturb_key] = {}
            for csv_file in (perturb_base / perturb_key).glob("*_metrics.csv"):
                param = float(csv_file.stem.split("_")[0])
                results[perturb_key][param] = _load_all_metrics(csv_file)

        print("RESULTS STRUCTURE:", results)

        # ── Plot ──────────────────────────────────────────────────────────────
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