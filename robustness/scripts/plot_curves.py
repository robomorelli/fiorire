from pathlib import Path
from typing import cast
from omegaconf import OmegaConf, DictConfig
from fire import Fire
import torch
from pytorch_lightning import Trainer

from robustness.lightning_module.lit_module import LitAutoEncoder
from robustness.dataset.data_module import DataModule
from robustness.evaluation.robustness_curves import plot_robustness_curves

torch.set_float32_matmul_precision('medium')


def run_and_plot(config_path: str | Path):
    base_cfg = OmegaConf.load(config_path)
    base_cfg = cast(DictConfig, base_cfg)

    if not base_cfg["curves"]["enabled"]:
        raise ValueError("curves.enabled è False nel config")

    base_run_name = base_cfg["trainer"]["run_name"]
    out_root = Path(base_cfg["trainer"]["out_dir"]) / base_cfg["trainer"]["name_exp"]

    datamodule = DataModule(base_cfg, mode="test")

    ckpt = torch.load(
        base_cfg["defense"]["checkpoint_path"],
        map_location="cpu",
        weights_only=False
    )
    cfg_ckpt = ckpt["hyper_parameters"]
    merged_cfg = OmegaConf.merge(base_cfg, cfg_ckpt)
    # sovrascrivo paramentri di output
    merged_cfg.trainer.out_dir = base_cfg.trainer.out_dir
    merged_cfg.trainer.run_name = base_cfg.trainer.run_name
    merged_cfg.trainer.name_exp = base_cfg.trainer.name_exp
    
    model = LitAutoEncoder.load_from_checkpoint(
        base_cfg["defense"]["checkpoint_path"],
        cfg=merged_cfg,
        strict=True,
        weights_only=False
    )
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

        # clean test
        print("Running CLEAN test")
        model.set_test_configuration(
            test_mode=f"{suffix}/clean",
            perturb=False,
            apply_defense=defense_flag,
            defense_suffix=suffix,
        )
        results_clean = trainer.test(model, datamodule=datamodule)
        clean_metrics = results_clean[0]

        # perturbed test
        print("Running PERTURBED test")
        attack_cfg = base_cfg["attack"]
        real_p = base_cfg["attack"]["real_noise"]

        # ricava il parametro specifico dell'attacco dal config
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
        results_pert = trainer.test(model, datamodule=datamodule)
        anom_metrics = results_pert[0]

        # univariate perturbation sweep for robustness curves
        curves_cfg = base_cfg["curves"]

        # map: (perturb_type_key, set_test_configuration kwarg, param_list)
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

        results = {}
        for perturb_key, extra_kwargs, param_list in perturbation_sweep:
            results[perturb_key] = {}
            for p in param_list:
                print(f"Running {perturb_key}={p}")
                sweep_kwarg = {perturb_key: p}
                model.set_test_configuration(
                    test_mode=f"{suffix}/perturbations/{perturb_key}/{p}",
                    perturb=True,
                    apply_defense=defense_flag,
                    defense_suffix=suffix,
                    **extra_kwargs,
                    **sweep_kwarg,
                )
                results[perturb_key][p] = trainer.test(model, datamodule=datamodule)[0]

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