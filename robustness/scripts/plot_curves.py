from pathlib import Path
from typing import Any, Mapping, cast
from omegaconf import OmegaConf, DictConfig
from fire import Fire
import torch
from pytorch_lightning import Trainer

from robustness.scripts.perturb_budget import compute_perturbation_budget
from robustness.lightning_module.lit_module import LitAutoEncoder
from robustness.evaluation.metrics import robustness_delta, save_robustness_summary
from robustness.dataset.data_module import DataModule
from robustness.evaluation.robustness_curves import plot_robustness_curves

torch.set_float32_matmul_precision('medium')



def run_and_plot(config_path: str | Path) -> None:
    base_cfg = OmegaConf.load(config_path)
    base_cfg = cast(DictConfig, base_cfg)
 
    if not base_cfg["curves"]["enabled"]:
        raise ValueError("curves.enabled è False nel config")
 
    base_run_name = base_cfg["trainer"]["run_name"]
    out_root = Path(base_cfg["trainer"]["out_dir"]) / base_cfg["trainer"]["name_exp"]
 
    ckpt = torch.load(
        base_cfg["defense"]["checkpoint_path"],
        map_location="cpu",
        weights_only=False,
    )
    cfg_ckpt = ckpt["hyper_parameters"]
    merged_cfg = OmegaConf.merge(base_cfg, cfg_ckpt)
    merged_cfg = cast(DictConfig, merged_cfg)
    merged_cfg.trainer = base_cfg.trainer
 
    datamodule = DataModule(merged_cfg, mode="test")
 
    model = LitAutoEncoder.load_from_checkpoint(
        base_cfg["defense"]["checkpoint_path"],
        cfg=merged_cfg,
        strict=True,
        weights_only=False,
    )
    raw_model = model.model                   # salva prima
    model.model = torch.compile(model.model)  # type: ignore
 
    trainer = Trainer(
        accelerator=merged_cfg["trainer"]["accelerator"],
        devices=merged_cfg["trainer"]["devices"],
        strategy=merged_cfg["trainer"]["strategy"],
        logger=False,
        inference_mode=False,
    )
    print("Model loaded.\n")
 
    attack_cfg = merged_cfg["attack"]
    curves_cfg = merged_cfg["curves"]
    feat_weight = merged_cfg["defense"]["use_feature_weighting"]
 
    perturbation_sweep = [
        ("l1_budget",        {"attack_type": "l1"},     curves_cfg["attacks"]["l1_budget"]),
        ("l0_k",             {"attack_type": "l0"},     curves_cfg["attacks"]["l0_k"]),
        ("random_noise_std", {"attack_type": "random"}, curves_cfg["attacks"]["random_noise_std"]),
    ]
 
    for defense_flag in [False, True]:
        suffix = "def_on" if defense_flag else "def_off"
        defense_folder = out_root / base_run_name / suffix
        defense_folder.mkdir(parents=True, exist_ok=True)
 
        print(f"\nRunning tests with defense={defense_flag} ({suffix}) | feature_weighting={feat_weight}")
 
        # --- clean test ---
        print("Running CLEAN test")
        model.set_test_configuration(
            test_mode=f"{suffix}/clean",
            perturb=False,
            apply_defense=defense_flag,
            use_feature_weighting=feat_weight,
            defense_suffix=suffix,
        )
        clean_metrics: Mapping[str, Any] = trainer.test(model, datamodule=datamodule)[0]
        # use tau95 from checkpoint (computed on clean validation scores at training time)
        # fall back to 0.004 if not found
        p95 = getattr(model, "val_tau95", merged_cfg["metrics"]["p95"])
        print(f"[{suffix}] Using tau95={p95:.6f} from checkpoint (val clean p95)")
        if p95 is not None:
            model.clean_threshold = p95
            device = torch.device(f"cuda:{trainer.device_ids[0]}" if trainer.device_ids else "cpu")
            raw_model = raw_model.to(device)
            compute_perturbation_budget(
                model=raw_model,        # raw nn.Module, not LitAutoEncoder
                datamodule=datamodule,
                threshold=p95,
                defense_folder=defense_folder,
                **merged_cfg["metrics"]["perturbation_budget"]
            )
 
        # --- perturbed baseline test ---
        print("Running PERTURBED test")
        l1_budget = float(attack_cfg["budget"]) if attack_cfg["type"] == "l1" else None
        l0_k = int(attack_cfg["k"]) if attack_cfg["type"] == "l0" else None
 
        model.set_test_configuration(
            test_mode=f"{suffix}/perturbed",
            perturb=True,
            apply_defense=defense_flag,
            use_feature_weighting=feat_weight,
            defense_suffix=suffix,
            attack_type=attack_cfg["type"],
            l1_budget=l1_budget,
            l0_k=l0_k,
        )
        anom_metrics: Mapping[str, Any] = trainer.test(model, datamodule=datamodule)[0]
 
        delta_baseline = robustness_delta(clean_metrics, anom_metrics)
        print(f"[{suffix}] Robustness deltas (clean vs perturbed): {delta_baseline}")
 
        # --- sweep univariato ---
        results: dict[str, dict[Any, Any]] = {}
        sweep_deltas: dict[str, dict[Any, Any]] = {}
 
        for perturb_key, extra_kwargs, param_list in perturbation_sweep:
            results[perturb_key] = {}
            sweep_deltas[perturb_key] = {}
 
            for p in param_list:
                print(f"Running {perturb_key}={p}")
                model.set_test_configuration(
                    test_mode=f"{suffix}/perturbations/{perturb_key}/{p}",
                    perturb=True,
                    apply_defense=defense_flag,
                    use_feature_weighting=feat_weight,
                    defense_suffix=suffix,
                    attack_data_ratio=1.0,
                    **extra_kwargs,
                    **{perturb_key: p},
                )
                r: Mapping[str, Any] = trainer.test(model, datamodule=datamodule)[0]
                results[perturb_key][p] = r
                sweep_deltas[perturb_key][p] = robustness_delta(clean_metrics, r)
 
        # --- salva summary e plot ---
        save_robustness_summary(
            defense_folder=defense_folder,
            suffix=suffix,
            clean_metrics=clean_metrics,
            anom_metrics=anom_metrics,
            delta_baseline=delta_baseline,
            results=results,
            sweep_deltas=sweep_deltas,
        )
 
        print("Plotting robustness curves")
        plot_robustness_curves(
            clean_metrics=clean_metrics,
            anom_metrics=anom_metrics,
            results=results,
            out_dir=str(defense_folder / "robustness_curves"),
        )
        print(f"Robustness curves saved ({suffix})")
 
 
if __name__ == "__main__":
    Fire(run_and_plot)