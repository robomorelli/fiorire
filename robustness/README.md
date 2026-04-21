# Robustness Analysis Framework (End-to-End Documentation)

---

## 1. Project Overview

This framework is designed for **robustness analysis in anomaly detection models**, not only standard accuracy evaluation.

It enables:

* sensitivity analysis under controlled perturbations
* adversarial robustness evaluation
* robustness curve construction across attack strengths
* quantitative robustness comparison between configurations (defense ON/OFF)
* stress-testing models under distribution shifts

The system is built on **PyTorch Lightning**, with modular components for:

* training
* testing
* adversarial attacks
* robustness evaluation
* plotting and reporting

---

## 2. Repository Structure

```text
robustness/
├── config.yaml
├── config_2.yaml
├── config_adv.yaml
├── config_reg_aoc.yaml
├── config_reg_cmg.yaml
├── config_curves.yaml
├── config_curves_aoc.yaml
├── config_curves_cmg.yaml
├── README.md
├── requirements.txt
├── pyproject.toml
│
├── evaluation/
│   ├── write_csv.py
│   ├── robustness_curves.py
│   ├── metrics.py
│
├── input_perturbation/
│   ├── real.py
│   ├── pgd.py
│   ├── defenses.py
│
├── lightning_module/
│   ├── lit_module.py
│   ├── losses.py
│   ├── scheduler.py
│
├── dataset/
│   ├── dataset.py
│   ├── data_module.py
│   ├── wombats.py
│
├── scripts/
│   ├── run.py
│   ├── plot_curves.py
│   ├── auc_curves.py
│   ├── perturb_budget.py
│
└── lightning_logs/
    └── CMG/
        └── test_vanilla_final/
```

---

## 3. Configuration (Example)

```yaml
model:
  kernel_size: 5
  base_filters: 16
  num_layers: 2
  compression_factor: 8
  bottleneck_conv: 0
  decoder_mode: "progressive"
  activation: "Relu"
  bottleneck_activation: null
  aux_channels: 1
  increasing: 1
  flattened: 1
  pool: 1
  halve_both: 1
  halve_time: 0
  halve_features: 0
  double_deconv: 1

dataset:
  csv_path: "./data/fiorire_2/AllScenarios_FSS.csv"
  n_seq_chunk: 200
  test_chunk_ratio: 0.2
  val_ratio: 0.2
  seq_in_length: 16
  seq_stride_train: 4
  seq_stride_val: 16
  seq_stride_test: 16
  num_workers: 16
  label_granularity: "sequence"

opt:
  lr: 1e-3
  batch_size: 500
  es_patience: 30
  monitor_metric: "val_roc_auc"
  monitor_mode: "max"

attack:
  type: "l1"
  num_iter: 25
  k: 5
  budget: 1
  random_noise_std: 0.001

metrics:
  types: ["mse/mae", "pr_auc", "roc_auc"]
  return_curves: true
  perturb_test: true
  p95:
    def_off: 0.004
    def_on: 0.05

curves:
  enabled: true
  attacks:
    l1_budget: [0.01, 0.5, 1.0, 5.0, 25.0, 100.0]
    l0_k: [1, 2, 5, 10, 15]
    random_noise_std: [0.001, 0.01, 0.1, 1.0]
```

---

## 4. Training Pipeline (`main`)

Supports:

* training from scratch
* checkpoint saving
* early stopping
* validation monitoring

```python
def main(config_path, mode="train"):
    cfg = OmegaConf.load(config_path)

    datamodule = DataModule(cfg, mode=mode)
    model = LitAutoEncoder(cfg)

    if mode == "train":

        early_stopping = EarlyStopping(
            monitor=cfg["opt"]["monitor_metric"],
            patience=cfg["opt"]["es_patience"],
            mode=cfg["opt"]["monitor_mode"],
        )

        checkpoint_cb = ModelCheckpoint(
            monitor=cfg["opt"]["monitor_metric"],
            mode=cfg["opt"]["monitor_mode"],
            save_top_k=1,
        )

        trainer = Trainer(
            max_epochs=cfg["trainer"]["epochs"],
            callbacks=[early_stopping, checkpoint_cb],
        )

        trainer.fit(model, datamodule)

    elif mode == "test":
        model = LitAutoEncoder.load_from_checkpoint(
            cfg["defense"]["checkpoint_path"],
            cfg=cfg,
        )

        trainer.test(model, datamodule)
```

---

## 5. Robustness Evaluation Pipeline (`run_and_plot`)

### Execution modes

* `def_off` → no defense
* `def_on` → defense enabled

---

### Pipeline steps

1. **Clean evaluation**
2. **Adversarial evaluation**
3. **Perturbation sweep**
4. **Metric aggregation**
5. **Plot generation**

---

## 6. Metrics Definition

### Core metrics

* ROC AUC
* PR AUC
* Recall@FPR=0.05
* Score separation (Cohen’s d)
* Mean score shift

---

### Robustness delta

\[\Delta = \text{clean} - \text{attacked}\]

---

### ASR (Attack Success Rate)

Fraction of attacked anomalies below threshold.

---

## 7. Model (Lightning Module)

### Loss

\[L = L_{reconstruction} + \lambda L_{jacobian}\]

---

### Key features

* convolutional autoencoder
* Lipschitz regularization
* EMA-based λ controller
* feature weighting defense
* adversarial reconstruction mode

---

## 8. Validation

* ROC AUC / PR AUC computation
* $\tau_{95}$ estimation
* Lipschitz norm tracking
* checkpoint metric logging

---

## 9. Test-Time Behavior

Supports:

* clean evaluation
* adversarial evaluation
* defense-enabled reconstruction
* attacked sample tracking

---

## 10. Robustness Summary Output

Each run generates:

```
robustness_summary.csv
```

Includes:

* clean metrics
* perturbed metrics
* sweep results
* robustness deltas
* ASR

---

## 11. Output Structure

```text
lightning_logs/
└── CMG/
    └── test_vanilla_final/
        ├── pr_compare.png
        ├── roc_compare.png
        │
        ├── def_off/
        │   ├── clean/
        │   ├── perturbed/
        │   ├── robustness_summary.csv
        │   ├── curves.npz
        │   └── robustness_curves/
        │
        └── def_on/
            ├── clean/
            ├── perturbed/
            ├── robustness_summary.csv
            ├── curves.npz
            └── robustness_curves/
```

---

## 12. Curve Plotting (`plot_curves.py`)

Generates:

* ROC AUC vs perturbation
* PR AUC vs perturbation
* Recall@FPR vs perturbation
* Score separation curves
* Mean shift curves

---

## 13. Design Principle

This framework focuses on:

> **robustness under adversarial and distribution shifts, not just accuracy**

It enables systematic evaluation of failure modes under stress.

---

## 14. Notes

* deterministic evaluation per config
* full logging of all perturbations
* τ₉₅ computed from clean validation
* robustness = degradation under attack

---

## End of README
