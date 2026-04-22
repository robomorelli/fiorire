# Robustness Analysis Framework (End-to-End Documentation)

---

## 1. Project Overview

This framework provides a **complete pipeline for robustness evaluation in anomaly detection**, going beyond standard accuracy metrics.

It is designed for reconstruction-based models (autoencoders) operating on multivariate time series and focuses on:

* adversarial and random attacks
* robustness curve construction across attack strengths
* quantitative robustness comparison between different configurations.

This framework evaluates and improves robustness via:

* regularization (training-time)
* adversarial attacks and projection defense (test-time)
* robustness metrics and curves.

The system is entirely built using **PyTorch Lightning**.

---

## 2. Repository Structure

```text
robustness/
├── config.yaml
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
  csv_path: "./davinci-1/home/morellir/artificial_intelligence/repos/fiorire/data/fiorire_2/AllScenarios_FSS.csv"
  n_seq_chunk: 200                    # how many sequences for each chunk
  test_chunk_ratio: 0.2               # % test set
  val_ratio: 0.2                      # % validation set
  n_features: null                    # features dimension F - overridden
  seq_in_length: 16                   # temporal dimension T
  seq_stride_train: 4                 # stride for train
  seq_stride_val: 16                  # stride for validation
  seq_stride_test: 16                 # stride for test
  num_workers: 16                     # workers for the dataloader
  label_granularity: "sequence"       # label for each sequence ("sequence") or for timestamp ("timestamp")
  val_anomaly_ratio: 1.0              # % of windwows to corrupt to add to validation set
  test_anomaly_ratio: 1.0             # % of windwows to corrupt to add to test set
  shuffle_train: false                # flag for random sampler in train
  shuffle_val: true                   # flog for random sampler in validation
  shuffle_test: true                  # flag for random sampler in test
  delta_min: 0.2                      # delta range of anomalies - extract from uniform distribution
  delta_max: 0.2

opt:
  lr: 1e-3
  batch_size: 500
  lr_patience: 10                     # patience for the scheduler
  es_patience: 30                     # patience for early stopping
  lr_factor: 0.5                      # factor for the scheduler
  lr_min: 1e-6                        # min learning to stop when reached
  monitor_metric: "val_roc_auc"       # metrica da monitorare
  monitor_mode: "max"                 # "min" o "max"

attack:
  type: "l1"                 # attack type: l1 | l0
  attack_data_ratio: 0.5     # percentage of adversarial (1-ratio randomized)
  # shared params
  num_iter: 25               # number of attack iterations
  # l0-specific
  k: 5                       # number of features to perturb (top-k by gradient magnitude)
  # l1-specific
  budget: 1                  # max l1 perturbation budget (as % of input l1 norm)
  # real noise applied to clean samples
  random_num_vectors: 100
  random_noise_std: 0.001

defense:
  # path to load the checkpoint from
  checkpoint_path: "/davinci-1/home/lorenrossi/log_fiorire/lightning_logs/CMG/vanilla/checkpoints/best-epoch=109-val_roc_auc=0.9436.ckpt"
  apply_defense: false                # if applying defense in test or not (overridden during execution)
  use_feature_weighting: false        # if apply feature weightning in defense
  alpha: 1e-2                         # intensity for each step size of pgd in latent space
  num_iter: 25                        # number of projection steps
  # adversarial training
  regularization: false               # flag to use jacobian regularization
  controller: "ratio"                 # decide which EMA is used: "ratio" | "norm"
  save_lipschitz: 5                   # number of epochs after we save the norm on validation set
  lambda_init: 1e-2                   # max lambda allowed (lambda for latent loss)
  lambda_min: 1e-6                    # lambda minimo consentito
  lambda_max: 1e-1                    # max lambda consentito
  target_norm: 1.0                    # parametro specifico per "norm"
  target_ratio: 0.2                   # parametro specifico per "ratio"
  lr_lambda: 0.01                     # velocità adattamento lambda
  ema_decay: 0.95                     # smoothing EMA norma

trainer:
  accelerator: "gpu"
  devices: 1
  strategy: "auto"
  epochs: 1000
  precision: "bf16-mixed"
  accumulate_grad_batches: 4
  out_dir: "/davinci-1/home/morellir/artificial_intelligence/repos/fiorire/robustness/lightning_module"
  name_exp: FSS
  run_name: test_reg

metrics:
  types: ["mse/mae", "pr_auc", "roc_auc"]  # metrics to be computed in validation and test
  return_curves: true                      # to plot pr/roc curves
  perturb_test: true                       # flag to decide to perturb (bot adv e real) the test set or not
  p95:  # 95-th percentile of clean scores used as threshold
    def_off: 0.004
    def_on: 0.05
  perturbation_budget:
    n_samples: 500
    budget_high: 100.0
    n_iter_search: 20
    n_iter_attack: 25
    tol: 0.05
    boundary_tol: 1.0
    n_plot: 5

curves:
  enabled: true
  attacks:
    l1_budget:    [0.01,0.5,1.0,5.0,25.0,100.0]
    l0_k:         [1,2,5,10,15]
    random_noise_std:   [0.001, 0.01, 0.1, 1.0]
```

---

## 4. Model

The model used is a **Convolutional 2D Autoencoder**. The parameters are specified in the example config.

---

## 5. Training Pipeline (`run.py`)

The command line takes a only argument: the `config_path`.
```
python -m robustness.scripts.run --config_path robustness/config.yaml
```

Training can be done with a regularized loss or vanilla (only reconstruction loss):

$$L = L_{\text{reconstruction}} + \lambda L_{\text{jacobian}}$$

It supports:
* training from scratch of the model
* checkpoint saving the best performance
* early stopping
* validation monitoring

### Validation

* ROC AUC / PR AUC computation
* $\tau_{95}$ estimation
* Lipschitz norm tracking
* checkpoint metric logging

### Single Test

A single test can be made.
```
python -m robustness.scripts.run --config_path robustness/config.yaml --mode test
```

---

## 6. Multiple Testing Pipeline (`plot_curves.py`)

Command line:
```
python -m robustness.scripts.plot_curves --config_path robustness/config_curves_cmg.yaml
```

Various tests are made.
1. **Clean** test: no data perturbed, used as baseline.
2. **Perturbation Budget**: computed using a batched binary search algorithm.
3. **Perturbed** test: fixed perturbation parameters, mix of adversial and random attacks.
4. **Sweeps**: fixed perturbation type, varying the attack intensity.

Finally, perturbation curves are plotted and saved.

### Execution modes

Defense and feature weighting can be applied in inference time.

* `apply_defense`: flag for projection defense
* `use_feature_weighting`: flag for feature weighting.

The `apply_defense` is changed during the execution: the code runs with both value True and False. Two separate folders are saved.

---

## 7. Metrics Definition

Multiple metrics are used for evaluation.

### Core metrics

* ROC AUC
* PR AUC
* Recall@FPR=0.05
* Cohen’s d)
* Mean score shift
* Attack Success Rate
* Perturbation Budget

---

## 8. Output Structure

Each run output depends on `name_exp` and `run_name`.

The output includes:
* clean metrics
* perturbed metrics
* sweep results
* robustness curves plots

```text
lightning_logs/
└── name_exp/
    └── run_name/
        ├── def_off/
        └── robustness_curves/
            ├── l1_budget_pr_auc.png
            ├── random_noise_std_pr_auc.png
            ├── random_noise_std_recall_at_fpr5.png
            ├── random_noise_std_roc_auc.png
            ├── l0_k_pr_auc.png
            ├── l0_k_score_separation_d.png
            ├── l0_k_recall_at_fpr5.png
            ├── l1_budget_anomaly_score.png
            ├── random_noise_std_anomaly_score.png
            ├── l0_k_anomaly_score.png
            ├── l0_k_roc_auc.png
            ├── l1_budget_score_delta_mean.png
            ├── l0_k_score_p95_clean.png
            ├── random_noise_std_score_delta_mean.png
            ├── l1_budget_recall_at_fpr5.png
            ├── l1_budget_score_p95_clean.png
            ├── l0_k_score_delta_mean.png
            ├── l1_budget_roc_auc.png
            ├── random_noise_std_score_separation_d.png
            ├── l1_budget_score_separation_d.png
            ├── random_noise_std_score_p95_clean.png
            ├── robustness_summary.csv
            ├── perturbed/
            │   ├── curves.npz
            │   └── metrics.csv
            ├── perturbation_budget/
            │   ├── perturbation_budget.csv
            │   └── boundary_anomalies/
            │       ├── boundary_anomalies.pt
            │       └── plots/
            │           ├── boundary_00_heatmap.png
            │           ├── boundary_00_lineplot.png
            │           ├── boundary_01_heatmap.png
            │           ├── boundary_01_lineplot.png
            │           ├── boundary_02_heatmap.png
            │           ├── boundary_02_lineplot.png
            │           ├── boundary_03_heatmap.png
            │           ├── boundary_03_lineplot.png
            │           ├── boundary_04_heatmap.png
            │           └── boundary_04_lineplot.png
            ├── perturbations/
            │   ├── l1_budget/
            │   │   ├── 5/
            │   │   │   ├── curves.npz
            │   │   │   └── 5_metrics.csv
            │   │   ├── 10/
            │   │   │   ├── curves.npz
            │   │   │   └── 10_metrics.csv
            │   │   ├── 20/
            │   │   │   ├── curves.npz
            │   │   │   └── 20_metrics.csv
            │   │   ├── 30/
            │   │   │   ├── curves.npz
            │   │   │   └── 30_metrics.csv
            │   │   └── 40/
            │   │       ├── curves.npz
            │   │       └── 40_metrics.csv
            │   ├── random_noise_std/
            │   │   ├── 0.01/
            │   │   │   ├── curves.npz
            │   │   │   └── 0.01_metrics.csv
            │   │   ├── 0.1/
            │   │   │   ├── curves.npz
            │   │   │   └── 0.1_metrics.csv
            │   │   └── 1.0/
            │   │       ├── curves.npz
            │   │       └── 1.0_metrics.csv
            │   └── l0_k/
            │       ├── 1/
            │       │   ├── curves.npz
            │       │   └── 1_metrics.csv
            │       ├── 2/
            │       │   ├── curves.npz
            │       │   └── 2_metrics.csv
            │       ├── 5/
            │       │   ├── curves.npz
            │       │   └── 5_metrics.csv
            │       ├── 10/
            │       │   ├── curves.npz
            │       │   └── 10_metrics.csv
            │       └── 15/
            │           ├── curves.npz
            │           └── 15_metrics.csv
            └── clean/
                ├── curves.npz
                └── metrics.csv
        └── def_on/
            └── ...
```

The `curves.npz` files can be used plot ROC or PR curves.

---

## End of README
