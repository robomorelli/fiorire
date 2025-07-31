# Fiorire — Semi-Supervised Time Series Anomaly Detection with ConvAE & LSTM

**Fiorire** is a modular, scalable framework for training and tuning time series models — including **Convolutional Autoencoders**, **LSTM-based Autoencoders**, and **LSTM Predictive Models** — in a **semi-supervised anomaly detection** setup.

It leverages **Ray Tune** for distributed hyperparameter optimization, and supports both local and HPC deployment.

---

## 🧠 Core Features

- ✅ **ConvAE1D**, **LSTM Autoencoder**, and **LSTM Predictor** architectures
- ✅ Training in **semi-supervised mode** for anomaly detection (train on normal data)
- ✅ Built-in **Ray Tune** integration for scalable hyperparameter search
- ✅ Easily extensible with custom models
- ✅ Ready for **HPC** or **local cluster** execution

---

## 🚀 How to Run

### ➤ Option 1: On HPC (recommended for large-scale tuning)

Use the provided `launch_wrapper.sh` script:

```bash
```bash
Example of usage:
sh launch_wrapper.sh num_nodes 2 num_gpus 1 num_cpus 16 config_file conv_ae1D
all the args not specified are replaced from the default of the bash script (see the default value into the bash file) or from the main args default arguments
```

This script handles Ray cluster setup and experiment launch with appropriate environment variables.

### ➤ Option 2: Locally (dev/test)

1. Start a Ray head node:

```bash
ray start --head
```

2. Run the experiment script manually:

```bash
python run_experiment.py --config config/your_config.yaml
```

---

## 🛠 How to Add a New Model

To integrate a new model into **fiorire**, follow these 3 steps:

### 1. Add a New Config File

Create a YAML config in `config/`, e.g., `my_model_config.yaml`:

- Include dataset paths, model defaults, and `tune_config` block for Ray Tune.
- Example:

```yaml
tune_config:
  model.hidden_dim: tune.choice([64, 128])
  opt.lr: tune.choice([0.001, 0.0003])
  latent_dim: tune.choice([16, 32])
model:
  name: my_model_name
```

### 2. Define the Model

Add a class for your model in `model/your_model_def.py`. Example structure:

```python
class MyModel(nn.Module):
    def __init__(self, ...):
        ...
    def forward(self, x):
        ...
```

### 3. Add a Trainer

Inside `model/your_model_trainer.py`, define the training logic:

```python
def train_my_model(config, cfg):
    model = MyModel(...)
    ...
    return final_results
```

Ensure it follows the interface expected by `trainable.py`.

---

## 🔧 Configuration

YAML config files (in `config/`) define datasets, tuning options, and model parameters.

Key fields include:

- `tune_config`: search space for Ray Tune
- `dataset`: path, features, and split ratio
- `model`: type and architecture settings
- `resources`: CPU/GPU per trial
- `opt`: loss, metrics, LR schedulers

Example: `config/conv_ae_1D_config.yaml`

---

## 🗃 Repository Overview

```
fiorire/
├── config/                   # YAML configuration files
│   └── conv_ae_1D_config.yaml
├── data/                     # Serialized input data (.pkl)
│   └── ...
├── extract_config.py         # Helpers for parsing configs
├── model/                    # Model definitions and trainers
│   ├── conv_ae1d.py
│   ├── lstm_ae.py
│   ├── lstm_predictor.py
│   └── ...
├── trainable.py              # Ray Tune Trainable interface
├── run_experiment.py         # Launch script
├── launch_wrapper.sh         # HPC wrapper (SLURM or bash)
└── requirements.txt          # Dependencies
```

---

## 📊 Output & Results

- Ray Tune logs each trial with training/validation loss
- Best checkpoints saved automatically
- TensorBoard supported (optional)

---

## 🔍 Anomaly Detection

Trained models are intended to reconstruct or predict **normal** sequences, and detect anomalies when:

- Reconstruction loss is high (ConvAE, LSTM AE)
- Prediction error is high (LSTM predictor)

---

## 📬 Contributing

PRs and issues are welcome. Add your model via the 3‑step flow above and open a PR!

---

## 📝 License

MIT License