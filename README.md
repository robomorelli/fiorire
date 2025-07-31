
# Fiorire — Semi-Supervised Time Series Anomaly Detection with Conv & LSTM base model

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
## 📁 Repository Structure Overview

├── config/ # Configuration files (.yaml) for each model and tuning
├── dataset/ # Scripts and utilities for dataset handling
├── model/ # Model definitions and training classes
├── scripts/ # Scripts to launch training locally or on HPC
├── utils/ # Utility functions (e.g., preprocessing, plotting)
├── launch_wrapper.sh # Wrapper to launch jobs with Ray Tune
└── README.md # This file

## 🚀 How to Run

### ➤ Option 1: On HPC (recommended for large-scale tuning)

Use the provided `launch_wrapper.sh` script:

```bash
Example of usage:
#sh launch_wrapper.sh num_nodes 2 num_gpus 1 num_cpus 16 config_file conv_ae1D
all the args not specified are replaced from the default of the bash script (see the default value into the bash file) or from the main args default arguments
