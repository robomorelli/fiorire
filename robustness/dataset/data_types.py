from dataclasses import dataclass
from typing import Literal

@dataclass
class ModelConfig:
    kernel_size: int
    base_filters: int
    num_layers: int
    flattened: bool
    compression_factor: int
    bottleneck_conv: bool
    decoder_mode: str
    activation: str
    bottleneck_activation: str

@dataclass
class DatasetConfig:
    csv_path: str
    n_seq_chunk: int
    n_wombats_ref: int
    test_chunk_ratio: float
    val_ratio: float
    seq_in_length: int
    seq_stride_train: int
    seq_stride_val: int
    seq_stride_test: int  
    batch_size: int
    num_workers: int
    val_anomaly_ratio: float
    val_shuffle_augmented: bool
    delta_min: float
    delta_max: float

@dataclass
class TrainerConfig:
    accelerator: str
    devices: int
    epochs: int
    precision: Literal[16, 32, 64, "16-true", "transformer-engine", "transformer-engine-float16"]
    out_dir: str

@dataclass
class OptConfig:
    lr: float
    batch_size: int
    lr_patience: int
    lr_factor: float        # fattore per ReduceLROnPlateau
    lr_min: float           # lr minimo
    es_patience: int
    checkpoint_path: str

@dataclass
class DefenseConfig:
    alpha: float
    num_iter: int
    epsilon: float        # adversarial perturbation strength
    lambda_latent: float  # peso regolarizzazione latente
    p_adv: float          # frazione batch adversarial
    pgd_steps: int

@dataclass
class Config:
    model: ModelConfig
    dataset: DatasetConfig
    trainer: TrainerConfig
    opt: OptConfig
    defense: DefenseConfig
