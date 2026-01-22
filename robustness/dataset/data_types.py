from dataclasses import dataclass
from typing import Literal

@dataclass
class DatasetConfig:
    csv_path: str
    n_chunks: int
    test_chunk_ratio: float
    val_ratio: float
    seq_in_length: int
    stride: int
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
    checkpoint_path: str

@dataclass
class Config:
    dataset: DatasetConfig
    trainer: TrainerConfig
    opt: OptConfig