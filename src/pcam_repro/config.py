"""Typed TOML configuration used by all experiment tracks."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class DataConfig:
    root: str = "data/pcam"
    image_size: int = 96
    batch_size: int = 128
    eval_batch_size: int | None = None
    workers: int = 4
    labeled_fraction: float = 1.0
    augmentation: str = "pathology"
    normalize: str = "pcam"
    subset_seed: int = 17
    pin_memory: bool = True


@dataclass(slots=True)
class ModelConfig:
    name: str = "small_cnn"
    pretrained: bool = False
    checkpoint: str = ""
    freeze_encoder: bool = False
    num_classes: int = 1
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OptimizerConfig:
    name: str = "adamw"
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    momentum: float = 0.9


@dataclass(slots=True)
class SchedulerConfig:
    name: str = "cosine"
    warmup_epochs: int = 5
    minimum_learning_rate: float = 1e-6


@dataclass(slots=True)
class TrainingConfig:
    mode: str = "supervised"
    epochs: int = 100
    seed: int = 17
    amp: bool = True
    gradient_clip_norm: float = 1.0
    accumulation_steps: int = 1
    monitor: str = "auc_roc"
    patience: int = 20
    consistency_weight: float = 1.0
    ema_decay: float = 0.999
    temperature: float = 0.2
    noise_rate: float = 0.0
    noise_seed: int = 23
    robust_loss: str = "bce"


@dataclass(slots=True)
class OutputConfig:
    directory: str = "runs"
    experiment_name: str = "pcam_experiment"
    save_predictions: bool = True


@dataclass(slots=True)
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def _construct(data: dict[str, Any]) -> ExperimentConfig:
    allowed = {"data", "model", "optimizer", "scheduler", "training", "output"}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"Unknown top-level configuration sections: {sorted(unknown)}")
    return ExperimentConfig(
        data=DataConfig(**data.get("data", {})),
        model=ModelConfig(**data.get("model", {})),
        optimizer=OptimizerConfig(**data.get("optimizer", {})),
        scheduler=SchedulerConfig(**data.get("scheduler", {})),
        training=TrainingConfig(**data.get("training", {})),
        output=OutputConfig(**data.get("output", {})),
    )


def validate_config(config: ExperimentConfig) -> None:
    if not 0 < config.data.labeled_fraction <= 1:
        raise ValueError("data.labeled_fraction must be in (0, 1].")
    if config.data.image_size <= 0 or config.data.batch_size <= 0:
        raise ValueError("Image size and training batch size must be positive.")
    if (
        config.data.eval_batch_size is not None
        and config.data.eval_batch_size <= 0
    ):
        raise ValueError("Evaluation batch size must be positive when provided.")
    if config.training.mode not in {"supervised", "mean_teacher", "simclr"}:
        raise ValueError("training.mode must be supervised, mean_teacher, or simclr.")
    if not 0 <= config.training.noise_rate < 0.5:
        raise ValueError("training.noise_rate must be in [0, 0.5).")
    if config.model.num_classes != 1:
        raise ValueError("PCam is a binary task; num_classes must be 1.")


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    config = _construct(raw)
    validate_config(config)
    return config
