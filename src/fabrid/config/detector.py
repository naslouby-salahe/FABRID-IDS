"""Typed loader for the frozen detector hyperparameter config (`detector.yaml`)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fabrid.config.protocol import read_yaml_mapping

DETECTOR_CONFIG_PATH = Path(__file__).with_name("detector.yaml")


@dataclass(frozen=True, slots=True)
class DetectorHyperparameters:
    hidden_dims: tuple[int, ...]
    learning_rate: float
    local_epochs: int
    rounds: int
    batch_size: int

    def __post_init__(self) -> None:
        if not self.hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer size")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.local_epochs < 1:
            raise ValueError(f"local_epochs must be at least 1, got {self.local_epochs}")
        if self.rounds < 1:
            raise ValueError(f"rounds must be at least 1, got {self.rounds}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {self.batch_size}")


def load_detector_hyperparameters(path: Path = DETECTOR_CONFIG_PATH) -> DetectorHyperparameters:
    payload = read_yaml_mapping(path)
    return DetectorHyperparameters(
        hidden_dims=tuple(int(dim) for dim in payload["architecture"]["hidden_dims"]),
        learning_rate=float(payload["training"]["learning_rate"]),
        local_epochs=int(payload["training"]["local_epochs"]),
        rounds=int(payload["training"]["rounds"]),
        batch_size=int(payload["training"]["batch_size"]),
    )


def load_detector_seeds(path: Path = DETECTOR_CONFIG_PATH) -> tuple[int, ...]:
    payload = read_yaml_mapping(path)
    return tuple(int(seed) for seed in payload["seeds"])
