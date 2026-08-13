from __future__ import annotations

from dataclasses import dataclass

from fabrid.domain.values import (
    BatchSize,
    FederatedRoundCount,
    LayerWidth,
    LearningRate,
    LocalEpochCount,
)


@dataclass(frozen=True, slots=True)
class DetectorHyperparameters:
    hidden_layers: tuple[LayerWidth, ...]
    learning_rate: LearningRate
    local_epochs: LocalEpochCount
    rounds: FederatedRoundCount
    batch_size: BatchSize

    def __post_init__(self) -> None:
        if not self.hidden_layers:
            raise ValueError("detector requires at least one hidden layer")


DETECTOR_HYPERPARAMETERS = DetectorHyperparameters(
    hidden_layers=(LayerWidth(64), LayerWidth(16)),
    learning_rate=LearningRate(0.001),
    local_epochs=LocalEpochCount(3),
    rounds=FederatedRoundCount(10),
    batch_size=BatchSize(64),
)
