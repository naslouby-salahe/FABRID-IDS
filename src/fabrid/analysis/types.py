from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HypothesisIndex:
    value: int

    def __post_init__(self) -> None:
        if self.value < 0:
            raise ValueError(f"hypothesis index must be non-negative, got {self.value}")


@dataclass(frozen=True, slots=True)
class PercentagePointDifference:
    value: float
