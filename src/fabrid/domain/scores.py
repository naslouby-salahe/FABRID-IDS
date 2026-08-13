from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fabrid.domain.values import AnomalyScore, RowCount, SourceRowIndex


@dataclass(frozen=True, slots=True)
class ScoreVector:
    values: np.ndarray

    def __post_init__(self) -> None:
        if self.values.ndim != 1:
            raise ValueError("score vector must be one-dimensional")
        if not np.isfinite(self.values).all():
            raise ValueError("score vector must contain only finite values")

    @property
    def row_count(self) -> RowCount:
        return RowCount(self.values.shape[0])

    def at(self, index: SourceRowIndex) -> AnomalyScore:
        if index.value >= self.row_count.value:
            raise IndexError(index.value)
        return AnomalyScore(float(self.values[index.value]))


@dataclass(frozen=True, slots=True)
class AlertVector:
    values: np.ndarray

    def __post_init__(self) -> None:
        if self.values.ndim != 1:
            raise ValueError("alert vector must be one-dimensional")
        if self.values.dtype != np.bool_:
            raise ValueError("alert vector must contain booleans")

    @property
    def row_count(self) -> RowCount:
        return RowCount(self.values.shape[0])
