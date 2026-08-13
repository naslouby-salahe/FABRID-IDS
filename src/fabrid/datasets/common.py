from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fabrid.domain.identifiers import (
    AttackSubtypeId,
    ClientId,
    SourceFileId,
)
from fabrid.domain.values import FeatureCount, RowCount


@dataclass(frozen=True, slots=True)
class FeatureMatrix:
    values: np.ndarray

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ValueError(
                f"feature matrix must be two-dimensional, got {self.values.ndim}"
            )
        if not np.issubdtype(self.values.dtype, np.number):
            raise ValueError("feature matrix must contain numeric values")
        if not np.isfinite(self.values).all():
            raise ValueError("feature matrix must contain only finite values")

    @property
    def row_count(self) -> RowCount:
        return RowCount(self.values.shape[0])

    @property
    def feature_count(self) -> FeatureCount:
        return FeatureCount(self.values.shape[1])

    def prefix(self, row_count: RowCount) -> FeatureMatrix:
        if row_count.value > self.row_count.value:
            raise ValueError("requested prefix exceeds feature matrix row count")
        return FeatureMatrix(self.values[: row_count.value])

    def between(self, start: RowCount, end: RowCount) -> FeatureMatrix:
        if start.value > end.value:
            raise ValueError("feature matrix range start may not exceed end")
        if end.value > self.row_count.value:
            raise ValueError("feature matrix range exceeds row count")
        return FeatureMatrix(self.values[start.value : end.value])


@dataclass(frozen=True, slots=True)
class AttackFeatureBlock:
    subtype: AttackSubtypeId
    source_file: SourceFileId
    features: FeatureMatrix


@dataclass(frozen=True, slots=True)
class DeviceDataset:
    client_id: ClientId
    benign_source_file: SourceFileId
    benign: FeatureMatrix
    attacks: tuple[AttackFeatureBlock, ...]

    def __post_init__(self) -> None:
        subtypes = tuple(block.subtype for block in self.attacks)
        if len(set(subtypes)) != len(subtypes):
            raise ValueError("device dataset contains duplicate attack subtypes")
        if any(
            block.features.feature_count != self.benign.feature_count
            for block in self.attacks
        ):
            raise ValueError("benign and attack feature matrices must share feature width")

    def attack(self, subtype: AttackSubtypeId) -> AttackFeatureBlock:
        for block in self.attacks:
            if block.subtype == subtype:
                return block
        raise KeyError(subtype.value)
