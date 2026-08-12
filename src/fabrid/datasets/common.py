from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fabrid.domain.identifiers import AttackSubtypeId, ClientId
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


@dataclass(frozen=True, slots=True)
class AttackFeatureBlock:
    subtype: AttackSubtypeId
    features: FeatureMatrix


@dataclass(frozen=True, slots=True)
class DeviceDataset:
    client_id: ClientId
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

    def attack(self, subtype: AttackSubtypeId) -> FeatureMatrix:
        for block in self.attacks:
            if block.subtype == subtype:
                return block.features
        raise KeyError(subtype.value)
