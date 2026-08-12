"""Deterministic source-order partitioning (roadmap sections 24-26).

Pure index-arithmetic module: given a row count, computes the exact partition
boundaries the roadmap specifies. Row order must already be source order
(chronological as emitted by the original dataset); this module does not read
or reorder data, it only assigns partition membership by position.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum


class BenignSplit(StrEnum):
    TRAIN = "benign_train"
    FRONTIER = "benign_frontier"
    FINAL_CAL = "benign_final_cal"
    TEST = "benign_test"


class AttackSplit(StrEnum):
    VALIDATION = "attack_validation"
    TEST = "attack_test"


_BENIGN_TRAIN_FRACTION = 0.50
_BENIGN_FRONTIER_BOUNDARY_FRACTION = 0.70
_BENIGN_FINAL_CAL_BOUNDARY_FRACTION = 0.80
_ATTACK_VALIDATION_FRACTION = 0.20


@dataclass(frozen=True, slots=True)
class BenignSplitBoundaries:
    """Half-open row-index boundaries: [0, i1), [i1, i2), [i2, i3), [i3, n)."""

    i1: int
    i2: int
    i3: int
    n: int

    def __post_init__(self) -> None:
        if not (0 <= self.i1 <= self.i2 <= self.i3 <= self.n):
            raise ValueError(
                f"invalid benign split boundaries for n={self.n}: "
                f"i1={self.i1}, i2={self.i2}, i3={self.i3}"
            )

    def split_of(self, row_index: int) -> BenignSplit:
        if not (0 <= row_index < self.n):
            raise ValueError(f"row_index {row_index} out of range [0, {self.n})")
        if row_index < self.i1:
            return BenignSplit.TRAIN
        if row_index < self.i2:
            return BenignSplit.FRONTIER
        if row_index < self.i3:
            return BenignSplit.FINAL_CAL
        return BenignSplit.TEST

    def counts(self) -> dict[BenignSplit, int]:
        return {
            BenignSplit.TRAIN: self.i1,
            BenignSplit.FRONTIER: self.i2 - self.i1,
            BenignSplit.FINAL_CAL: self.i3 - self.i2,
            BenignSplit.TEST: self.n - self.i3,
        }


@dataclass(frozen=True, slots=True)
class AttackSplitBoundary:
    """Half-open row-index boundary: [0, j) -> VALIDATION, [j, n) -> TEST."""

    j: int
    n: int

    def __post_init__(self) -> None:
        if not (0 <= self.j <= self.n):
            raise ValueError(f"invalid attack split boundary for n={self.n}: j={self.j}")

    def split_of(self, row_index: int) -> AttackSplit:
        if not (0 <= row_index < self.n):
            raise ValueError(f"row_index {row_index} out of range [0, {self.n})")
        return AttackSplit.VALIDATION if row_index < self.j else AttackSplit.TEST

    def counts(self) -> dict[AttackSplit, int]:
        return {AttackSplit.VALIDATION: self.j, AttackSplit.TEST: self.n - self.j}


def compute_benign_split_boundaries(n: int) -> BenignSplitBoundaries:
    """Roadmap section 24: i1=floor(0.5n), i2=floor(0.7n), i3=floor(0.8n)."""
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    i1 = math.floor(_BENIGN_TRAIN_FRACTION * n)
    i2 = math.floor(_BENIGN_FRONTIER_BOUNDARY_FRACTION * n)
    i3 = math.floor(_BENIGN_FINAL_CAL_BOUNDARY_FRACTION * n)
    return BenignSplitBoundaries(i1=i1, i2=i2, i3=i3, n=n)


def compute_attack_split_boundary(n: int) -> AttackSplitBoundary:
    """Roadmap section 26: j_a = floor(0.2 * n_a)."""
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    j = math.floor(_ATTACK_VALIDATION_FRACTION * n)
    return AttackSplitBoundary(j=j, n=n)
