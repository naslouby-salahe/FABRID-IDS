"""Deterministic source-order partitioning of benign and attack rows.

Pure index-arithmetic module: given a row count and the configured split
fractions, computes exact partition boundaries. Row order must already be
source order (chronological as emitted by the original dataset); this module
does not read or reorder data, it only assigns partition membership by
position.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import NewType

from fabrid.config.protocol import AttackSplitFraction, BenignSplitFractions

RowCount = NewType("RowCount", int)
RowIndex = NewType("RowIndex", int)


class BenignSplit(StrEnum):
    TRAIN = "benign_train"
    FRONTIER = "benign_frontier"
    FINAL_CAL = "benign_final_cal"
    TEST = "benign_test"


class AttackSplit(StrEnum):
    VALIDATION = "attack_validation"
    TEST = "attack_test"


@dataclass(frozen=True, slots=True)
class BenignSplitBoundaries:
    """Half-open row-index boundaries partitioning ``[0, total_rows)`` into four splits."""

    train_end: RowCount
    frontier_end: RowCount
    final_cal_end: RowCount
    total_rows: RowCount

    def __post_init__(self) -> None:
        if not (0 <= self.train_end <= self.frontier_end <= self.final_cal_end <= self.total_rows):
            raise ValueError(
                f"invalid benign split boundaries for total_rows={self.total_rows}: "
                f"train_end={self.train_end}, frontier_end={self.frontier_end}, "
                f"final_cal_end={self.final_cal_end}"
            )

    def split_of(self, row_index: RowIndex) -> BenignSplit:
        if not (0 <= row_index < self.total_rows):
            raise ValueError(f"row_index {row_index} out of range [0, {self.total_rows})")
        if row_index < self.train_end:
            return BenignSplit.TRAIN
        if row_index < self.frontier_end:
            return BenignSplit.FRONTIER
        if row_index < self.final_cal_end:
            return BenignSplit.FINAL_CAL
        return BenignSplit.TEST

    def counts(self) -> dict[BenignSplit, RowCount]:
        return {
            BenignSplit.TRAIN: self.train_end,
            BenignSplit.FRONTIER: RowCount(self.frontier_end - self.train_end),
            BenignSplit.FINAL_CAL: RowCount(self.final_cal_end - self.frontier_end),
            BenignSplit.TEST: RowCount(self.total_rows - self.final_cal_end),
        }


@dataclass(frozen=True, slots=True)
class AttackSplitBoundary:
    """Half-open row-index boundary: ``[0, validation_end)`` VALIDATION, rest TEST."""

    validation_end: RowCount
    total_rows: RowCount

    def __post_init__(self) -> None:
        if not (0 <= self.validation_end <= self.total_rows):
            raise ValueError(
                f"invalid attack split boundary for total_rows={self.total_rows}: "
                f"validation_end={self.validation_end}"
            )

    def split_of(self, row_index: RowIndex) -> AttackSplit:
        if not (0 <= row_index < self.total_rows):
            raise ValueError(f"row_index {row_index} out of range [0, {self.total_rows})")
        return AttackSplit.VALIDATION if row_index < self.validation_end else AttackSplit.TEST

    def counts(self) -> dict[AttackSplit, RowCount]:
        return {
            AttackSplit.VALIDATION: self.validation_end,
            AttackSplit.TEST: RowCount(self.total_rows - self.validation_end),
        }


def compute_benign_split_boundaries(
    total_rows: RowCount, fractions: BenignSplitFractions
) -> BenignSplitBoundaries:
    if total_rows < 0:
        raise ValueError(f"total_rows must be non-negative, got {total_rows}")
    return BenignSplitBoundaries(
        train_end=RowCount(math.floor(fractions.train_end_fraction * total_rows)),
        frontier_end=RowCount(math.floor(fractions.frontier_end_fraction * total_rows)),
        final_cal_end=RowCount(math.floor(fractions.final_cal_end_fraction * total_rows)),
        total_rows=total_rows,
    )


def compute_attack_split_boundary(
    total_rows: RowCount, fraction: AttackSplitFraction
) -> AttackSplitBoundary:
    if total_rows < 0:
        raise ValueError(f"total_rows must be non-negative, got {total_rows}")
    return AttackSplitBoundary(
        validation_end=RowCount(math.floor(fraction.validation_end_fraction * total_rows)),
        total_rows=total_rows,
    )
