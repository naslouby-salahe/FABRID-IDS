from __future__ import annotations

import math
from dataclasses import dataclass

from fabrid.domain.enums import AttackSplit, BenignSplit
from fabrid.domain.values import RowCount, SourceRowIndex
from fabrid.protocol.models import AttackSplitFraction, BenignSplitFractions


@dataclass(frozen=True, slots=True)
class BenignSplitCounts:
    train: RowCount
    frontier: RowCount
    final_cal: RowCount
    test: RowCount

    @property
    def total(self) -> RowCount:
        return RowCount(
            self.train.value
            + self.frontier.value
            + self.final_cal.value
            + self.test.value
        )


@dataclass(frozen=True, slots=True)
class AttackSplitCounts:
    validation: RowCount
    test: RowCount

    @property
    def total(self) -> RowCount:
        return RowCount(self.validation.value + self.test.value)


@dataclass(frozen=True, slots=True)
class BenignSplitBoundaries:
    train_end: RowCount
    frontier_end: RowCount
    final_cal_end: RowCount
    total_rows: RowCount

    def __post_init__(self) -> None:
        values = (
            self.train_end.value,
            self.frontier_end.value,
            self.final_cal_end.value,
            self.total_rows.value,
        )
        if values != tuple(sorted(values)):
            raise ValueError("benign split boundaries must be monotonically increasing")

    def split_of(self, row_index: SourceRowIndex) -> BenignSplit:
        if row_index.value >= self.total_rows.value:
            raise ValueError(f"row index {row_index.value} is outside benign population")
        if row_index.value < self.train_end.value:
            return BenignSplit.TRAIN
        if row_index.value < self.frontier_end.value:
            return BenignSplit.FRONTIER
        if row_index.value < self.final_cal_end.value:
            return BenignSplit.FINAL_CAL
        return BenignSplit.TEST

    def counts(self) -> BenignSplitCounts:
        return BenignSplitCounts(
            train=self.train_end,
            frontier=RowCount(self.frontier_end.value - self.train_end.value),
            final_cal=RowCount(self.final_cal_end.value - self.frontier_end.value),
            test=RowCount(self.total_rows.value - self.final_cal_end.value),
        )


@dataclass(frozen=True, slots=True)
class AttackSplitBoundary:
    validation_end: RowCount
    total_rows: RowCount

    def __post_init__(self) -> None:
        if self.validation_end.value > self.total_rows.value:
            raise ValueError("attack validation boundary may not exceed total rows")

    def split_of(self, row_index: SourceRowIndex) -> AttackSplit:
        if row_index.value >= self.total_rows.value:
            raise ValueError(f"row index {row_index.value} is outside attack population")
        if row_index.value < self.validation_end.value:
            return AttackSplit.VALIDATION
        return AttackSplit.TEST

    def counts(self) -> AttackSplitCounts:
        return AttackSplitCounts(
            validation=self.validation_end,
            test=RowCount(self.total_rows.value - self.validation_end.value),
        )


def compute_benign_split_boundaries(
    total_rows: RowCount,
    fractions: BenignSplitFractions,
) -> BenignSplitBoundaries:
    total = total_rows.value
    return BenignSplitBoundaries(
        train_end=RowCount(math.floor(fractions.train_end.value * total)),
        frontier_end=RowCount(math.floor(fractions.frontier_end.value * total)),
        final_cal_end=RowCount(math.floor(fractions.final_cal_end.value * total)),
        total_rows=total_rows,
    )


def compute_attack_split_boundary(
    total_rows: RowCount,
    fraction: AttackSplitFraction,
) -> AttackSplitBoundary:
    return AttackSplitBoundary(
        validation_end=RowCount(
            math.floor(fraction.validation_end.value * total_rows.value)
        ),
        total_rows=total_rows,
    )
