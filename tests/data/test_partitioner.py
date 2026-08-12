from __future__ import annotations

import pytest

from fabrid.config.protocol import AttackSplitFraction, BenignSplitFractions
from fabrid.data.partitioner import (
    AttackSplit,
    BenignSplit,
    RowCount,
    RowIndex,
    compute_attack_split_boundary,
    compute_benign_split_boundaries,
)

_BENIGN_FRACTIONS = BenignSplitFractions(
    train_end_fraction=0.50, frontier_end_fraction=0.70, final_cal_end_fraction=0.80
)
_ATTACK_FRACTION = AttackSplitFraction(validation_end_fraction=0.20)

# Published per-client benign split counts, used as a regression fixture for the
# boundary arithmetic.
_NBAIOT_EXPECTED_COUNTS = {
    "Danmini": (49_548, 24_774, 9_909, 4_955, 9_910),
    "Ennio": (39_100, 19_550, 7_820, 3_910, 7_820),
    "Ecobee": (13_113, 6_556, 2_623, 1_311, 2_623),
    "Philips": (175_240, 87_620, 35_047, 17_525, 35_048),
    "PT-737E": (62_154, 31_077, 12_430, 6_216, 12_431),
    "PT-838": (98_514, 49_257, 19_702, 9_852, 19_703),
    "SimpleHome-1002": (46_585, 23_292, 9_317, 4_659, 9_317),
    "SimpleHome-1003": (19_528, 9_764, 3_905, 1_953, 3_906),
    "Samsung": (52_150, 26_075, 10_430, 5_215, 10_430),
}


@pytest.mark.parametrize(
    ("client", "total_rows", "train", "frontier", "final_cal", "test"),
    [(name, *counts) for name, counts in _NBAIOT_EXPECTED_COUNTS.items()],
)
def test_benign_split_matches_published_counts(
    client: str, total_rows: int, train: int, frontier: int, final_cal: int, test: int
) -> None:
    boundaries = compute_benign_split_boundaries(RowCount(total_rows), _BENIGN_FRACTIONS)
    counts = boundaries.counts()
    assert counts[BenignSplit.TRAIN] == train, client
    assert counts[BenignSplit.FRONTIER] == frontier, client
    assert counts[BenignSplit.FINAL_CAL] == final_cal, client
    assert counts[BenignSplit.TEST] == test, client


@pytest.mark.parametrize("total_rows", [1, 2, 9, 10, 100, 1_000, 49_548, 175_240])
def test_benign_split_exclusivity_and_coverage(total_rows: int) -> None:
    """Every row index belongs to exactly one partition, all rows covered."""
    boundaries = compute_benign_split_boundaries(RowCount(total_rows), _BENIGN_FRACTIONS)
    seen = [boundaries.split_of(RowIndex(i)) for i in range(total_rows)]
    assert len(seen) == total_rows
    counts = boundaries.counts()
    assert sum(counts.values()) == total_rows
    assert all(v >= 0 for v in counts.values())


def test_benign_split_zero_rows() -> None:
    boundaries = compute_benign_split_boundaries(RowCount(0), _BENIGN_FRACTIONS)
    assert boundaries.counts() == {
        BenignSplit.TRAIN: 0,
        BenignSplit.FRONTIER: 0,
        BenignSplit.FINAL_CAL: 0,
        BenignSplit.TEST: 0,
    }


def test_benign_split_negative_total_rows_rejected() -> None:
    with pytest.raises(ValueError):
        compute_benign_split_boundaries(RowCount(-1), _BENIGN_FRACTIONS)


def test_benign_split_out_of_range_row_rejected() -> None:
    boundaries = compute_benign_split_boundaries(RowCount(10), _BENIGN_FRACTIONS)
    with pytest.raises(ValueError):
        boundaries.split_of(RowIndex(10))


def test_benign_split_negative_row_rejected() -> None:
    boundaries = compute_benign_split_boundaries(RowCount(10), _BENIGN_FRACTIONS)
    with pytest.raises(ValueError):
        boundaries.split_of(RowIndex(-1))


@pytest.mark.parametrize("total_rows", [1, 2, 5, 10, 100, 1_000])
def test_attack_split_exclusivity_and_coverage(total_rows: int) -> None:
    boundary = compute_attack_split_boundary(RowCount(total_rows), _ATTACK_FRACTION)
    seen = [boundary.split_of(RowIndex(i)) for i in range(total_rows)]
    assert len(seen) == total_rows
    counts = boundary.counts()
    assert sum(counts.values()) == total_rows


def test_attack_split_fraction() -> None:
    boundary = compute_attack_split_boundary(RowCount(100), _ATTACK_FRACTION)
    assert boundary.validation_end == 20
    assert boundary.counts() == {AttackSplit.VALIDATION: 20, AttackSplit.TEST: 80}


def test_attack_split_negative_total_rows_rejected() -> None:
    with pytest.raises(ValueError):
        compute_attack_split_boundary(RowCount(-1), _ATTACK_FRACTION)


def test_invalid_benign_fractions_rejected() -> None:
    with pytest.raises(ValueError):
        BenignSplitFractions(
            train_end_fraction=0.80, frontier_end_fraction=0.70, final_cal_end_fraction=0.90
        )


def test_invalid_attack_fraction_rejected() -> None:
    with pytest.raises(ValueError):
        AttackSplitFraction(validation_end_fraction=1.5)
