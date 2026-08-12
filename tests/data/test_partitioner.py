from __future__ import annotations

import pytest

from fabrid.data.partitioner import (
    AttackSplit,
    BenignSplit,
    compute_attack_split_boundary,
    compute_benign_split_boundaries,
)

# Roadmap section 25: exact published per-client benign split counts.
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
    ("client", "n", "train", "frontier", "final_cal", "test"),
    [(name, *counts) for name, counts in _NBAIOT_EXPECTED_COUNTS.items()],
)
def test_benign_split_matches_roadmap_table(
    client: str, n: int, train: int, frontier: int, final_cal: int, test: int
) -> None:
    boundaries = compute_benign_split_boundaries(n)
    counts = boundaries.counts()
    assert counts[BenignSplit.TRAIN] == train, client
    assert counts[BenignSplit.FRONTIER] == frontier, client
    assert counts[BenignSplit.FINAL_CAL] == final_cal, client
    assert counts[BenignSplit.TEST] == test, client


@pytest.mark.parametrize("n", [1, 2, 9, 10, 100, 1_000, 49_548, 175_240])
def test_benign_split_exclusivity_and_coverage(n: int) -> None:
    """T01: every row index belongs to exactly one partition, all rows covered."""
    boundaries = compute_benign_split_boundaries(n)
    seen = [boundaries.split_of(i) for i in range(n)]
    assert len(seen) == n
    counts = boundaries.counts()
    assert sum(counts.values()) == n
    assert all(v >= 0 for v in counts.values())


def test_benign_split_zero_rows() -> None:
    boundaries = compute_benign_split_boundaries(0)
    assert boundaries.counts() == {
        BenignSplit.TRAIN: 0,
        BenignSplit.FRONTIER: 0,
        BenignSplit.FINAL_CAL: 0,
        BenignSplit.TEST: 0,
    }


def test_benign_split_negative_n_rejected() -> None:
    with pytest.raises(ValueError):
        compute_benign_split_boundaries(-1)


def test_benign_split_out_of_range_row_rejected() -> None:
    boundaries = compute_benign_split_boundaries(10)
    with pytest.raises(ValueError):
        boundaries.split_of(10)
    with pytest.raises(ValueError):
        boundaries.split_of(-1)


@pytest.mark.parametrize("n", [1, 2, 5, 10, 100, 1_000])
def test_attack_split_exclusivity_and_coverage(n: int) -> None:
    boundary = compute_attack_split_boundary(n)
    seen = [boundary.split_of(i) for i in range(n)]
    assert len(seen) == n
    counts = boundary.counts()
    assert sum(counts.values()) == n


def test_attack_split_fraction() -> None:
    boundary = compute_attack_split_boundary(100)
    assert boundary.j == 20
    assert boundary.counts() == {AttackSplit.VALIDATION: 20, AttackSplit.TEST: 80}


def test_attack_split_negative_n_rejected() -> None:
    with pytest.raises(ValueError):
        compute_attack_split_boundary(-1)
