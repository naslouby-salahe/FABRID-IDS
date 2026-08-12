from __future__ import annotations

import pytest

from fabrid.evaluation.record_level import AttackSubtype, ClientId
from fabrid.frontier.utility import SubtypeConfusionCounts, build_utility_curve, client_utility


def test_true_positive_rate() -> None:
    counts = SubtypeConfusionCounts(true_positive=8, false_negative=2)
    assert counts.true_positive_rate() == pytest.approx(0.8)


def test_negative_counts_rejected() -> None:
    with pytest.raises(ValueError):
        SubtypeConfusionCounts(true_positive=-1, false_negative=2)


def test_zero_total_rejected() -> None:
    with pytest.raises(ValueError):
        SubtypeConfusionCounts(true_positive=0, false_negative=0)


def test_client_utility_averages_by_subtype_not_rows() -> None:
    # "udp" has far more rows than "scan" but must not dominate the mean.
    counts = {
        AttackSubtype("scan"): SubtypeConfusionCounts(true_positive=10, false_negative=0),
        AttackSubtype("udp"): SubtypeConfusionCounts(true_positive=0, false_negative=10_000),
    }
    assert client_utility(counts) == pytest.approx(0.5)


def test_client_utility_requires_at_least_one_subtype() -> None:
    with pytest.raises(ValueError):
        client_utility({})


def test_build_utility_curve_matches_candidate_order() -> None:
    grid = (0.0, 0.01, 0.02)
    counts_by_candidate = [
        {AttackSubtype("scan"): SubtypeConfusionCounts(0, 10)},
        {AttackSubtype("scan"): SubtypeConfusionCounts(5, 5)},
        {AttackSubtype("scan"): SubtypeConfusionCounts(10, 0)},
    ]
    curve = build_utility_curve(ClientId("1"), grid, counts_by_candidate)
    assert curve.utility == pytest.approx((0.0, 0.5, 1.0))
    assert curve.alpha_grid == grid


def test_build_utility_curve_length_mismatch_rejected() -> None:
    with pytest.raises(ValueError):
        build_utility_curve(
            ClientId("1"),
            (0.0, 0.01),
            [{AttackSubtype("scan"): SubtypeConfusionCounts(5, 5)}],
        )
