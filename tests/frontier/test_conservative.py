from __future__ import annotations

import pytest

from fabrid.evaluation.record_level import AttackSubtype, ClientId
from fabrid.frontier.conservative import (
    build_conservative_utility_curve,
    conservative_client_utility,
    one_sided_lower_confidence_bound,
)
from fabrid.frontier.utility import SubtypeConfusionCounts


def test_lcb_below_raw_tpr() -> None:
    counts = SubtypeConfusionCounts(true_positive=8, false_negative=2)
    lcb = one_sided_lower_confidence_bound(counts)
    assert lcb < counts.true_positive_rate()
    assert lcb > 0


def test_lcb_zero_successes_is_zero() -> None:
    counts = SubtypeConfusionCounts(true_positive=0, false_negative=10)
    assert one_sided_lower_confidence_bound(counts) == 0.0


def test_lcb_all_successes_positive_but_below_one() -> None:
    counts = SubtypeConfusionCounts(true_positive=10, false_negative=0)
    lcb = one_sided_lower_confidence_bound(counts)
    assert 0 < lcb < 1.0


def test_lcb_tighter_with_more_data() -> None:
    small = SubtypeConfusionCounts(true_positive=8, false_negative=2)
    large = SubtypeConfusionCounts(true_positive=800, false_negative=200)
    # same raw TPR (0.8), but more data -> LCB closer to raw rate.
    assert one_sided_lower_confidence_bound(large) > one_sided_lower_confidence_bound(small)


def test_invalid_confidence_rejected() -> None:
    counts = SubtypeConfusionCounts(true_positive=5, false_negative=5)
    with pytest.raises(ValueError):
        one_sided_lower_confidence_bound(counts, confidence=1.5)


def test_conservative_client_utility_averages_lcb_by_subtype() -> None:
    counts = {
        AttackSubtype("scan"): SubtypeConfusionCounts(true_positive=100, false_negative=0),
        AttackSubtype("udp"): SubtypeConfusionCounts(true_positive=0, false_negative=100),
    }
    utility = conservative_client_utility(counts)
    assert utility < 0.5  # scan's LCB < 1.0, udp's LCB == 0.0 -> mean below the raw-TPR mean of 0.5


def test_build_conservative_utility_curve_length_mismatch_rejected() -> None:
    with pytest.raises(ValueError):
        build_conservative_utility_curve(
            ClientId("1"),
            (0.0, 0.01),
            [{AttackSubtype("scan"): SubtypeConfusionCounts(5, 5)}],
        )
