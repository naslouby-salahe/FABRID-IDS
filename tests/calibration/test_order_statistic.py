from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from fabrid.calibration.order_statistic import (
    Threshold,
    calibrate_threshold,
    minimum_resolvable_rate,
)


def test_alpha_zero_is_always_infinite() -> None:
    scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    threshold = calibrate_threshold(scores, alpha=0.0)
    assert threshold.value == math.inf


def test_empty_calibration_set_is_infinite() -> None:
    threshold = calibrate_threshold(np.array([]), alpha=0.05)
    assert threshold.value == math.inf


def test_hand_computed_rank_examples() -> None:
    # n=5, sorted scores [1,2,3,4,5]
    scores = np.array([3.0, 1.0, 5.0, 2.0, 4.0])

    # alpha=0.2 -> r=ceil(6*0.8)=5 -> tau=s_(5)=5
    assert calibrate_threshold(scores, alpha=0.2).value == 5.0

    # alpha=0.4 -> r=ceil(6*0.6)=4 -> tau=s_(4)=4
    assert calibrate_threshold(scores, alpha=0.4).value == 4.0

    # alpha=0.05 -> r=ceil(6*0.95)=ceil(5.7)=6 > n=5 -> +inf
    assert calibrate_threshold(scores, alpha=0.05).value == math.inf


def test_strict_greater_than_ties_are_non_alerts() -> None:
    threshold = Threshold(5.0)
    scores = np.array([4.0, 5.0, 5.0, 6.0])
    np.testing.assert_array_equal(threshold.alerts(scores), [False, False, False, True])


def test_finite_sample_resolution_below_threshold_yields_infinite() -> None:
    # Targets below 1/(n+1) must yield +inf, never a silently substituted value.
    n = 100
    scores = np.linspace(0.0, 1.0, n)
    resolution = minimum_resolvable_rate(n)
    below_resolution = resolution / 2
    threshold = calibrate_threshold(scores, alpha=below_resolution)
    assert threshold.value == math.inf


def test_minimum_resolvable_rate_matches_published_final_cal_resolutions() -> None:
    assert minimum_resolvable_rate(4_955) == pytest.approx(1 / 4956, rel=1e-9)
    assert minimum_resolvable_rate(17_525) == pytest.approx(1 / 17526, rel=1e-9)
    assert minimum_resolvable_rate(1_311) == pytest.approx(1 / 1312, rel=1e-9)


def test_minimum_resolvable_rate_zero_rows_is_infinite() -> None:
    assert minimum_resolvable_rate(0) == math.inf


def test_negative_alpha_rejected() -> None:
    with pytest.raises(ValueError):
        calibrate_threshold(np.array([1.0, 2.0]), alpha=-0.1)


def test_negative_n_rejected() -> None:
    with pytest.raises(ValueError):
        minimum_resolvable_rate(-1)


def test_monotone_decreasing_threshold_in_alpha() -> None:
    """Larger alpha (more permissive) must never produce a stricter threshold."""
    rng = np.random.default_rng(0)
    scores = rng.normal(size=2_000)
    alphas = [0.001, 0.0025, 0.005, 0.01, 0.02, 0.05]
    thresholds = [calibrate_threshold(scores, a).value for a in alphas]
    for lower, higher in itertools.pairwise(thresholds):
        assert higher <= lower


def test_alert_rate_close_to_target_on_large_sample() -> None:
    rng = np.random.default_rng(1)
    n = 50_000
    scores = rng.normal(size=n)
    alpha = 0.01
    threshold = calibrate_threshold(scores, alpha=alpha)
    empirical_rate = float(np.mean(threshold.alerts(scores)))
    assert empirical_rate == pytest.approx(alpha, abs=0.002)
