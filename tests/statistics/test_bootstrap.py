from __future__ import annotations

import numpy as np
import pytest

from fabrid.statistics.bootstrap import paired_bootstrap_ci


def test_ci_brackets_the_mean_for_consistent_positive_differences() -> None:
    differences = np.array([1.0, 1.1, 0.9, 1.05, 0.95, 1.0, 1.1, 0.9, 1.0, 1.0])
    result = paired_bootstrap_ci(differences, resamples=5000, seed=0)
    assert result.mean_difference == pytest.approx(np.mean(differences))
    assert result.confidence_interval_low < result.mean_difference < result.confidence_interval_high
    assert result.confidence_interval_low > 0  # clearly positive effect, CI should exclude 0


def test_deterministic_given_fixed_seed() -> None:
    differences = np.array([1.0, 2.0, 3.0, -1.0, 0.5])
    first = paired_bootstrap_ci(differences, resamples=1000, seed=42)
    second = paired_bootstrap_ci(differences, resamples=1000, seed=42)
    assert first == second


def test_empty_differences_rejected() -> None:
    with pytest.raises(ValueError):
        paired_bootstrap_ci(np.array([]), resamples=100, seed=0)


def test_invalid_confidence_rejected() -> None:
    with pytest.raises(ValueError):
        paired_bootstrap_ci(np.array([1.0]), resamples=100, seed=0, confidence=1.5)


def test_invalid_resamples_rejected() -> None:
    with pytest.raises(ValueError):
        paired_bootstrap_ci(np.array([1.0]), resamples=0, seed=0)
