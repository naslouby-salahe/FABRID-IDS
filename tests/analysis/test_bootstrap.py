from __future__ import annotations

import pytest

from fabrid.analysis.bootstrap import paired_bootstrap_ci
from fabrid.domain.values import AnalysisSeed, MetricDifference, Probability, RowCount


def _differences(*values: float) -> tuple[MetricDifference, ...]:
    return tuple(MetricDifference(value) for value in values)


def test_ci_brackets_mean_for_consistent_positive_differences() -> None:
    differences = _differences(1.0, 1.1, 0.9, 1.05, 0.95, 1.0, 1.1, 0.9, 1.0, 1.0)

    result = paired_bootstrap_ci(
        paired_differences=differences,
        resamples=RowCount(5_000),
        seed=AnalysisSeed(0),
        confidence=Probability(0.95),
    )

    assert result.mean_difference.value == pytest.approx(1.0)
    assert result.confidence_interval_low.value < result.mean_difference.value
    assert result.mean_difference.value < result.confidence_interval_high.value
    assert result.confidence_interval_low.value > 0.0


def test_bootstrap_is_deterministic_for_analysis_seed() -> None:
    differences = _differences(1.0, 2.0, 3.0, -1.0, 0.5)
    arguments = {
        "paired_differences": differences,
        "resamples": RowCount(1_000),
        "seed": AnalysisSeed(42),
        "confidence": Probability(0.95),
    }

    assert paired_bootstrap_ci(**arguments) == paired_bootstrap_ci(**arguments)


def test_empty_paired_differences_are_rejected() -> None:
    with pytest.raises(ValueError):
        paired_bootstrap_ci(
            paired_differences=(),
            resamples=RowCount(100),
            seed=AnalysisSeed(0),
            confidence=Probability(0.95),
        )
