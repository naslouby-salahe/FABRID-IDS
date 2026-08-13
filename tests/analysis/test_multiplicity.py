from __future__ import annotations

import pytest

from fabrid.analysis.multiplicity import holm_correction
from fabrid.domain.enums import HypothesisDecision
from fabrid.domain.values import Probability


def _probabilities(*values: float) -> tuple[Probability, ...]:
    return tuple(Probability(value) for value in values)


def test_holm_rejects_entire_strong_family() -> None:
    results = holm_correction(
        _probabilities(0.001, 0.002, 0.003),
        Probability(0.05),
    )

    assert all(result.decision is HypothesisDecision.REJECT for result in results)


def test_holm_step_down_retains_after_first_failure() -> None:
    results = holm_correction(
        _probabilities(0.01, 0.5, 0.02),
        Probability(0.05),
    )

    assert results[0].decision is HypothesisDecision.REJECT
    assert results[2].decision is HypothesisDecision.REJECT
    assert results[1].decision is HypothesisDecision.RETAIN


def test_adjusted_probabilities_are_monotone_by_raw_probability() -> None:
    results = holm_correction(
        _probabilities(0.001, 0.2, 0.01, 0.3),
        Probability(0.05),
    )
    ordered = tuple(sorted(results, key=lambda result: result.p_value.value))
    adjusted = tuple(result.adjusted_p_value.value for result in ordered)

    assert adjusted == tuple(sorted(adjusted))


def test_empty_hypothesis_family_is_rejected() -> None:
    with pytest.raises(ValueError):
        holm_correction((), Probability(0.05))
