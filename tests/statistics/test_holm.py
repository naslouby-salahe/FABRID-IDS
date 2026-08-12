from __future__ import annotations

import pytest

from fabrid.statistics.holm import holm_correction


def test_all_significant_reject_all() -> None:
    results = holm_correction((0.001, 0.002, 0.003), alpha=0.05)
    assert all(r.reject_at_alpha for r in results)


def test_step_down_stops_rejecting_after_first_failure() -> None:
    # sorted p-values: 0.01, 0.02, 0.5 -> thresholds 0.05/3, 0.05/2, 0.05/1
    p_values = (0.01, 0.5, 0.02)
    results = holm_correction(p_values, alpha=0.05)
    by_index = {r.original_index: r for r in results}
    assert by_index[0].reject_at_alpha  # p=0.01 <= 0.05/3
    assert by_index[2].reject_at_alpha  # p=0.02 <= 0.05/2
    assert not by_index[1].reject_at_alpha  # p=0.5 > 0.05/1


def test_adjusted_p_values_are_monotone_nondecreasing_in_rank() -> None:
    p_values = (0.001, 0.2, 0.01, 0.3)
    results = holm_correction(p_values, alpha=0.05)
    ordered = sorted(results, key=lambda r: r.p_value)
    adjusted = [r.adjusted_p_value for r in ordered]
    assert adjusted == sorted(adjusted)


def test_adjusted_p_values_capped_at_one() -> None:
    results = holm_correction((0.9, 0.9, 0.9), alpha=0.05)
    assert all(r.adjusted_p_value <= 1.0 for r in results)


def test_empty_p_values_rejected() -> None:
    with pytest.raises(ValueError):
        holm_correction((), alpha=0.05)


def test_invalid_alpha_rejected() -> None:
    with pytest.raises(ValueError):
        holm_correction((0.1,), alpha=1.5)


def test_out_of_range_p_value_rejected() -> None:
    with pytest.raises(ValueError):
        holm_correction((1.5,), alpha=0.05)
