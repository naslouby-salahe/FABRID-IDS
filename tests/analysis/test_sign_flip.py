from __future__ import annotations

import pytest

from fabrid.analysis.sign_flip import exact_sign_flip_test
from fabrid.domain.values import MetricDifference


def _differences(*values: float) -> tuple[MetricDifference, ...]:
    return tuple(MetricDifference(value) for value in values)


def test_ten_seed_sign_flip_enumerates_all_assignments() -> None:
    result = exact_sign_flip_test(
        _differences(1.0, 2.0, 3.0, -1.0, 0.5, 1.5, 2.5, -0.5, 1.0, 0.8)
    )

    assert result.enumerated_assignments.value == 1_024


def test_all_positive_differences_reach_minimum_two_sided_probability() -> None:
    result = exact_sign_flip_test(_differences(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0))

    assert result.p_value.value == pytest.approx(2.0 / 1_024.0, abs=1e-9)


def test_zero_differences_produce_unit_probability() -> None:
    result = exact_sign_flip_test(_differences(0.0, 0.0, 0.0, 0.0, 0.0))

    assert result.p_value.value == pytest.approx(1.0)


def test_empty_paired_differences_are_rejected() -> None:
    with pytest.raises(ValueError):
        exact_sign_flip_test(())
