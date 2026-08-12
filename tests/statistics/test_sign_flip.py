from __future__ import annotations

import pytest

from fabrid.statistics.sign_flip import exact_sign_flip_test


def test_ten_seeds_enumerates_1024_assignments() -> None:
    differences = (1.0, 2.0, 3.0, -1.0, 0.5, 1.5, 2.5, -0.5, 1.0, 0.8)
    result = exact_sign_flip_test(differences)
    assert result.enumerated_sign_assignments == 1024


def test_all_positive_differences_yield_minimal_p_value() -> None:
    differences = tuple(float(i + 1) for i in range(10))
    result = exact_sign_flip_test(differences)
    # two-sided: only the all-positive and all-negative sign assignments reach
    # the observed extreme |mean| (5.5), since it is the unique maximum.
    assert result.p_value == pytest.approx(2 / 1024, abs=1e-9)


def test_all_zero_differences_yield_p_value_one() -> None:
    result = exact_sign_flip_test((0.0,) * 5)
    assert result.p_value == pytest.approx(1.0)


def test_symmetric_differences_yield_large_p_value() -> None:
    # +d and -d cancel; observed mean is 0, so every sign assignment is at
    # least as extreme -> p-value 1.
    result = exact_sign_flip_test((1.0, -1.0, 2.0, -2.0))
    assert result.p_value == pytest.approx(1.0)


def test_empty_input_rejected() -> None:
    with pytest.raises(ValueError):
        exact_sign_flip_test(())
