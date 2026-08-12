from __future__ import annotations

import itertools

import pytest

from fabrid.audit.determinism import DeterminismError, assert_deterministic


def test_deterministic_function_passes() -> None:
    result = assert_deterministic(lambda: 1 + 1, repetitions=50)
    assert result == 2


def test_nondeterministic_function_raises() -> None:
    counter = itertools.count()
    with pytest.raises(DeterminismError):
        assert_deterministic(lambda: next(counter), repetitions=5)


def test_invalid_repetitions_rejected() -> None:
    with pytest.raises(ValueError):
        assert_deterministic(lambda: 1, repetitions=0)
