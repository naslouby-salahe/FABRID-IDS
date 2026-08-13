from __future__ import annotations

from collections.abc import Callable

from fabrid.domain.values import RowCount


class DeterminismValidationError(Exception):
    pass


def validate_determinism[T](
    compute: Callable[[], T],
    repetitions: RowCount,
) -> T:
    if repetitions.value < 1:
        raise ValueError("determinism validation requires at least one repetition")
    reference = compute()
    for _ in range(1, repetitions.value):
        result = compute()
        if result != reference:
            raise DeterminismValidationError("repeated computation produced different results")
    return reference
