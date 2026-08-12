"""T12: repeated identical solves must produce bitwise-identical results."""

from __future__ import annotations

from collections.abc import Callable


class DeterminismError(Exception):
    pass


def assert_deterministic[T](compute: Callable[[], T], repetitions: int = 100) -> T:
    if repetitions < 1:
        raise ValueError(f"repetitions must be at least 1, got {repetitions}")
    reference = compute()
    for attempt in range(1, repetitions):
        result = compute()
        if result != reference:
            raise DeterminismError(
                f"non-deterministic result on repetition {attempt + 1}/{repetitions}: "
                f"{result!r} != {reference!r}"
            )
    return reference
