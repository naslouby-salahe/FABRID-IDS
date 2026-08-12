"""Exact two-sided sign-flip test on paired seed differences.

The experimental unit is the detector seed, not an individual network row.
For the primary ten-seed design this enumerates all `2^10 = 1024` sign
assignments exactly rather than approximating via resampling.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

_ZERO_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class SignFlipResult:
    observed_mean_difference: float
    p_value: float
    enumerated_sign_assignments: int


def exact_sign_flip_test(paired_differences: tuple[float, ...]) -> SignFlipResult:
    n = len(paired_differences)
    if n == 0:
        raise ValueError("exact_sign_flip_test requires at least one paired difference")

    observed_mean = sum(paired_differences) / n
    observed_abs = abs(observed_mean)

    extreme_count = 0
    total = 0
    for signs in itertools.product((1.0, -1.0), repeat=n):
        flipped_mean = sum(s * d for s, d in zip(signs, paired_differences, strict=True)) / n
        if abs(flipped_mean) >= observed_abs - _ZERO_TOLERANCE:
            extreme_count += 1
        total += 1

    return SignFlipResult(
        observed_mean_difference=observed_mean,
        p_value=extreme_count / total,
        enumerated_sign_assignments=total,
    )
