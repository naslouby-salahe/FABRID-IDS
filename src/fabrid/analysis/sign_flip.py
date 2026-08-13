from __future__ import annotations

import itertools
from dataclasses import dataclass

from fabrid.domain.values import MetricDifference, Probability, RowCount

_ZERO_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class SignFlipResult:
    observed_mean_difference: MetricDifference
    p_value: Probability
    enumerated_assignments: RowCount


def exact_sign_flip_test(
    paired_differences: tuple[MetricDifference, ...],
) -> SignFlipResult:
    if not paired_differences:
        raise ValueError("exact sign-flip test requires paired differences")

    observed_mean = sum(
        difference.value for difference in paired_differences
    ) / len(paired_differences)
    observed_absolute = abs(observed_mean)
    extreme_count = 0
    assignment_count = 0
    for signs in itertools.product((1.0, -1.0), repeat=len(paired_differences)):
        flipped_mean = sum(
            sign * difference.value
            for sign, difference in zip(signs, paired_differences, strict=True)
        ) / len(paired_differences)
        if abs(flipped_mean) >= observed_absolute - _ZERO_TOLERANCE:
            extreme_count += 1
        assignment_count += 1

    return SignFlipResult(
        observed_mean_difference=MetricDifference(observed_mean),
        p_value=Probability(extreme_count / assignment_count),
        enumerated_assignments=RowCount(assignment_count),
    )
