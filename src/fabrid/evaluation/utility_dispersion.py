from __future__ import annotations

import math

from fabrid.allocation.contracts import ClientUtilityCurves
from fabrid.domain.values import UtilityDispersion


def _population_standard_deviation(values: tuple[float, ...]) -> UtilityDispersion:
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return UtilityDispersion(math.sqrt(variance))


def utility_dispersion_per_candidate(
    utility_curves: ClientUtilityCurves,
) -> tuple[UtilityDispersion, ...]:
    candidate_count = len(utility_curves.clients[0].points)
    return tuple(
        _population_standard_deviation(
            tuple(curve.points[index].utility.value for curve in utility_curves.clients)
        )
        for index in range(candidate_count)
    )


def aggregate_utility_dispersion(
    utility_curves: ClientUtilityCurves,
) -> UtilityDispersion:
    dispersion = utility_dispersion_per_candidate(utility_curves)
    return UtilityDispersion(
        sum(item.value for item in dispersion) / len(dispersion)
    )
