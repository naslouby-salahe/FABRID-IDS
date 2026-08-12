"""Client utility-curve heterogeneity diagnostic: H_u(alpha_j) and aggregate H_U.

Measures whether clients benefit differently from additional false-alert
capacity — the mechanism FABRID is hypothesized to exploit — independent of
any weight-based traffic-volume framing.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from fabrid.evaluation.record_level import ClientId
from fabrid.schemas.allocation import ClientUtilityCurve


def _population_std_dev(values: tuple[float, ...]) -> float:
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    return math.sqrt(variance)


def utility_dispersion_per_candidate(
    utility_curves: Mapping[ClientId, ClientUtilityCurve],
) -> tuple[float, ...]:
    """H_u(alpha_j) = SD_k(u_{k,j}) for every candidate index j."""
    if not utility_curves:
        raise ValueError("utility_dispersion_per_candidate requires at least one client")

    curves = list(utility_curves.values())
    shared_grid = curves[0].alpha_grid
    for curve in curves:
        if curve.alpha_grid != shared_grid:
            raise ValueError("all clients must share the same candidate target-rate grid")

    return tuple(
        _population_std_dev(tuple(curve.utility[j] for curve in curves))
        for j in range(len(shared_grid))
    )


def aggregate_heterogeneity(utility_curves: Mapping[ClientId, ClientUtilityCurve]) -> float:
    """H_U = mean over candidates of H_u(alpha_j)."""
    dispersion = utility_dispersion_per_candidate(utility_curves)
    return sum(dispersion) / len(dispersion)
