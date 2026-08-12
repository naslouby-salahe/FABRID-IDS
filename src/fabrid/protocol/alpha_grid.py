from __future__ import annotations

import math

from fabrid.domain.values import TargetFalsePositiveRate
from fabrid.protocol.models import AlphaGrid

_LOG_MINIMUM = 1e-4
_LOG_MAXIMUM = 0.05
_LOG_POINT_COUNT = 201
_EXTRA_POINTS = (0.001, 0.0025, 0.005, 0.01, 0.02)
_DEDUPLICATION_TOLERANCE = 1e-12
_EXPECTED_POINT_COUNT = 207


def build_alpha_grid() -> AlphaGrid:
    log_minimum = math.log10(_LOG_MINIMUM)
    log_maximum = math.log10(_LOG_MAXIMUM)
    log_values = tuple(
        10.0
        ** (
            log_minimum
            + index * (log_maximum - log_minimum) / (_LOG_POINT_COUNT - 1)
        )
        for index in range(_LOG_POINT_COUNT)
    )
    ordered = sorted((0.0, *log_values, *_EXTRA_POINTS))

    deduplicated: list[float] = []
    for value in ordered:
        if (
            not deduplicated
            or abs(value - deduplicated[-1]) > _DEDUPLICATION_TOLERANCE
        ):
            deduplicated.append(value)

    if len(deduplicated) != _EXPECTED_POINT_COUNT:
        raise RuntimeError(
            f"alpha grid must contain {_EXPECTED_POINT_COUNT} values, "
            f"got {len(deduplicated)}"
        )

    return AlphaGrid(
        values=tuple(TargetFalsePositiveRate(value) for value in deduplicated)
    )
