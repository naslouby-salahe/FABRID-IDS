from __future__ import annotations

import pytest

from fabrid.allocation.contracts import (
    ClientUtilityCurve,
    ClientUtilityCurves,
    ClientUtilityPoint,
)
from fabrid.domain.identifiers import ClientId
from fabrid.domain.values import DetectionUtility, TargetFalsePositiveRate
from fabrid.evaluation.utility_dispersion import (
    aggregate_utility_dispersion,
    utility_dispersion_per_candidate,
)


def _curve(client_id: str, utilities: tuple[float, ...]) -> ClientUtilityCurve:
    rates = (0.0, 0.01, 0.02)
    return ClientUtilityCurve(
        client_id=ClientId(client_id),
        points=tuple(
            ClientUtilityPoint(
                target_rate=TargetFalsePositiveRate(rate),
                utility=DetectionUtility(utility),
            )
            for rate, utility in zip(rates, utilities, strict=True)
        ),
    )


def test_identical_curves_have_zero_dispersion() -> None:
    curves = ClientUtilityCurves(
        (_curve("1", (0.0, 0.5, 1.0)), _curve("2", (0.0, 0.5, 1.0)))
    )
    dispersion = utility_dispersion_per_candidate(curves)
    assert tuple(item.value for item in dispersion) == pytest.approx((0.0, 0.0, 0.0))
    assert aggregate_utility_dispersion(curves).value == pytest.approx(0.0)
