from __future__ import annotations

import pytest

from fabrid.evaluation.heterogeneity import (
    aggregate_heterogeneity,
    utility_dispersion_per_candidate,
)
from fabrid.evaluation.record_level import ClientId
from fabrid.schemas.allocation import ClientUtilityCurve

_GRID = (0.0, 0.01, 0.02)


def test_identical_curves_have_zero_dispersion() -> None:
    curves = {
        ClientId("1"): ClientUtilityCurve(ClientId("1"), _GRID, (0.0, 0.5, 1.0)),
        ClientId("2"): ClientUtilityCurve(ClientId("2"), _GRID, (0.0, 0.5, 1.0)),
    }
    dispersion = utility_dispersion_per_candidate(curves)
    assert dispersion == pytest.approx((0.0, 0.0, 0.0))
    assert aggregate_heterogeneity(curves) == pytest.approx(0.0)


def test_divergent_curves_have_positive_dispersion() -> None:
    curves = {
        ClientId("1"): ClientUtilityCurve(ClientId("1"), _GRID, (0.0, 0.9, 1.0)),
        ClientId("2"): ClientUtilityCurve(ClientId("2"), _GRID, (0.0, 0.1, 0.2)),
    }
    dispersion = utility_dispersion_per_candidate(curves)
    assert dispersion[1] > 0
    assert aggregate_heterogeneity(curves) > 0


def test_mismatched_grids_rejected() -> None:
    curves = {
        ClientId("1"): ClientUtilityCurve(ClientId("1"), (0.0, 0.01), (0.0, 0.5)),
        ClientId("2"): ClientUtilityCurve(ClientId("2"), (0.0, 0.02), (0.0, 0.5)),
    }
    with pytest.raises(ValueError):
        utility_dispersion_per_candidate(curves)


def test_empty_curves_rejected() -> None:
    with pytest.raises(ValueError):
        utility_dispersion_per_candidate({})
