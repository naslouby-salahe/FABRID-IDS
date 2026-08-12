from __future__ import annotations

import pytest

from fabrid.allocation.test_oracle import OracleAccessToken, allocate_test_oracle
from fabrid.config.protocol import SolverSettings
from fabrid.evaluation.record_level import ClientId
from fabrid.schemas.allocation import AllocationPolicy, ClientUtilityCurve

_SETTINGS = SolverSettings(mip_rel_gap=0.0, time_limit_seconds=10.0, accept_mip_gap_leq=1e-9)
_GRID = (0.0, 0.01, 0.02, 0.03)


def test_token_requires_explicit_acknowledgment() -> None:
    with pytest.raises(ValueError):
        OracleAccessToken(acknowledged_non_deployable=False)


def test_oracle_allocates_and_is_labeled_test_oracle() -> None:
    token = OracleAccessToken(acknowledged_non_deployable=True)
    curves = {
        ClientId("1"): ClientUtilityCurve(ClientId("1"), _GRID, (0.0, 0.3, 0.55, 0.6)),
    }
    weight = {ClientId("1"): 1.0}
    allocation = allocate_test_oracle(
        token, curves, weight, remaining_budget=0.02, settings=_SETTINGS
    )
    assert allocation.policy is AllocationPolicy.TEST_ORACLE
    assert allocation.alpha_of(ClientId("1")) == pytest.approx(0.02)
