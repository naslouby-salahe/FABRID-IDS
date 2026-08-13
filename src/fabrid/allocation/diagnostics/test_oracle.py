from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from fabrid.allocation.contracts import Allocation, AllocationWeights, ClientUtilityCurves
from fabrid.allocation.policies.fabrid_macro import allocate_fabrid_macro
from fabrid.allocation.solver import OptimizedAllocation
from fabrid.domain.enums import AllocationPolicy
from fabrid.domain.values import FalsePositiveBudget
from fabrid.protocol.models import SolverSettings


class OracleAuthorization(StrEnum):
    ACKNOWLEDGED_NON_DEPLOYABLE = "acknowledged_non_deployable"


@dataclass(frozen=True, slots=True)
class OracleAccessToken:
    authorization: OracleAuthorization


def allocate_test_oracle(
    token: OracleAccessToken,
    test_utility_curves: ClientUtilityCurves,
    weights: AllocationWeights,
    remaining_budget: FalsePositiveBudget,
    settings: SolverSettings,
) -> OptimizedAllocation:
    if token.authorization is not OracleAuthorization.ACKNOWLEDGED_NON_DEPLOYABLE:
        raise ValueError("TEST_ORACLE requires explicit non-deployable authorization")
    optimized = allocate_fabrid_macro(
        test_utility_curves,
        weights,
        remaining_budget,
        settings,
    )
    return OptimizedAllocation(
        allocation=Allocation(
            policy=AllocationPolicy.TEST_ORACLE,
            decisions=optimized.allocation.decisions,
        ),
        solver=optimized.solver,
    )
