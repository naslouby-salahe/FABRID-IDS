"""TEST_ORACLE: non-deployable upper bound using test-attack utility curves.

Never enters primary hypothesis tests, hyperparameter selection, budget
selection, or success/failure determination. Isolated from the rest of the
allocation package: callers must construct an explicit `OracleAccessToken`
to prove intent before this module will run, so default execution paths
cannot reach it by accident.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from fabrid.allocation.fabrid_macro import allocate_fabrid_macro
from fabrid.config.protocol import SolverSettings
from fabrid.evaluation.record_level import ClientId
from fabrid.schemas.allocation import Allocation, AllocationPolicy, ClientUtilityCurve


@dataclass(frozen=True, slots=True)
class OracleAccessToken:
    """Deliberate friction: callers must construct this explicitly to prove intent."""

    acknowledged_non_deployable: bool

    def __post_init__(self) -> None:
        if not self.acknowledged_non_deployable:
            raise ValueError(
                "OracleAccessToken requires acknowledged_non_deployable=True; TEST_ORACLE is a "
                "non-deployable upper bound and must never be used as a competing practical policy"
            )


def allocate_test_oracle(
    token: OracleAccessToken,
    test_utility_curves: Mapping[ClientId, ClientUtilityCurve],
    weight: Mapping[ClientId, float],
    remaining_budget: float,
    settings: SolverSettings,
) -> Allocation:
    del token  # existence, not content, is what gates the call
    allocation = allocate_fabrid_macro(test_utility_curves, weight, remaining_budget, settings)
    return Allocation(policy=AllocationPolicy.TEST_ORACLE, decisions=allocation.decisions)
