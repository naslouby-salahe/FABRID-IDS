"""T09/T10: budget feasibility and exactly-one-target-per-client invariants."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from fabrid.evaluation.record_level import ClientId
from fabrid.schemas.allocation import Allocation


class BudgetInvariantError(Exception):
    pass


def assert_budget_feasible(
    allocation: Allocation, weight: Mapping[ClientId, float], budget: float
) -> None:
    if not allocation.is_budget_feasible(weight, budget):
        raise BudgetInvariantError(
            f"weighted allocation cost {allocation.total_weighted_cost(weight)} exceeds "
            f"budget {budget}"
        )


def assert_one_target_per_client(
    allocation: Allocation, expected_client_ids: Sequence[ClientId]
) -> None:
    decided = set(allocation.decisions.keys())
    expected = set(expected_client_ids)
    if decided != expected:
        raise BudgetInvariantError(
            f"allocation covers {sorted(decided)} but expected exactly {sorted(expected)}"
        )
