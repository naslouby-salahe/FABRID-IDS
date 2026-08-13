from __future__ import annotations

from fabrid.allocation.contracts import (
    Allocation,
    AllocationDecision,
    FederationWeights,
)
from fabrid.domain.enums import AllocationPolicy, BudgetFeasibility
from fabrid.domain.values import FalsePositiveBudget, TargetFalsePositiveRate

_BUDGET_TOLERANCE = 1.0e-12


def _maximum_target_rate(
    weights: FederationWeights,
    budget: FalsePositiveBudget,
    local_cap: TargetFalsePositiveRate,
) -> TargetFalsePositiveRate:
    smallest_weight = min(client.weight.value for client in weights.clients)
    return TargetFalsePositiveRate(
        min(local_cap.value, budget.value / smallest_weight)
    )


def allocate_equal_alert(
    weights: FederationWeights,
    budget: FalsePositiveBudget,
    local_cap: TargetFalsePositiveRate,
) -> Allocation:
    maximum = _maximum_target_rate(weights, budget, local_cap)
    low = 0.0
    high = maximum.value

    for _ in range(80):
        midpoint = (low + high) / 2.0
        weighted_cost = sum(
            client.weight.value
            * min(midpoint / client.weight.value, local_cap.value)
            for client in weights.clients
        )
        if weighted_cost <= budget.value + _BUDGET_TOLERANCE:
            low = midpoint
        else:
            high = midpoint

    decisions = tuple(
        AllocationDecision(
            client_id=client.client_id,
            target_rate=TargetFalsePositiveRate(
                min(low / client.weight.value, local_cap.value)
            ),
        )
        for client in weights.clients
    )
    allocation = Allocation(policy=AllocationPolicy.EQ_ALERT, decisions=decisions)
    if (
        allocation.budget_feasibility(weights.allocation_weights, budget)
        is BudgetFeasibility.INFEASIBLE
    ):
        raise ValueError("equal-alert allocation violates federation budget")
    return allocation
