from __future__ import annotations

from fabrid.allocation.contracts import (
    Allocation,
    AllocationDecision,
    FederationWeights,
)
from fabrid.domain.enums import AllocationPolicy
from fabrid.domain.values import FalsePositiveBudget, TargetFalsePositiveRate

_WEIGHT_EQUALITY_TOLERANCE = 1e-12
_BISECTION_ITERATIONS = 100


def _max_constant_budget_share(
    weights: FederationWeights,
    budget: FalsePositiveBudget,
    maximum_target_rate: TargetFalsePositiveRate,
) -> float:
    def total_cost(constant_share: float) -> float:
        return sum(
            min(
                constant_share,
                maximum_target_rate.value * client.weight.value,
            )
            for client in weights.clients
        )

    upper = maximum_target_rate.value * max(
        client.weight.value for client in weights.clients
    )
    if total_cost(upper) <= budget.value:
        return upper

    low = 0.0
    high = upper
    for _ in range(_BISECTION_ITERATIONS):
        midpoint = (low + high) / 2.0
        if total_cost(midpoint) <= budget.value:
            low = midpoint
        else:
            high = midpoint
    return low


def allocate_equal_alert(
    weights: FederationWeights,
    budget: FalsePositiveBudget,
    maximum_target_rate: TargetFalsePositiveRate,
) -> Allocation:
    values = tuple(client.weight.value for client in weights.clients)
    if max(values) - min(values) < _WEIGHT_EQUALITY_TOLERANCE:
        raise ValueError(
            "EQ_ALERT is identical to EQ_FPR under equal-client weighting"
        )

    constant_share = _max_constant_budget_share(
        weights,
        budget,
        maximum_target_rate,
    )
    return Allocation(
        policy=AllocationPolicy.EQ_ALERT,
        decisions=tuple(
            AllocationDecision(
                client_id=client.client_id,
                target_rate=TargetFalsePositiveRate(
                    min(
                        constant_share / client.weight.value,
                        maximum_target_rate.value,
                    )
                ),
            )
            for client in weights.clients
        ),
    )
