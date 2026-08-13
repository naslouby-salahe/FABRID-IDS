from __future__ import annotations

from fabrid.allocation.contracts import Allocation, AllocationDecision
from fabrid.domain.enums import AllocationPolicy
from fabrid.domain.population import ClientPopulation
from fabrid.domain.values import FalsePositiveBudget, TargetFalsePositiveRate


def allocate_equal_fpr(
    population: ClientPopulation,
    budget: FalsePositiveBudget,
    maximum_target_rate: TargetFalsePositiveRate,
) -> Allocation:
    if budget.value > maximum_target_rate.value:
        raise ValueError(
            f"budget {budget.value} exceeds local cap {maximum_target_rate.value}"
        )

    target_rate = TargetFalsePositiveRate(budget.value)
    return Allocation(
        policy=AllocationPolicy.EQ_FPR,
        decisions=tuple(
            AllocationDecision(client_id=client_id, target_rate=target_rate)
            for client_id in population.clients
        ),
    )
