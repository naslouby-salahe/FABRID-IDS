from __future__ import annotations

from fabrid.allocation.contracts import Allocation, AllocationWeights
from fabrid.domain.population import ClientPopulation
from fabrid.domain.values import FalsePositiveBudget


class BudgetValidationError(Exception):
    pass


def validate_budget_feasibility(
    allocation: Allocation,
    weights: AllocationWeights,
    budget: FalsePositiveBudget,
) -> None:
    cost = allocation.total_weighted_cost(weights)
    if cost.value > budget.value:
        raise BudgetValidationError(
            f"weighted allocation cost {cost.value} exceeds budget {budget.value}"
        )


def validate_population_coverage(
    allocation: Allocation,
    population: ClientPopulation,
) -> None:
    decided = tuple(decision.client_id for decision in allocation.decisions)
    if set(decided) != set(population.clients):
        raise BudgetValidationError("allocation does not cover exactly the expected population")
