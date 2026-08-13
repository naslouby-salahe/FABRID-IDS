from __future__ import annotations

from collections.abc import Callable

import pytest

from fabrid.allocation.baselines.equal_alert import allocate_equal_alert
from fabrid.allocation.contracts import (
    AllocationWeights,
    ClientBudgetWeight,
    ClientUtilityCurve,
    ClientUtilityCurves,
    ClientUtilityPoint,
    FederationWeights,
)
from fabrid.allocation.policies.equal_fpr import allocate_equal_fpr
from fabrid.allocation.policies.fabrid_macro import allocate_fabrid_macro
from fabrid.allocation.policies.fabrid_minimax import allocate_fabrid_minimax
from fabrid.allocation.policies.greedy import allocate_greedy
from fabrid.allocation.solver import OptimizedAllocation
from fabrid.domain.enums import AllocationPolicy, BudgetFeasibility, SolverStatus
from fabrid.domain.identifiers import ClientId
from fabrid.domain.population import ClientPopulation
from fabrid.domain.values import (
    ClientWeight,
    DetectionUtility,
    FalsePositiveBudget,
    TargetFalsePositiveRate,
)
from fabrid.protocol.models import SolverSettings
from fabrid.protocol.specification import PROTOCOL

FabridAllocator = Callable[
    [ClientUtilityCurves, AllocationWeights, FalsePositiveBudget, SolverSettings],
    OptimizedAllocation,
]


def _curve(client_id: ClientId, utilities: tuple[float, ...]) -> ClientUtilityCurve:
    rates = (0.0, 0.01, 0.02)
    return ClientUtilityCurve(
        client_id=client_id,
        points=tuple(
            ClientUtilityPoint(
                target_rate=TargetFalsePositiveRate(rate),
                utility=DetectionUtility(utility),
            )
            for rate, utility in zip(rates, utilities, strict=True)
        ),
    )


def _weights() -> FederationWeights:
    return FederationWeights(
        AllocationWeights(
            (
                ClientBudgetWeight(ClientId("a"), ClientWeight(0.7)),
                ClientBudgetWeight(ClientId("b"), ClientWeight(0.3)),
            )
        )
    )


def test_equal_fpr_assigns_same_rate_to_population() -> None:
    population = ClientPopulation((ClientId("a"), ClientId("b")))
    allocation = allocate_equal_fpr(
        population,
        FalsePositiveBudget(0.01),
        TargetFalsePositiveRate(0.05),
    )

    assert allocation.policy is AllocationPolicy.EQ_FPR
    assert allocation.decision(ClientId("a")).target_rate == TargetFalsePositiveRate(0.01)
    assert allocation.decision(ClientId("b")).target_rate == TargetFalsePositiveRate(0.01)


def test_equal_alert_allocates_more_rate_to_lower_weight_client() -> None:
    weights = _weights()
    budget = FalsePositiveBudget(0.01)
    allocation = allocate_equal_alert(
        weights,
        budget,
        TargetFalsePositiveRate(0.05),
    )

    assert allocation.policy is AllocationPolicy.EQ_ALERT
    assert (
        allocation.decision(ClientId("a")).target_rate.value
        < allocation.decision(ClientId("b")).target_rate.value
    )
    assert (
        allocation.budget_feasibility(weights.allocation_weights, budget)
        is BudgetFeasibility.FEASIBLE
    )


def test_greedy_prefers_higher_incremental_utility() -> None:
    curves = ClientUtilityCurves(
        (
            _curve(ClientId("a"), (0.0, 0.8, 0.9)),
            _curve(ClientId("b"), (0.0, 0.1, 0.2)),
        )
    )
    weights = AllocationWeights(
        (
            ClientBudgetWeight(ClientId("a"), ClientWeight(0.5)),
            ClientBudgetWeight(ClientId("b"), ClientWeight(0.5)),
        )
    )
    allocation = allocate_greedy(
        curves,
        weights,
        FalsePositiveBudget(0.005),
        TargetFalsePositiveRate(0.05),
    )

    assert allocation.policy is AllocationPolicy.GREEDY
    assert allocation.decision(ClientId("a")).target_rate == TargetFalsePositiveRate(0.01)
    assert allocation.decision(ClientId("b")).target_rate == TargetFalsePositiveRate(0.0)


@pytest.mark.parametrize(
    ("allocator", "policy"),
    (
        (allocate_fabrid_macro, AllocationPolicy.FABRID_MACRO),
        (allocate_fabrid_minimax, AllocationPolicy.FABRID_MINIMAX),
    ),
)
def test_fabrid_optimizers_return_budget_feasible_optimal_allocations(
    allocator: FabridAllocator,
    policy: AllocationPolicy,
) -> None:
    curves = ClientUtilityCurves(
        (
            _curve(ClientId("a"), (0.0, 0.8, 1.0)),
            _curve(ClientId("b"), (0.0, 0.6, 0.9)),
        )
    )
    weights = AllocationWeights(
        (
            ClientBudgetWeight(ClientId("a"), ClientWeight(0.5)),
            ClientBudgetWeight(ClientId("b"), ClientWeight(0.5)),
        )
    )
    budget = FalsePositiveBudget(0.01)

    optimized = allocator(curves, weights, budget, PROTOCOL.solver)

    assert optimized.allocation.policy is policy
    assert optimized.solver.status is SolverStatus.OPTIMAL
    assert (
        optimized.allocation.budget_feasibility(weights, budget)
        is BudgetFeasibility.FEASIBLE
    )
