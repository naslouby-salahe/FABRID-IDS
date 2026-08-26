from __future__ import annotations

from itertools import product

import pytest

from fabrid.allocation.optimization import OptimizedAllocation, SolverStage
from fabrid.allocation.policies import allocate_equal_fpr, allocate_fabrid_macro
from fabrid.allocation.problem import (
    AllocationProblem,
    FederationFrontierInputs,
    FrontierScoreArtifacts,
    build_allocation_problem,
    build_client_frontier_inputs,
    equal_client_weights,
)
from fabrid.config import LOCAL_TARGET_RATE_CAP, AllocationPolicy, SolverObjective
from tests.support import production_protocol

from .synthetic_federation import (
    ALPHA_GRID,
    eligibility_config,
    synthetic_client_scores,
    synthetic_population,
    synthetic_problem,
)


def _solve(problem: AllocationProblem) -> OptimizedAllocation:
    return allocate_fabrid_macro(problem, production_protocol().solver)


def _macro_utility_objective(optimized: OptimizedAllocation) -> SolverObjective:
    stage = next(
        stage
        for stage in optimized.solver.stages
        if stage.stage is SolverStage.MACRO_PRIMARY_UTILITY
    )
    return -float(stage.objective)


def test_t09_budget_feasibility() -> None:
    problem = synthetic_problem(("a", "b", "c"), 0.01)
    optimized = _solve(problem)
    cost = optimized.allocation.total_weighted_cost(problem.weights.allocation_weights)
    assert cost <= problem.remaining_budget + 1e-12


def test_t10_one_target_rate_per_client() -> None:
    problem = synthetic_problem(("a", "b", "c"), 0.01)
    curves = problem.frontier.eligible_curves()
    assert curves is not None
    optimized = _solve(problem)
    decisions = optimized.allocation.decisions
    client_ids = [decision.client_id for decision in decisions]
    assert len(client_ids) == len(set(client_ids)), "exactly one target rate per client"
    candidates = {
        curve.client_id: {point.target_rate for point in curve.points} for curve in curves.clients
    }
    for decision in decisions:
        assert decision.target_rate in candidates[decision.client_id], (
            "target rate must be one of the candidate grid rates"
        )


def test_t11_brute_force_parity_of_fabrid_macro() -> None:
    problem = synthetic_problem(("a", "b"), 0.005)
    curves = problem.frontier.eligible_curves()
    assert curves is not None
    weights = problem.weights.allocation_weights
    candidates = {
        curve.client_id: tuple((point.target_rate, float(point.utility)) for point in curve.points)
        for curve in curves.clients
    }
    client_ids = tuple(candidates)
    best: tuple[float, tuple[float, ...]] | None = None
    for combo in product(*(candidates[client_id] for client_id in client_ids)):
        total_cost = sum(
            weights.for_client(client_id) * rate
            for (rate, _), client_id in zip(combo, client_ids, strict=True)
        )
        if total_cost > problem.remaining_budget + 1e-9:
            continue
        mean_utility = sum(
            utility for (_, utility), _ in zip(combo, client_ids, strict=True)
        ) / len(client_ids)
        if best is None or mean_utility > best[0] + 1e-12:
            best = (mean_utility, tuple(rate for rate, _ in combo))
    assert best is not None, "a feasible allocation must exist"
    optimized = _solve(problem)
    assert _macro_utility_objective(optimized) == pytest.approx(best[0], abs=1e-6), (
        "solver objective must match exhaustive search over the candidate grid"
    )
    assert optimized.allocation.total_weighted_cost(weights) <= problem.remaining_budget + 1e-6


def test_t13_zero_budget_yields_no_alerts() -> None:
    allocation = allocate_equal_fpr(synthetic_problem(("a", "b"), 0.0))
    for decision in allocation.decisions:
        assert decision.target_rate == 0.0
    problem = synthetic_problem(("a", "b"), 0.0)
    optimized = _solve(problem)
    assert optimized.allocation.policy is AllocationPolicy.FABRID_MACRO
    assert optimized.allocation.total_weighted_cost(problem.weights.allocation_weights) == 0.0
    for decision in optimized.allocation.decisions:
        assert decision.target_rate == 0.0


def test_t14_single_client_behavior() -> None:
    allocation = allocate_equal_fpr(synthetic_problem(("a",), 0.005))
    assert len(allocation.decisions) == 1
    assert allocation.decisions[0].target_rate == pytest.approx(0.005)
    problem = synthetic_problem(("a",), 0.005)
    curves = problem.frontier.eligible_curves()
    assert curves is not None
    curve = curves.clients[0]
    feasible = [
        point
        for point in curve.points
        if float(point.target_rate) <= problem.remaining_budget + 1e-9
    ]
    best = max(feasible, key=lambda point: float(point.utility))
    optimized = _solve(problem)
    decision = optimized.allocation.decisions[0]
    assert decision.target_rate == pytest.approx(float(best.target_rate))
    assert _macro_utility_objective(optimized) == pytest.approx(float(best.utility))


def test_t15_equal_utility_curves_share_rates() -> None:
    population = synthetic_population(("a", "b"))
    frontier_a, validation_a = synthetic_client_scores("a", seed=7)
    frontier_b, validation_b = synthetic_client_scores("b", seed=7)
    inputs = FederationFrontierInputs(
        clients=(
            build_client_frontier_inputs(
                FrontierScoreArtifacts(benign_frontier=frontier_a, attack_validation=validation_a),
                ALPHA_GRID,
                frontier_row_count=frontier_a.row_count,
            ),
            build_client_frontier_inputs(
                FrontierScoreArtifacts(benign_frontier=frontier_b, attack_validation=validation_b),
                ALPHA_GRID,
                frontier_row_count=frontier_b.row_count,
            ),
        )
    )
    weights = equal_client_weights(population)
    problem = build_allocation_problem(
        inputs, weights, 0.005, eligibility_config(), maximum_target_rate=LOCAL_TARGET_RATE_CAP
    )
    curves = problem.frontier.eligible_curves()
    assert curves is not None
    first = curves.clients[0].points
    for curve in curves.clients[1:]:
        assert tuple(point.target_rate for point in curve.points) == tuple(
            point.target_rate for point in first
        )
        assert tuple(point.utility for point in curve.points) == tuple(
            point.utility for point in first
        )
    optimized = _solve(problem)
    rates = tuple(decision.target_rate for decision in optimized.allocation.decisions)
    assert len(set(rates)) == 1, "identical curves and weights must yield identical rates"


def test_t16_monotonic_budget_feasibility() -> None:
    low = synthetic_problem(("a", "b", "c"), 0.005)
    high = synthetic_problem(("a", "b", "c"), 0.02)
    solved_low = _solve(low)
    solved_high = _solve(high)
    assert (
        solved_low.allocation.total_weighted_cost(low.weights.allocation_weights)
        <= high.remaining_budget + 1e-12
    )
    assert _macro_utility_objective(solved_high) >= _macro_utility_objective(solved_low) - 1e-9
