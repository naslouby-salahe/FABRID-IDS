from __future__ import annotations

import numpy as np
import pytest

from fabrid.allocation import policies as allocation_policies
from fabrid.allocation.optimization import OptimizedAllocation
from fabrid.allocation.policies import (
    allocate_common_rate_macro,
    allocate_equal_alert,
    allocate_equal_fpr,
    allocate_fabrid_cvar,
    allocate_fabrid_macro,
    allocate_fabrid_macro_without_rate_minimization,
    allocate_greedy,
)
from fabrid.allocation.problem import (
    AllocationProblem,
    ClientFrontierInputs,
    ClientUtilityCurve,
    ClientUtilityCurves,
    ClientUtilityPoint,
    FederationFrontierInputs,
    FrontierScoreArtifacts,
    build_allocation_problem,
    build_client_frontier_inputs,
    equal_client_weights,
)
from fabrid.config import LOCAL_TARGET_RATE_CAP, AllocationPolicy, SolverConfig
from tests.support import production_protocol

from .synthetic_federation import (
    ALPHA_GRID,
    eligibility_config,
    synthetic_client_scores,
    synthetic_population,
    synthetic_problem,
)


def _solver_config() -> SolverConfig:
    return SolverConfig(
        requested_gap=0.05,
        time_limit_seconds=10.0,
        accepted_gap=0.1,
        accepted_absolute_gap=0.001,
        equal_alert_bisection_iterations=80,
        cvar_tail_fraction=0.25,
        calibration_resolution_factor=2.0,
    )


def test_allocate_equal_fpr_sets_budget_everywhere() -> None:
    problem = synthetic_problem(("a", "b", "c"), 0.005)
    allocation = allocate_equal_fpr(problem)
    assert allocation.policy is AllocationPolicy.EQ_FPR
    for decision in allocation.decisions:
        assert decision.target_rate == pytest.approx(0.005)
    oversized = synthetic_problem(("a", "b", "c"), 0.1)
    with pytest.raises(ValueError):
        allocate_equal_fpr(oversized)


def test_allocate_greedy_is_monotone_in_budget() -> None:
    allocation_low = allocate_greedy(synthetic_problem(("a", "b"), 0.005))
    allocation_high = allocate_greedy(synthetic_problem(("a", "b"), 0.02))
    for decision_low, decision_high in zip(
        allocation_low.decisions, allocation_high.decisions, strict=True
    ):
        assert decision_high.target_rate >= decision_low.target_rate


def _greedy_problem(utilities: dict[str, tuple[float, float]], budget: float) -> AllocationProblem:
    problem = synthetic_problem(tuple(sorted(utilities)), budget)
    curves = ClientUtilityCurves(
        tuple(
            ClientUtilityCurve(
                client_id=client_id,
                points=tuple(
                    ClientUtilityPoint(
                        target_rate=rate,
                        utility=points[0] if rate == 0.0 else points[1],
                        utility_variance=0.0,
                    )
                    for rate in ALPHA_GRID
                ),
            )
            for client_id, points in sorted(utilities.items())
        )
    )
    return problem.with_eligible_curves(curves)


@pytest.mark.parametrize("next_utility", [0.0, 0.4])
def test_greedy_stops_without_a_positive_marginal_gain(next_utility: float) -> None:
    problem = _greedy_problem({"a": (0.5, next_utility)}, budget=0.001)
    allocation = allocate_greedy(problem)
    assert allocation.decision("a").target_rate == 0.0


def test_greedy_selects_positive_gain_with_deterministic_tie_break() -> None:
    problem = _greedy_problem({"b": (0.0, 0.5), "a": (0.0, 0.5)}, budget=0.0005)
    allocation = allocate_greedy(problem)
    assert allocation.decision("a").target_rate == 0.001
    assert allocation.decision("b").target_rate == 0.0


def test_greedy_stops_when_positive_increment_is_infeasible() -> None:
    problem = _greedy_problem({"a": (0.0, 0.5)}, budget=0.0005)
    allocation = allocate_greedy(problem)
    assert allocation.decision("a").target_rate == 0.0


def test_allocate_fabrid_macro_respects_budget() -> None:
    problem = synthetic_problem(("a", "b", "c"), 0.01)
    optimized = allocate_fabrid_macro(problem, _solver_config())
    assert optimized.allocation.policy is AllocationPolicy.FABRID_MACRO
    weighted_cost = optimized.allocation.total_weighted_cost(problem.weights.allocation_weights)
    assert weighted_cost <= problem.remaining_budget + 1e-6
    for decision in optimized.allocation.decisions:
        assert decision.target_rate <= LOCAL_TARGET_RATE_CAP + 1e-9


def test_macro_without_rate_minimization_omits_cost_secondary_stage() -> None:
    problem = synthetic_problem(("a", "b", "c"), 0.01)
    optimized = allocate_fabrid_macro_without_rate_minimization(problem, _solver_config())
    assert optimized.allocation.policy is AllocationPolicy.FABRID_MACRO_NO_RATE_MINIMIZATION
    stages = tuple(stage.stage for stage in optimized.solver.stages)
    assert "macro_minimum_budget" not in stages
    assert stages == ("macro_primary_utility", "macro_lowest_variance", "macro_tie_break")


def test_common_rate_macro_has_one_feasible_rate_for_every_eligible_client() -> None:
    problem = synthetic_problem(("a", "b", "c"), 0.01)
    allocation = allocate_common_rate_macro(problem)
    assert allocation.policy is AllocationPolicy.COMMON_RATE_MACRO
    assert len({decision.target_rate for decision in allocation.decisions}) == 1
    assert allocation.total_weighted_cost(problem.weights.allocation_weights) <= (
        problem.remaining_budget + 1e-6
    )


def test_allocate_fabrid_cvar_respects_budget() -> None:
    problem = synthetic_problem(("a", "b", "c"), 0.01)
    optimized = allocate_fabrid_cvar(problem, _solver_config())
    assert optimized.allocation.policy is AllocationPolicy.FABRID_CVAR
    weighted_cost = optimized.allocation.total_weighted_cost(problem.weights.allocation_weights)
    assert weighted_cost <= problem.remaining_budget + 1e-6


def test_allocate_equal_alert_bounds_weighted_cost() -> None:
    problem = synthetic_problem(("a", "b"), 0.01)
    allocation = allocate_equal_alert(problem, production_protocol().solver)
    assert allocation.policy is AllocationPolicy.EQ_ALERT
    assert allocation.total_weighted_cost(problem.weights.allocation_weights) <= (
        problem.budget + 1e-6
    )


def test_g12_solver_determinism_identical_across_repeated_solves() -> None:
    config = production_protocol()
    problem = synthetic_problem(("a", "b", "c"), 0.01)
    solutions = [allocate_fabrid_macro(problem, config.solver) for _ in range(100)]
    first = tuple(
        (decision.client_id, decision.target_rate) for decision in solutions[0].allocation.decisions
    )
    for solution in solutions[1:]:
        current = tuple(
            (decision.client_id, decision.target_rate) for decision in solution.allocation.decisions
        )
        assert current == first


def _cvar_solver(quantile: float) -> SolverConfig:
    return SolverConfig(
        requested_gap=0.05,
        time_limit_seconds=30.0,
        accepted_gap=0.1,
        accepted_absolute_gap=0.001,
        equal_alert_bisection_iterations=80,
        cvar_tail_fraction=quantile,
        calibration_resolution_factor=2.0,
    )


def test_fabrid_cvar_uses_cvar_stage_chain() -> None:
    problem = synthetic_problem(("a", "b", "c"), 0.01)
    optimized = allocate_fabrid_cvar(problem, _cvar_solver(0.5))
    stages = tuple(stage.stage for stage in optimized.solver.stages)
    assert stages == (
        "cvar_utility",
        "cvar_mean_utility",
        "cvar_minimum_budget",
        "cvar_lowest_variance",
        "cvar_tie_break",
    )


def test_fabrid_macro_uses_lowest_variance_stage() -> None:
    problem = synthetic_problem(("a", "b", "c"), 0.01)
    optimized = allocate_fabrid_macro(problem, _solver_config())
    stages = tuple(stage.stage for stage in optimized.solver.stages)
    assert stages == (
        "macro_primary_utility",
        "macro_minimum_budget",
        "macro_lowest_variance",
        "macro_tie_break",
    )


def test_fabrid_cvar_q2_protects_worst_pair() -> None:
    problem = synthetic_problem(("a", "b", "c"), 0.01)
    hard_min = allocate_fabrid_cvar(problem, _cvar_solver(0.25))
    cvar_pair = allocate_fabrid_cvar(problem, _cvar_solver(0.5))

    def worst_pair_mean(allocation: OptimizedAllocation) -> float:
        curves = problem.frontier.eligible_curves()
        assert curves is not None
        utilities: list[float] = []
        for decision in allocation.allocation.decisions:
            curve = curves.for_client(decision.client_id)
            point = next(p for p in curve.points if p.target_rate == decision.target_rate)
            utilities.append(point.utility)
        return sum(sorted(utilities)[:2]) / 2.0

    assert worst_pair_mean(cvar_pair) >= worst_pair_mean(hard_min) - 1e-9


def test_fabrid_cvar_resolution_guard_excludes_infeasible_targets() -> None:
    inputs_list: list[ClientFrontierInputs] = []
    for client_id, calibration_count in (("a", 100_000), ("b", 100), ("c", 100_000)):
        frontier, validation = synthetic_client_scores(client_id, seed=1)
        inputs_list.append(
            build_client_frontier_inputs(
                FrontierScoreArtifacts(benign_frontier=frontier, attack_validation=validation),
                ALPHA_GRID,
                frontier_row_count=calibration_count,
            )
        )
    inputs = FederationFrontierInputs(clients=tuple(inputs_list))
    problem = build_allocation_problem(
        inputs,
        equal_client_weights(synthetic_population(("a", "b", "c"))),
        budget=0.002,
        eligibility=eligibility_config(),
        maximum_target_rate=LOCAL_TARGET_RATE_CAP,
    )
    curves = problem.require_eligible_curves()
    client_ids = tuple(curve.client_id for curve in curves.clients)
    target_rates = np.array(
        [point.target_rate for curve in curves.clients for point in curve.points],
        dtype=np.float64,
    )
    rows = allocation_policies.resolution_forbidden_rows(
        target_rates,
        curves.clients,
        problem.frontier_row_counts(),
        2.0,
    )
    assert client_ids == ("a", "b", "c")
    assert rows.shape == (3, 18)
    assert rows[0].sum() == 0.0
    assert rows[2].sum() == 0.0
    assert rows[1].tolist() == [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    optimized = allocate_fabrid_cvar(problem, _cvar_solver(0.25))
    counts = {item.client_id: item.row_count for item in problem.frontier_row_counts()}
    forbidden_by_client: dict[str, set[float]] = {}
    for curve in curves.clients:
        resolution_floor = 2.0 / (counts[curve.client_id] + 1)
        forbidden_by_client[curve.client_id] = {
            point.target_rate
            for point in curve.points
            if 0.0 < point.target_rate < resolution_floor
        }
    for decision in optimized.allocation.decisions:
        assert decision.target_rate not in forbidden_by_client[decision.client_id]


def test_fabrid_macro_and_cvar_use_the_same_resolution_feasible_grid() -> None:
    inputs_list: list[ClientFrontierInputs] = []
    for client_id, calibration_count in (("a", 100), ("b", 100_000)):
        frontier, validation = synthetic_client_scores(client_id, seed=1)
        inputs_list.append(
            build_client_frontier_inputs(
                FrontierScoreArtifacts(benign_frontier=frontier, attack_validation=validation),
                ALPHA_GRID,
                frontier_row_count=calibration_count,
            )
        )
    problem = build_allocation_problem(
        FederationFrontierInputs(clients=tuple(inputs_list)),
        equal_client_weights(synthetic_population(("a", "b"))),
        budget=0.002,
        eligibility=eligibility_config(),
        maximum_target_rate=LOCAL_TARGET_RATE_CAP,
    )
    counts = {item.client_id: item.row_count for item in problem.frontier_row_counts()}
    for allocation in (
        allocate_fabrid_macro(problem, _cvar_solver(0.5)).allocation,
        allocate_fabrid_cvar(problem, _cvar_solver(0.5)).allocation,
    ):
        for decision in allocation.decisions:
            assert not (0.0 < decision.target_rate < 2.0 / (counts[decision.client_id] + 1))
