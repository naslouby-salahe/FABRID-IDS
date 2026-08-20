from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint

from fabrid.allocation.optimization import (
    OptimizedAllocation,
    SolverEvidence,
    SolverStage,
    SolverStatus,
    budget_constraint,
    lexicographic_weights,
    one_hot_constraint,
    pad_constraint_columns,
    solve_milp,
)
from fabrid.allocation.problem import (
    Allocation,
    AllocationDecision,
    AllocationProblem,
    AllocationWeights,
    ClientRowCount,
    ClientUtilityCurve,
    ClientUtilityCurves,
)
from fabrid.config import (
    LEXICOGRAPHIC_TOLERANCE,
    MIP_FEASIBILITY_TOLERANCE,
    TIGHT_TOLERANCE,
    AllocationPolicy,
    CandidateCount,
    CandidateIndex,
    ClientCount,
    ClientId,
    FalsePositiveBudget,
    IncrementalBudgetCost,
    MarginalEfficiency,
    NonNegativeFloat,
    Probability,
    SolverConfig,
    TargetFalsePositiveRate,
    UtilityDifference,
)


def allocate_equal_fpr(problem: AllocationProblem) -> Allocation:
    if problem.budget > problem.maximum_target_rate:
        raise ValueError(f"budget {problem.budget} exceeds local cap {problem.maximum_target_rate}")
    target_rate = problem.budget
    return Allocation(
        policy=AllocationPolicy.EQ_FPR,
        decisions=tuple(
            AllocationDecision(client_id=client_id, target_rate=target_rate)
            for client_id in problem.population.clients
        ),
    )


def allocate_equal_alert(problem: AllocationProblem, solver: SolverConfig) -> Allocation:
    weights = problem.weights
    budget = problem.budget
    maximum_target_rate = problem.maximum_target_rate
    smallest_weight = min(client.weight for client in weights.clients)
    high = min(maximum_target_rate, budget / smallest_weight)
    low = 0.0
    for _ in range(solver.equal_alert_bisection_iterations):
        midpoint = (low + high) / 2.0
        weighted_cost = sum(
            min(midpoint, maximum_target_rate * client.weight) for client in weights.clients
        )
        if weighted_cost <= budget + TIGHT_TOLERANCE:
            low = midpoint
        else:
            high = midpoint
    decisions = tuple(
        AllocationDecision(
            client_id=client.client_id,
            target_rate=min(low / client.weight, maximum_target_rate),
        )
        for client in weights.clients
    )
    allocation = Allocation(policy=AllocationPolicy.EQ_ALERT, decisions=decisions)
    if not allocation.budget_feasibility(weights.allocation_weights, budget):
        raise ValueError("equal-alert allocation violates federation budget")
    return allocation


@dataclass(frozen=True, slots=True)
class _Position:
    client_id: ClientId
    point_index: CandidateIndex


@dataclass(frozen=True, slots=True)
class _Increment:
    client_id: ClientId
    next_index: CandidateIndex
    resulting_rate: TargetFalsePositiveRate
    delta_utility: UtilityDifference
    incremental_cost: IncrementalBudgetCost
    efficiency: MarginalEfficiency


def _best_increment(increments: tuple[_Increment, ...]) -> _Increment:
    return min(
        increments,
        key=lambda increment: (
            -increment.efficiency,
            -increment.delta_utility,
            increment.incremental_cost,
            increment.client_id,
            increment.resulting_rate,
        ),
    )


def _feasible_increment(
    curve: ClientUtilityCurve,
    index: CandidateIndex,
    weights: AllocationWeights,
    remaining_budget: FalsePositiveBudget,
    maximum_target_rate: TargetFalsePositiveRate,
) -> _Increment | None:
    next_index = index + 1
    if next_index >= len(curve.points):
        return None
    current_point = curve.points[index]
    next_point = curve.points[next_index]
    if next_point.target_rate > maximum_target_rate + TIGHT_TOLERANCE:
        return None
    delta_rate = next_point.target_rate - current_point.target_rate
    incremental_cost = weights.for_client(curve.client_id) * delta_rate
    if incremental_cost > remaining_budget + TIGHT_TOLERANCE:
        return None
    delta_utility = next_point.utility - current_point.utility
    efficiency = (
        math.inf if incremental_cost <= TIGHT_TOLERANCE else delta_utility / incremental_cost
    )
    return _Increment(
        client_id=curve.client_id,
        next_index=next_index,
        resulting_rate=next_point.target_rate,
        delta_utility=delta_utility,
        incremental_cost=incremental_cost,
        efficiency=efficiency,
    )


def allocate_greedy(problem: AllocationProblem) -> Allocation:
    utility_curves = problem.require_eligible_curves()
    weights = problem.eligible_weights()
    remaining_budget = problem.remaining_budget
    maximum_target_rate = problem.maximum_target_rate
    positions = [
        _Position(client_id=curve.client_id, point_index=0) for curve in utility_curves.clients
    ]
    while True:
        candidates: list[_Increment] = []
        for position in positions:
            increment = _feasible_increment(
                curve=utility_curves.for_client(position.client_id),
                index=position.point_index,
                weights=weights,
                remaining_budget=remaining_budget,
                maximum_target_rate=maximum_target_rate,
            )
            if increment is not None:
                candidates.append(increment)
        if not candidates:
            break
        chosen = _best_increment(tuple(candidates))
        positions = [
            _Position(
                client_id=position.client_id,
                point_index=(
                    chosen.next_index
                    if position.client_id == chosen.client_id
                    else position.point_index
                ),
            )
            for position in positions
        ]
        remaining_budget = max(0.0, remaining_budget - chosen.incremental_cost)
    return Allocation(
        policy=AllocationPolicy.GREEDY,
        decisions=tuple(
            AllocationDecision(
                client_id=position.client_id,
                target_rate=utility_curves.for_client(position.client_id)
                .points[position.point_index]
                .target_rate,
            )
            for position in positions
        ),
    )


@dataclass(frozen=True, slots=True)
class _DiscreteProgram:
    curves: tuple[ClientUtilityCurve, ...]
    client_count: ClientCount
    candidate_count: CandidateCount
    utility: np.ndarray
    utility_variance: np.ndarray
    cost: np.ndarray
    target_rates: np.ndarray


def _discrete_program(
    utility_curves: ClientUtilityCurves,
    weights: AllocationWeights,
) -> _DiscreteProgram:
    curve_client_ids = {curve.client_id for curve in utility_curves.clients}
    weight_client_ids = {client.client_id for client in weights.clients}
    if curve_client_ids != weight_client_ids:
        raise ValueError("utility curves and allocation weights must share clients")
    curves = tuple(sorted(utility_curves.clients, key=lambda curve: curve.client_id))
    return _DiscreteProgram(
        curves=curves,
        client_count=len(curves),
        candidate_count=len(curves[0].points),
        utility=np.array(
            [point.utility for curve in curves for point in curve.points],
            dtype=np.float64,
        ),
        utility_variance=np.array(
            [point.utility_variance for curve in curves for point in curve.points],
            dtype=np.float64,
        ),
        cost=np.array(
            [
                weights.for_client(curve.client_id) * point.target_rate
                for curve in curves
                for point in curve.points
            ],
            dtype=np.float64,
        ),
        target_rates=np.array(
            [point.target_rate for curve in curves for point in curve.points],
            dtype=np.float64,
        ),
    )


def _selected_allocation(
    policy: AllocationPolicy,
    program: _DiscreteProgram,
    selection: np.ndarray,
) -> Allocation:
    return Allocation(
        policy=policy,
        decisions=tuple(
            AllocationDecision(
                client_id=curve.client_id,
                target_rate=curve.points[int(np.argmax(selection[index]))].target_rate,
            )
            for index, curve in enumerate(program.curves)
        ),
    )


def allocate_fabrid_macro(
    problem: AllocationProblem,
    solver: SolverConfig,
) -> OptimizedAllocation:
    program = _discrete_program(problem.require_eligible_curves(), problem.eligible_weights())
    remaining_budget = problem.remaining_budget
    one_hot = one_hot_constraint(program.client_count, program.candidate_count)
    budget = budget_constraint(program.cost, remaining_budget)
    bounds = Bounds(0.0, 1.0)
    integrality = np.ones(program.client_count * program.candidate_count)
    utility_tolerance = max(LEXICOGRAPHIC_TOLERANCE, solver.accepted_gap)
    budget_tolerance = max(LEXICOGRAPHIC_TOLERANCE, solver.accepted_gap)
    stage_one = solve_milp(
        SolverStage.MACRO_PRIMARY_UTILITY,
        -program.utility / program.client_count,
        (one_hot, budget),
        integrality,
        bounds,
        solver,
    )
    optimal_mean_utility = -stage_one.evidence.objective
    utility_floor = LinearConstraint(
        (program.utility / program.client_count).reshape(1, -1),
        lb=optimal_mean_utility - utility_tolerance,
        ub=np.inf,
    )
    utility_floor_cheat_guard = LinearConstraint(
        (program.utility / program.client_count).reshape(1, -1),
        lb=optimal_mean_utility - utility_tolerance + MIP_FEASIBILITY_TOLERANCE,
        ub=np.inf,
    )
    stage_two = solve_milp(
        SolverStage.MACRO_MINIMUM_BUDGET,
        program.cost,
        (one_hot, budget, utility_floor_cheat_guard),
        integrality,
        bounds,
        solver,
    )
    optimal_cost = stage_two.evidence.objective
    cost_ceiling = LinearConstraint(
        program.cost.reshape(1, -1),
        lb=-np.inf,
        ub=optimal_cost + budget_tolerance,
    )
    cost_ceiling_cheat_guard = LinearConstraint(
        program.cost.reshape(1, -1),
        lb=-np.inf,
        ub=optimal_cost + budget_tolerance - MIP_FEASIBILITY_TOLERANCE,
    )
    stage_three = solve_milp(
        SolverStage.MACRO_LOWEST_VARIANCE,
        program.utility_variance / program.client_count,
        (one_hot, budget, utility_floor_cheat_guard, cost_ceiling_cheat_guard),
        integrality,
        bounds,
        solver,
    )
    optimal_variance = stage_three.evidence.objective
    variance_ceiling = LinearConstraint(
        (program.utility_variance / program.client_count).reshape(1, -1),
        lb=-np.inf,
        ub=optimal_variance + utility_tolerance,
    )
    stage_four = solve_milp(
        SolverStage.MACRO_TIE_BREAK,
        program.target_rates * lexicographic_weights(program.client_count, program.candidate_count),
        (one_hot, budget, utility_floor, cost_ceiling, variance_ceiling),
        integrality,
        bounds,
        solver,
    )
    selection = (
        np.round(stage_four.variables)
        .astype(np.int64)
        .reshape(program.client_count, program.candidate_count)
    )
    return OptimizedAllocation(
        allocation=_selected_allocation(AllocationPolicy.FABRID_MACRO, program, selection),
        solver=SolverEvidence(
            status=SolverStatus.OPTIMAL,
            stages=(
                stage_one.evidence,
                stage_two.evidence,
                stage_three.evidence,
                stage_four.evidence,
            ),
        ),
    )


def _padded_row(row: np.ndarray, variable_count: CandidateCount) -> np.ndarray:
    padded = np.zeros(variable_count)
    padded[: row.shape[0]] = row
    return padded.reshape(1, -1)


def _cvar_client_count(client_count: ClientCount, quantile: Probability) -> ClientCount:
    return max(1, math.ceil(quantile * client_count))


def _cvar_auxiliary_variable_count(
    binary_variable_count: CandidateCount,
    client_count: ClientCount,
) -> CandidateCount:
    return binary_variable_count + 1 + client_count


def _cvar_threshold_index(binary_variable_count: CandidateCount) -> CandidateIndex:
    return binary_variable_count


def _cvar_captured_utility_indices(
    binary_variable_count: CandidateCount,
    client_count: ClientCount,
) -> np.ndarray:
    start = binary_variable_count + 1
    return np.arange(start, start + client_count, dtype=np.int64)


def _cvar_objective_coefficients(
    binary_variable_count: CandidateCount,
    client_count: ClientCount,
    q: ClientCount,
) -> np.ndarray:
    coefficients = np.zeros(binary_variable_count + 1 + client_count)
    coefficients[_cvar_threshold_index(binary_variable_count)] = float(q)
    coefficients[_cvar_captured_utility_indices(binary_variable_count, client_count)] = 1.0
    return coefficients


def _cvar_structure_constraints(
    utility: np.ndarray,
    client_count: ClientCount,
    candidate_count: CandidateCount,
    variable_count: CandidateCount,
) -> tuple[LinearConstraint, ...]:
    threshold_index = _cvar_threshold_index(variable_count - 1 - client_count)
    captured = _cvar_captured_utility_indices(variable_count - 1 - client_count, client_count)
    rows = np.zeros((client_count, variable_count))
    for client_index in range(client_count):
        start = client_index * candidate_count
        rows[client_index, start : start + candidate_count] = utility[
            start : start + candidate_count
        ]
        rows[client_index, threshold_index] = 1.0
        rows[client_index, captured[client_index]] = 1.0
    return (LinearConstraint(rows, lb=0.0, ub=np.inf),)


def resolution_forbidden_rows(
    target_rates: np.ndarray,
    curves: tuple[ClientUtilityCurve, ...],
    calibration_counts: tuple[ClientRowCount, ...],
    resolution_factor: NonNegativeFloat,
) -> np.ndarray:
    count_by_client = {item.client_id: item.row_count for item in calibration_counts}
    client_count = len(curves)
    candidate_count = len(curves[0].points)
    rates = target_rates.reshape(client_count, candidate_count)
    rows = np.zeros((client_count, client_count * candidate_count))
    for client_index, curve in enumerate(curves):
        calibration_count = count_by_client.get(curve.client_id)
        if calibration_count is None:
            raise KeyError(curve.client_id)
        resolution_floor = resolution_factor / (calibration_count + 1)
        for candidate_index, rate in enumerate(rates[client_index]):
            if 0.0 < rate < resolution_floor:
                rows[client_index, client_index * candidate_count + candidate_index] = 1.0
    return rows


def allocate_fabrid_minimax(
    problem: AllocationProblem,
    solver: SolverConfig,
) -> OptimizedAllocation:
    program = _discrete_program(problem.require_eligible_curves(), problem.eligible_weights())
    remaining_budget = problem.remaining_budget
    binary_variable_count = program.client_count * program.candidate_count
    variable_count = _cvar_auxiliary_variable_count(binary_variable_count, program.client_count)
    worst_client_count = _cvar_client_count(
        program.client_count, solver.minimax_worst_client_quantile
    )
    one_hot_binary = one_hot_constraint(program.client_count, program.candidate_count)
    budget_binary = budget_constraint(program.cost, remaining_budget)
    one_hot_matrix = np.zeros((program.client_count, binary_variable_count))
    for client_index in range(program.client_count):
        start = client_index * program.candidate_count
        one_hot_matrix[client_index, start : start + program.candidate_count] = 1.0
    cvar_structure = _cvar_structure_constraints(
        program.utility,
        program.client_count,
        program.candidate_count,
        variable_count,
    )
    resolution_rows = resolution_forbidden_rows(
        program.target_rates,
        program.curves,
        problem.calibration_row_counts(),
        solver.calibration_resolution_factor,
    )
    resolution = LinearConstraint(resolution_rows, lb=0.0, ub=0.0)
    constraints = (
        pad_constraint_columns(one_hot_binary, one_hot_matrix, 1 + program.client_count),
        pad_constraint_columns(
            budget_binary, program.cost.reshape(1, -1), 1 + program.client_count
        ),
        *cvar_structure,
        pad_constraint_columns(resolution, resolution_rows, 1 + program.client_count),
    )
    integrality = np.concatenate(
        [np.ones(binary_variable_count), np.zeros(1 + program.client_count)]
    )
    lower_bounds = np.concatenate(
        [
            np.zeros(binary_variable_count),
            np.array([-1.0]),
            np.zeros(program.client_count),
        ]
    )
    upper_bounds = np.concatenate(
        [
            np.ones(binary_variable_count),
            np.array([0.0]),
            np.ones(program.client_count),
        ]
    )
    bounds = Bounds(lower_bounds, upper_bounds)
    z_tolerance = max(LEXICOGRAPHIC_TOLERANCE, solver.accepted_gap)
    utility_tolerance = max(LEXICOGRAPHIC_TOLERANCE, solver.accepted_gap)
    budget_tolerance = max(LEXICOGRAPHIC_TOLERANCE, solver.accepted_gap)
    stage_one = solve_milp(
        SolverStage.MINIMAX_CVAR_UTILITY,
        _cvar_objective_coefficients(
            binary_variable_count, program.client_count, worst_client_count
        ),
        constraints,
        integrality,
        bounds,
        solver,
    )
    optimal_cvar = -stage_one.evidence.objective / worst_client_count
    cvar_floor_row = np.zeros((1, variable_count))
    cvar_floor_row[0, _cvar_threshold_index(binary_variable_count)] = float(worst_client_count)
    cvar_floor_row[
        0, _cvar_captured_utility_indices(binary_variable_count, program.client_count)
    ] = 1.0
    cvar_floor = LinearConstraint(
        cvar_floor_row,
        lb=-np.inf,
        ub=-worst_client_count * (optimal_cvar - z_tolerance),
    )
    cvar_floor_cheat_guard = LinearConstraint(
        cvar_floor_row,
        lb=-np.inf,
        ub=-worst_client_count * (optimal_cvar - z_tolerance + MIP_FEASIBILITY_TOLERANCE),
    )
    constraints_with_cvar = (*constraints, cvar_floor_cheat_guard)
    stage_two_objective = np.zeros(variable_count)
    stage_two_objective[:binary_variable_count] = -program.utility / program.client_count
    stage_two = solve_milp(
        SolverStage.MINIMAX_MEAN_UTILITY,
        stage_two_objective,
        constraints_with_cvar,
        integrality,
        bounds,
        solver,
    )
    optimal_mean_utility = -stage_two.evidence.objective
    mean_utility_floor = LinearConstraint(
        _padded_row(program.utility / program.client_count, variable_count),
        lb=optimal_mean_utility - utility_tolerance,
        ub=np.inf,
    )
    mean_utility_floor_cheat_guard = LinearConstraint(
        _padded_row(program.utility / program.client_count, variable_count),
        lb=optimal_mean_utility - utility_tolerance + MIP_FEASIBILITY_TOLERANCE,
        ub=np.inf,
    )
    constraints_with_mean = (
        *constraints,
        cvar_floor_cheat_guard,
        mean_utility_floor_cheat_guard,
    )
    stage_three_objective = np.zeros(variable_count)
    stage_three_objective[:binary_variable_count] = program.cost
    stage_three = solve_milp(
        SolverStage.MINIMAX_MINIMUM_BUDGET,
        stage_three_objective,
        constraints_with_mean,
        integrality,
        bounds,
        solver,
    )
    optimal_cost = stage_three.evidence.objective
    cost_ceiling = LinearConstraint(
        _padded_row(program.cost, variable_count),
        lb=-np.inf,
        ub=optimal_cost + budget_tolerance,
    )
    cost_ceiling_cheat_guard = LinearConstraint(
        _padded_row(program.cost, variable_count),
        lb=-np.inf,
        ub=optimal_cost + budget_tolerance - MIP_FEASIBILITY_TOLERANCE,
    )
    constraints_with_cost = (
        *constraints,
        cvar_floor_cheat_guard,
        mean_utility_floor_cheat_guard,
        cost_ceiling_cheat_guard,
    )
    stage_four_objective = np.zeros(variable_count)
    stage_four_objective[:binary_variable_count] = program.utility_variance / program.client_count
    stage_four = solve_milp(
        SolverStage.MINIMAX_LOWEST_VARIANCE,
        stage_four_objective,
        constraints_with_cost,
        integrality,
        bounds,
        solver,
    )
    optimal_variance = stage_four.evidence.objective
    variance_ceiling = LinearConstraint(
        _padded_row(program.utility_variance / program.client_count, variable_count),
        lb=-np.inf,
        ub=optimal_variance + utility_tolerance,
    )
    constraints_final = (
        *constraints,
        cvar_floor,
        mean_utility_floor,
        cost_ceiling,
        variance_ceiling,
    )
    stage_five_objective = np.zeros(variable_count)
    stage_five_objective[:binary_variable_count] = program.target_rates * lexicographic_weights(
        program.client_count, program.candidate_count
    )
    stage_five = solve_milp(
        SolverStage.MINIMAX_TIE_BREAK,
        stage_five_objective,
        constraints_final,
        integrality,
        bounds,
        solver,
    )
    selection = (
        np.round(stage_five.variables[:binary_variable_count])
        .astype(np.int64)
        .reshape(program.client_count, program.candidate_count)
    )
    return OptimizedAllocation(
        allocation=_selected_allocation(AllocationPolicy.FABRID_MINIMAX, program, selection),
        solver=SolverEvidence(
            status=SolverStatus.OPTIMAL,
            stages=(
                stage_one.evidence,
                stage_two.evidence,
                stage_three.evidence,
                stage_four.evidence,
                stage_five.evidence,
            ),
        ),
    )
