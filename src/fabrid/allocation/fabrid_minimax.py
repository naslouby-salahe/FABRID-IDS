from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds, LinearConstraint

from fabrid.allocation.contracts import (
    Allocation,
    AllocationDecision,
    ClientUtilityCurves,
    FederationWeights,
)
from fabrid.allocation.formulation import (
    budget_constraint,
    lexicographic_weights,
    one_hot_constraint,
    pad_constraint_columns,
)
from fabrid.allocation.solver import solve_milp
from fabrid.domain.enums import AllocationPolicy
from fabrid.domain.values import FalsePositiveBudget, RowCount
from fabrid.protocol.models import SolverSettings

_Z_TOLERANCE_FLOOR = 1e-9
_UTILITY_TOLERANCE_FLOOR = 1e-9
_BUDGET_TOLERANCE_FLOOR = 1e-12


def _per_client_utility_floor_constraint(
    utility: np.ndarray,
    client_count: RowCount,
    candidate_count: RowCount,
    variable_count: RowCount,
    minimum_utility_index: int,
) -> LinearConstraint:
    rows = np.zeros((client_count.value, variable_count.value))
    for client_index in range(client_count.value):
        start = client_index * candidate_count.value
        rows[
            client_index,
            start : start + candidate_count.value,
        ] = utility[start : start + candidate_count.value]
        rows[client_index, minimum_utility_index] = -1.0
    return LinearConstraint(rows, lb=0.0, ub=np.inf)


def _padded_row(row: np.ndarray, variable_count: RowCount) -> np.ndarray:
    padded = np.zeros(variable_count.value)
    padded[: row.shape[0]] = row
    return padded.reshape(1, -1)


def allocate_fabrid_minimax(
    utility_curves: ClientUtilityCurves,
    weights: FederationWeights,
    remaining_budget: FalsePositiveBudget,
    settings: SolverSettings,
) -> Allocation:
    curve_client_ids = {curve.client_id for curve in utility_curves.clients}
    weight_client_ids = {client.client_id for client in weights.clients}
    if curve_client_ids != weight_client_ids:
        raise ValueError("utility curves and federation weights must share clients")

    curves = tuple(
        sorted(utility_curves.clients, key=lambda curve: curve.client_id.value)
    )
    client_count = RowCount(len(curves))
    candidate_count = RowCount(len(curves[0].points))
    binary_variable_count = RowCount(client_count.value * candidate_count.value)
    variable_count = RowCount(binary_variable_count.value + 1)
    minimum_utility_index = binary_variable_count.value

    utility = np.array(
        [
            point.utility.value
            for curve in curves
            for point in curve.points
        ],
        dtype=np.float64,
    )
    cost = np.array(
        [
            weights.for_client(curve.client_id).value * point.target_rate.value
            for curve in curves
            for point in curve.points
        ],
        dtype=np.float64,
    )

    one_hot_binary = one_hot_constraint(client_count, candidate_count)
    budget_binary = budget_constraint(cost, remaining_budget)
    one_hot_matrix = np.zeros(
        (client_count.value, binary_variable_count.value)
    )
    for client_index in range(client_count.value):
        start = client_index * candidate_count.value
        one_hot_matrix[
            client_index,
            start : start + candidate_count.value,
        ] = 1.0

    utility_floor_per_client = _per_client_utility_floor_constraint(
        utility,
        client_count,
        candidate_count,
        variable_count,
        minimum_utility_index,
    )
    constraints = (
        pad_constraint_columns(one_hot_binary, one_hot_matrix, RowCount(1)),
        pad_constraint_columns(
            budget_binary,
            cost.reshape(1, -1),
            RowCount(1),
        ),
        utility_floor_per_client,
    )

    integrality = np.concatenate(
        [np.ones(binary_variable_count.value), np.zeros(1)]
    )
    bounds = Bounds(np.zeros(variable_count.value), np.ones(variable_count.value))

    z_tolerance = max(_Z_TOLERANCE_FLOOR, settings.accepted_gap.value)
    utility_tolerance = max(
        _UTILITY_TOLERANCE_FLOOR,
        settings.accepted_gap.value,
    )
    budget_tolerance = max(
        _BUDGET_TOLERANCE_FLOOR,
        settings.accepted_gap.value,
    )

    stage_one_objective = np.zeros(variable_count.value)
    stage_one_objective[minimum_utility_index] = -1.0
    stage_one = solve_milp(
        stage_one_objective,
        constraints,
        integrality,
        bounds,
        settings,
    )
    optimal_minimum_utility = -stage_one.objective.value

    minimum_utility_floor_row = np.zeros((1, variable_count.value))
    minimum_utility_floor_row[0, minimum_utility_index] = 1.0
    minimum_utility_floor = LinearConstraint(
        minimum_utility_floor_row,
        lb=optimal_minimum_utility - z_tolerance,
        ub=np.inf,
    )
    constraints_with_minimum = (*constraints, minimum_utility_floor)

    stage_two_objective = np.zeros(variable_count.value)
    stage_two_objective[: binary_variable_count.value] = (
        -utility / client_count.value
    )
    stage_two = solve_milp(
        stage_two_objective,
        constraints_with_minimum,
        integrality,
        bounds,
        settings,
    )
    optimal_mean_utility = -stage_two.objective.value

    mean_utility_floor = LinearConstraint(
        _padded_row(utility / client_count.value, variable_count),
        lb=optimal_mean_utility - utility_tolerance,
        ub=np.inf,
    )
    constraints_with_utility = (*constraints_with_minimum, mean_utility_floor)

    stage_three_objective = np.zeros(variable_count.value)
    stage_three_objective[: binary_variable_count.value] = cost
    stage_three = solve_milp(
        stage_three_objective,
        constraints_with_utility,
        integrality,
        bounds,
        settings,
    )
    optimal_cost = stage_three.objective.value

    cost_ceiling = LinearConstraint(
        _padded_row(cost, variable_count),
        lb=-np.inf,
        ub=optimal_cost + budget_tolerance,
    )
    constraints_final = (*constraints_with_utility, cost_ceiling)

    target_rates = np.array(
        [point.target_rate.value for curve in curves for point in curve.points],
        dtype=np.float64,
    )
    stage_four_objective = np.zeros(variable_count.value)
    stage_four_objective[: binary_variable_count.value] = (
        target_rates * lexicographic_weights(client_count, candidate_count)
    )
    stage_four = solve_milp(
        stage_four_objective,
        constraints_final,
        integrality,
        bounds,
        settings,
    )

    selection = np.round(stage_four.variables[: binary_variable_count.value]).astype(
        np.int64
    ).reshape(client_count.value, candidate_count.value)
    return Allocation(
        policy=AllocationPolicy.FABRID_MINIMAX,
        decisions=tuple(
            AllocationDecision(
                client_id=curve.client_id,
                target_rate=curve.points[
                    int(np.argmax(selection[index]))
                ].target_rate,
            )
            for index, curve in enumerate(curves)
        ),
    )
