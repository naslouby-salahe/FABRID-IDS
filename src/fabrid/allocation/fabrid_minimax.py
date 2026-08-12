"""FABRID_MINIMAX: choose one candidate rate per eligible client maximizing the worst-off client.

Two-stage optimization: first maximize the minimum per-client utility `z`
(via an auxiliary continuous variable appended after the one-hot binary
block), then, holding `z` fixed at its optimum, maximize mean utility.
Deterministic tie-breaking continues with budget minimization and
lexicographic alpha minimization, mirroring FABRID_MACRO's later stages.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy.optimize import Bounds, LinearConstraint

from fabrid.allocation.formulation import (
    budget_constraint,
    lexicographic_weights,
    one_hot_constraint,
    pad_constraint_columns,
)
from fabrid.config.protocol import SolverSettings
from fabrid.evaluation.record_level import ClientId
from fabrid.optimization.milp import solve_milp
from fabrid.schemas.allocation import (
    Allocation,
    AllocationDecision,
    AllocationPolicy,
    ClientUtilityCurve,
)

_Z_TOLERANCE = 1e-9
_UTILITY_TOLERANCE = 1e-9
_BUDGET_TOLERANCE = 1e-12


def _shared_client_grid(
    client_ids: list[ClientId], utility_curves: Mapping[ClientId, ClientUtilityCurve]
) -> tuple[float, ...]:
    shared_grid = utility_curves[client_ids[0]].alpha_grid
    for client_id in client_ids:
        if utility_curves[client_id].alpha_grid != shared_grid:
            raise ValueError("all clients must share the same candidate target-rate grid")
    return shared_grid


def _per_client_utility_floor_constraint(
    utility: np.ndarray, n_clients: int, n_candidates: int, n_vars: int, z_index: int
) -> LinearConstraint:
    """utility_k(selected) - z >= 0 for every client k."""
    rows = np.zeros((n_clients, n_vars))
    for client_index in range(n_clients):
        start = client_index * n_candidates
        rows[client_index, start : start + n_candidates] = utility[start : start + n_candidates]
        rows[client_index, z_index] = -1.0
    return LinearConstraint(rows, lb=0.0, ub=np.inf)


def _padded_row(row: np.ndarray, n_vars: int) -> np.ndarray:
    padded = np.zeros(n_vars)
    padded[: row.shape[0]] = row
    return padded.reshape(1, -1)


def allocate_fabrid_minimax(
    utility_curves: Mapping[ClientId, ClientUtilityCurve],
    weight: Mapping[ClientId, float],
    remaining_budget: float,
    settings: SolverSettings,
) -> Allocation:
    if not utility_curves:
        raise ValueError("allocate_fabrid_minimax requires at least one eligible client")
    if utility_curves.keys() != weight.keys():
        raise ValueError("utility_curves and weight must share the same client set")

    client_ids = sorted(utility_curves.keys())
    shared_grid = _shared_client_grid(client_ids, utility_curves)
    n_clients = len(client_ids)
    n_candidates = len(shared_grid)
    n_binary = n_clients * n_candidates
    n_vars = n_binary + 1
    z_index = n_binary

    utility = np.array([utility_curves[c].utility for c in client_ids], dtype=np.float64).reshape(
        -1
    )
    cost = np.array(
        [[weight[c] * alpha for alpha in shared_grid] for c in client_ids], dtype=np.float64
    ).reshape(-1)

    one_hot_binary_only = one_hot_constraint(n_clients, n_candidates)
    budget_binary_only = budget_constraint(cost, remaining_budget)
    one_hot_matrix = np.zeros((n_clients, n_binary))
    for client_index in range(n_clients):
        start = client_index * n_candidates
        one_hot_matrix[client_index, start : start + n_candidates] = 1.0
    utility_floor_per_client = _per_client_utility_floor_constraint(
        utility, n_clients, n_candidates, n_vars, z_index
    )
    constraints = [
        pad_constraint_columns(one_hot_binary_only, one_hot_matrix, extra_columns=1),
        pad_constraint_columns(budget_binary_only, cost.reshape(1, -1), extra_columns=1),
        utility_floor_per_client,
    ]

    integrality = np.concatenate([np.ones(n_binary), np.zeros(1)])
    bounds = Bounds(np.zeros(n_vars), np.ones(n_vars))

    # Stage 1: maximize z (worst-client utility).
    stage1_objective = np.zeros(n_vars)
    stage1_objective[z_index] = -1.0
    stage1 = solve_milp(stage1_objective, constraints, integrality, bounds, settings)
    optimal_z = -stage1.objective_value

    z_floor_row = np.zeros((1, n_vars))
    z_floor_row[0, z_index] = 1.0
    z_floor = LinearConstraint(z_floor_row, lb=optimal_z - _Z_TOLERANCE, ub=np.inf)
    constraints_with_z = [*constraints, z_floor]

    # Stage 2: among worst-client-optimal solutions, maximize mean utility.
    stage2_objective = np.zeros(n_vars)
    stage2_objective[:n_binary] = -utility / n_clients
    stage2 = solve_milp(stage2_objective, constraints_with_z, integrality, bounds, settings)
    optimal_mean_utility = -stage2.objective_value

    utility_floor_mean = LinearConstraint(
        _padded_row(utility / n_clients, n_vars),
        lb=optimal_mean_utility - _UTILITY_TOLERANCE,
        ub=np.inf,
    )
    constraints_with_utility = [*constraints_with_z, utility_floor_mean]

    # Stage 3: minimize total weighted budget consumption.
    stage3_objective = np.zeros(n_vars)
    stage3_objective[:n_binary] = cost
    stage3 = solve_milp(stage3_objective, constraints_with_utility, integrality, bounds, settings)
    optimal_cost = stage3.objective_value

    cost_ceiling = LinearConstraint(
        _padded_row(cost, n_vars), lb=-np.inf, ub=optimal_cost + _BUDGET_TOLERANCE
    )
    constraints_final = [*constraints_with_utility, cost_ceiling]

    # Stage 4: lexicographically minimize the selected alpha vector by client order.
    alpha_flat = np.array([alpha for _ in client_ids for alpha in shared_grid], dtype=np.float64)
    stage4_objective = np.zeros(n_vars)
    stage4_objective[:n_binary] = alpha_flat * lexicographic_weights(n_clients, n_candidates)
    stage4 = solve_milp(stage4_objective, constraints_final, integrality, bounds, settings)

    selection = np.round(stage4.x[:n_binary]).astype(np.int64).reshape(n_clients, n_candidates)
    decisions = {
        client_id: AllocationDecision(
            client_id=client_id, alpha_selected=shared_grid[int(np.argmax(selection[i]))]
        )
        for i, client_id in enumerate(client_ids)
    }
    return Allocation(policy=AllocationPolicy.FABRID_MINIMAX, decisions=decisions)
