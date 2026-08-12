"""FABRID_MACRO: choose one candidate rate per eligible client maximizing mean utility.

One-hot binary selection per client, subject to a shared weighted budget.
Deterministic tie-breaking via a sequential-solve procedure: maximize mean
utility, then minimize total budget consumption among optima, then
lexicographically minimize the selected rate vector ordered by client ID.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy.optimize import Bounds, LinearConstraint

from fabrid.allocation.formulation import (
    budget_constraint,
    lexicographic_weights,
    one_hot_constraint,
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


def _flatten_utility_and_cost(
    client_ids: list[ClientId],
    utility_curves: Mapping[ClientId, ClientUtilityCurve],
    weight: Mapping[ClientId, float],
    shared_grid: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray]:
    utility = np.array([utility_curves[c].utility for c in client_ids], dtype=np.float64).reshape(
        -1
    )
    cost = np.array(
        [[weight[c] * alpha for alpha in shared_grid] for c in client_ids], dtype=np.float64
    ).reshape(-1)
    return utility, cost


def allocate_fabrid_macro(
    utility_curves: Mapping[ClientId, ClientUtilityCurve],
    weight: Mapping[ClientId, float],
    remaining_budget: float,
    settings: SolverSettings,
) -> Allocation:
    """`remaining_budget` is `B_R`: the budget left after any fallback reservation."""
    if not utility_curves:
        raise ValueError("allocate_fabrid_macro requires at least one eligible client")
    if utility_curves.keys() != weight.keys():
        raise ValueError("utility_curves and weight must share the same client set")

    client_ids = sorted(utility_curves.keys())
    shared_grid = _shared_client_grid(client_ids, utility_curves)
    n_clients = len(client_ids)
    n_candidates = len(shared_grid)

    utility, cost = _flatten_utility_and_cost(client_ids, utility_curves, weight, shared_grid)

    one_hot = one_hot_constraint(n_clients, n_candidates)
    budget = budget_constraint(cost, remaining_budget)
    bounds = Bounds(0.0, 1.0)
    integrality = np.ones(n_clients * n_candidates)

    # Stage 1: maximize mean utility (milp minimizes, so negate).
    stage1 = solve_milp(-utility / n_clients, [one_hot, budget], integrality, bounds, settings)
    optimal_mean_utility = -stage1.objective_value

    # Stage 2: among near-optimal-utility solutions, minimize total budget consumption.
    utility_floor = LinearConstraint(
        (utility / n_clients).reshape(1, -1),
        lb=optimal_mean_utility - _UTILITY_TOLERANCE,
        ub=np.inf,
    )
    stage2 = solve_milp(cost, [one_hot, budget, utility_floor], integrality, bounds, settings)
    optimal_cost = stage2.objective_value

    # Stage 3: lexicographically minimize the selected alpha vector by client order.
    cost_ceiling = LinearConstraint(
        cost.reshape(1, -1), lb=-np.inf, ub=optimal_cost + _BUDGET_TOLERANCE
    )
    alpha_flat = np.array([alpha for _ in client_ids for alpha in shared_grid], dtype=np.float64)
    stage3 = solve_milp(
        alpha_flat * lexicographic_weights(n_clients, n_candidates),
        [one_hot, budget, utility_floor, cost_ceiling],
        integrality,
        bounds,
        settings,
    )

    selection = np.round(stage3.x).astype(np.int64).reshape(n_clients, n_candidates)
    decisions = {
        client_id: AllocationDecision(
            client_id=client_id, alpha_selected=shared_grid[int(np.argmax(selection[i]))]
        )
        for i, client_id in enumerate(client_ids)
    }
    return Allocation(policy=AllocationPolicy.FABRID_MACRO, decisions=decisions)
