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
)
from fabrid.allocation.solver import solve_milp
from fabrid.domain.enums import AllocationPolicy
from fabrid.domain.values import FalsePositiveBudget, RowCount
from fabrid.protocol.models import SolverSettings

_UTILITY_TOLERANCE_FLOOR = 1e-9
_BUDGET_TOLERANCE_FLOOR = 1e-12


def allocate_fabrid_macro(
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

    one_hot = one_hot_constraint(client_count, candidate_count)
    budget = budget_constraint(cost, remaining_budget)
    bounds = Bounds(0.0, 1.0)
    integrality = np.ones(client_count.value * candidate_count.value)

    utility_tolerance = max(
        _UTILITY_TOLERANCE_FLOOR,
        settings.accepted_gap.value,
    )
    budget_tolerance = max(
        _BUDGET_TOLERANCE_FLOOR,
        settings.accepted_gap.value,
    )

    stage_one = solve_milp(
        -utility / client_count.value,
        (one_hot, budget),
        integrality,
        bounds,
        settings,
    )
    optimal_mean_utility = -stage_one.objective.value

    utility_floor = LinearConstraint(
        (utility / client_count.value).reshape(1, -1),
        lb=optimal_mean_utility - utility_tolerance,
        ub=np.inf,
    )
    stage_two = solve_milp(
        cost,
        (one_hot, budget, utility_floor),
        integrality,
        bounds,
        settings,
    )
    optimal_cost = stage_two.objective.value

    cost_ceiling = LinearConstraint(
        cost.reshape(1, -1),
        lb=-np.inf,
        ub=optimal_cost + budget_tolerance,
    )
    target_rates = np.array(
        [point.target_rate.value for curve in curves for point in curve.points],
        dtype=np.float64,
    )
    stage_three = solve_milp(
        target_rates * lexicographic_weights(client_count, candidate_count),
        (one_hot, budget, utility_floor, cost_ceiling),
        integrality,
        bounds,
        settings,
    )

    selection = np.round(stage_three.variables).astype(np.int64).reshape(
        client_count.value,
        candidate_count.value,
    )
    return Allocation(
        policy=AllocationPolicy.FABRID_MACRO,
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
