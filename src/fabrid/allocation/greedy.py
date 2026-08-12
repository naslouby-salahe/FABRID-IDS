"""GREEDY baseline: iterative marginal-utility-efficiency budget allocation.

Starting from alpha_k = 0 for every client, repeatedly grants the single
feasible next-grid-point increment with the largest marginal utility per unit
weighted budget, until no client can afford its next increment.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from fabrid.evaluation.record_level import ClientId
from fabrid.schemas.allocation import (
    Allocation,
    AllocationDecision,
    AllocationPolicy,
    ClientUtilityCurve,
)

_BUDGET_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class _Increment:
    client_id: ClientId
    next_index: int
    resulting_alpha: float
    delta_utility: float
    incremental_cost: float
    efficiency: float


def _best_increment(increments: list[_Increment]) -> _Increment:
    # Tie order: largest efficiency; then larger delta_utility; then lower
    # incremental cost; then lower client_id; then lower resulting alpha.
    # Expressed as a min-key by negating the two "prefer largest" fields.
    return min(
        increments,
        key=lambda inc: (
            -inc.efficiency,
            -inc.delta_utility,
            inc.incremental_cost,
            inc.client_id,
            inc.resulting_alpha,
        ),
    )


def _feasible_increment(
    client_id: ClientId,
    curve: ClientUtilityCurve,
    index: int,
    client_weight: float,
    remaining_budget: float,
    alpha_max: float,
) -> _Increment | None:
    if index + 1 >= len(curve.alpha_grid):
        return None
    next_alpha = curve.alpha_grid[index + 1]
    if next_alpha > alpha_max + _BUDGET_TOLERANCE:
        return None
    delta_alpha = next_alpha - curve.alpha_grid[index]
    incremental_cost = client_weight * delta_alpha
    if incremental_cost > remaining_budget + _BUDGET_TOLERANCE:
        return None
    delta_utility = curve.utility_at_index(index + 1) - curve.utility_at_index(index)
    efficiency = math.inf if incremental_cost == 0 else delta_utility / incremental_cost
    return _Increment(
        client_id=client_id,
        next_index=index + 1,
        resulting_alpha=next_alpha,
        delta_utility=delta_utility,
        incremental_cost=incremental_cost,
        efficiency=efficiency,
    )


def _validate_inputs(
    utility_curves: Mapping[ClientId, ClientUtilityCurve],
    weight: Mapping[ClientId, float],
    budget: float,
) -> None:
    if not utility_curves:
        raise ValueError("allocate_greedy requires at least one client")
    if utility_curves.keys() != weight.keys():
        raise ValueError("utility_curves and weight must share the same client set")
    if budget < 0:
        raise ValueError(f"budget must be non-negative, got {budget}")

    shared_grid = next(iter(utility_curves.values())).alpha_grid
    for curve in utility_curves.values():
        if curve.alpha_grid != shared_grid:
            raise ValueError("all clients must share the same candidate target-rate grid")


def allocate_greedy(
    utility_curves: Mapping[ClientId, ClientUtilityCurve],
    weight: Mapping[ClientId, float],
    budget: float,
    alpha_max: float,
) -> Allocation:
    _validate_inputs(utility_curves, weight, budget)

    current_index: dict[ClientId, int] = dict.fromkeys(utility_curves, 0)
    remaining_budget = budget

    while True:
        increments = [
            increment
            for client_id, curve in utility_curves.items()
            if (
                increment := _feasible_increment(
                    client_id,
                    curve,
                    current_index[client_id],
                    weight[client_id],
                    remaining_budget,
                    alpha_max,
                )
            )
            is not None
        ]

        if not increments:
            break

        chosen = _best_increment(increments)
        current_index[chosen.client_id] = chosen.next_index
        remaining_budget -= chosen.incremental_cost

    decisions = {
        client_id: AllocationDecision(
            client_id=client_id,
            alpha_selected=utility_curves[client_id].alpha_grid[index],
        )
        for client_id, index in current_index.items()
    }
    return Allocation(policy=AllocationPolicy.GREEDY, decisions=decisions)
