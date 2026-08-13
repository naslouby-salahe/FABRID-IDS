from __future__ import annotations

import math
from dataclasses import dataclass

from fabrid.allocation.contracts import (
    Allocation,
    AllocationDecision,
    ClientUtilityCurve,
    ClientUtilityCurves,
    FederationWeights,
)
from fabrid.domain.enums import AllocationPolicy
from fabrid.domain.identifiers import ClientId
from fabrid.domain.values import FalsePositiveBudget, TargetFalsePositiveRate

_BUDGET_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class _Position:
    client_id: ClientId
    point_index: int


@dataclass(frozen=True, slots=True)
class _Increment:
    client_id: ClientId
    next_index: int
    resulting_rate: TargetFalsePositiveRate
    delta_utility: float
    incremental_cost: float
    efficiency: float


def _best_increment(increments: tuple[_Increment, ...]) -> _Increment:
    return min(
        increments,
        key=lambda increment: (
            -increment.efficiency,
            -increment.delta_utility,
            increment.incremental_cost,
            increment.client_id.value,
            increment.resulting_rate.value,
        ),
    )


def _feasible_increment(
    curve: ClientUtilityCurve,
    index: int,
    weights: FederationWeights,
    remaining_budget: FalsePositiveBudget,
    maximum_target_rate: TargetFalsePositiveRate,
) -> _Increment | None:
    if index + 1 >= len(curve.points):
        return None

    current_point = curve.points[index]
    next_point = curve.points[index + 1]
    if next_point.target_rate.value > maximum_target_rate.value + _BUDGET_TOLERANCE:
        return None

    delta_rate = next_point.target_rate.value - current_point.target_rate.value
    incremental_cost = weights.for_client(curve.client_id).value * delta_rate
    if incremental_cost > remaining_budget.value + _BUDGET_TOLERANCE:
        return None

    delta_utility = next_point.utility.value - current_point.utility.value
    efficiency = (
        math.inf if incremental_cost == 0.0 else delta_utility / incremental_cost
    )
    return _Increment(
        client_id=curve.client_id,
        next_index=index + 1,
        resulting_rate=next_point.target_rate,
        delta_utility=delta_utility,
        incremental_cost=incremental_cost,
        efficiency=efficiency,
    )


def allocate_greedy(
    utility_curves: ClientUtilityCurves,
    weights: FederationWeights,
    budget: FalsePositiveBudget,
    maximum_target_rate: TargetFalsePositiveRate,
) -> Allocation:
    curve_clients = {curve.client_id for curve in utility_curves.clients}
    weight_clients = {client.client_id for client in weights.clients}
    if curve_clients != weight_clients:
        raise ValueError("utility curves and federation weights must share clients")

    positions = [
        _Position(client_id=curve.client_id, point_index=0)
        for curve in utility_curves.clients
    ]
    remaining_budget = budget

    while True:
        increments = tuple(
            increment
            for position in positions
            if (
                increment := _feasible_increment(
                    curve=utility_curves.for_client(position.client_id),
                    index=position.point_index,
                    weights=weights,
                    remaining_budget=remaining_budget,
                    maximum_target_rate=maximum_target_rate,
                )
            )
            is not None
        )
        if not increments:
            break

        chosen = _best_increment(increments)
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
        remaining_budget = FalsePositiveBudget(
            max(0.0, remaining_budget.value - chosen.incremental_cost)
        )

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
