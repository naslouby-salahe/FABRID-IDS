from __future__ import annotations

import math
from dataclasses import dataclass

from fabrid.allocation.contracts import (
    Allocation,
    AllocationDecision,
    AllocationWeights,
    ClientUtilityCurve,
    ClientUtilityCurves,
)
from fabrid.domain.enums import AllocationPolicy
from fabrid.domain.identifiers import ClientId
from fabrid.domain.values import (
    CandidateIndex,
    FalsePositiveBudget,
    IncrementalBudgetCost,
    MarginalEfficiency,
    TargetFalsePositiveRate,
    UtilityDifference,
)

_BUDGET_TOLERANCE = 1e-12


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
            -increment.efficiency.value,
            -increment.delta_utility.value,
            increment.incremental_cost.value,
            increment.client_id.value,
            increment.resulting_rate.value,
        ),
    )


def _feasible_increment(
    curve: ClientUtilityCurve,
    index: CandidateIndex,
    weights: AllocationWeights,
    remaining_budget: FalsePositiveBudget,
    maximum_target_rate: TargetFalsePositiveRate,
) -> _Increment | None:
    next_index = CandidateIndex(index.value + 1)
    if next_index.value >= len(curve.points):
        return None

    current_point = curve.points[index.value]
    next_point = curve.points[next_index.value]
    if next_point.target_rate.value > maximum_target_rate.value + _BUDGET_TOLERANCE:
        return None

    delta_rate = next_point.target_rate.value - current_point.target_rate.value
    incremental_cost = IncrementalBudgetCost(
        weights.for_client(curve.client_id).value * delta_rate
    )
    if incremental_cost.value > remaining_budget.value + _BUDGET_TOLERANCE:
        return None

    delta_utility = UtilityDifference(next_point.utility.value - current_point.utility.value)
    efficiency = MarginalEfficiency(
        math.inf
        if incremental_cost.value == 0.0
        else delta_utility.value / incremental_cost.value
    )
    return _Increment(
        client_id=curve.client_id,
        next_index=next_index,
        resulting_rate=next_point.target_rate,
        delta_utility=delta_utility,
        incremental_cost=incremental_cost,
        efficiency=efficiency,
    )


def allocate_greedy(
    utility_curves: ClientUtilityCurves,
    weights: AllocationWeights,
    budget: FalsePositiveBudget,
    maximum_target_rate: TargetFalsePositiveRate,
) -> Allocation:
    curve_clients = {curve.client_id for curve in utility_curves.clients}
    weight_clients = {client.client_id for client in weights.clients}
    if curve_clients != weight_clients:
        raise ValueError("utility curves and allocation weights must share clients")

    positions = [
        _Position(client_id=curve.client_id, point_index=CandidateIndex(0))
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
            max(0.0, remaining_budget.value - chosen.incremental_cost.value)
        )

    return Allocation(
        policy=AllocationPolicy.GREEDY,
        decisions=tuple(
            AllocationDecision(
                client_id=position.client_id,
                target_rate=utility_curves.for_client(position.client_id)
                .points[position.point_index.value]
                .target_rate,
            )
            for position in positions
        ),
    )
