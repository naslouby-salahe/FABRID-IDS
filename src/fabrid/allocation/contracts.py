from __future__ import annotations

from dataclasses import dataclass

from fabrid.domain.enums import AllocationPolicy
from fabrid.domain.identifiers import ClientId
from fabrid.domain.values import (
    ClientWeight,
    DetectionUtility,
    FalsePositiveBudget,
    TargetFalsePositiveRate,
)

_FEASIBILITY_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class ClientUtilityPoint:
    target_rate: TargetFalsePositiveRate
    utility: DetectionUtility


@dataclass(frozen=True, slots=True)
class ClientUtilityCurve:
    client_id: ClientId
    points: tuple[ClientUtilityPoint, ...]

    def __post_init__(self) -> None:
        if not self.points:
            raise ValueError("client utility curve must contain at least one point")
        rates = tuple(point.target_rate.value for point in self.points)
        if rates != tuple(sorted(rates)):
            raise ValueError("client utility curve rates must be sorted")
        if len(set(rates)) != len(rates):
            raise ValueError("client utility curve rates must be unique")
        if rates[0] != 0.0:
            raise ValueError("client utility curve must start at zero target rate")

    def point(self, target_rate: TargetFalsePositiveRate) -> ClientUtilityPoint:
        for point in self.points:
            if point.target_rate == target_rate:
                return point
        raise KeyError(target_rate.value)


@dataclass(frozen=True, slots=True)
class ClientUtilityCurves:
    clients: tuple[ClientUtilityCurve, ...]

    def __post_init__(self) -> None:
        if not self.clients:
            raise ValueError("client utility curves must not be empty")
        client_ids = tuple(curve.client_id for curve in self.clients)
        if len(set(client_ids)) != len(client_ids):
            raise ValueError("client utility curves contain duplicate clients")
        shared_rates = tuple(point.target_rate for point in self.clients[0].points)
        for curve in self.clients[1:]:
            if tuple(point.target_rate for point in curve.points) != shared_rates:
                raise ValueError("all clients must share the same target-rate grid")

    def for_client(self, client_id: ClientId) -> ClientUtilityCurve:
        for curve in self.clients:
            if curve.client_id == client_id:
                return curve
        raise KeyError(client_id.value)


@dataclass(frozen=True, slots=True)
class ClientBudgetWeight:
    client_id: ClientId
    weight: ClientWeight


@dataclass(frozen=True, slots=True)
class FederationWeights:
    clients: tuple[ClientBudgetWeight, ...]

    def __post_init__(self) -> None:
        if not self.clients:
            raise ValueError("federation weights require at least one client")
        client_ids = tuple(client.client_id for client in self.clients)
        if len(set(client_ids)) != len(client_ids):
            raise ValueError("federation weights contain duplicate clients")
        total = sum(client.weight.value for client in self.clients)
        if abs(total - 1.0) > _FEASIBILITY_TOLERANCE:
            raise ValueError(f"federation weights must sum to one, got {total}")

    def for_client(self, client_id: ClientId) -> ClientWeight:
        for client in self.clients:
            if client.client_id == client_id:
                return client.weight
        raise KeyError(client_id.value)


@dataclass(frozen=True, slots=True)
class AllocationDecision:
    client_id: ClientId
    target_rate: TargetFalsePositiveRate


@dataclass(frozen=True, slots=True)
class Allocation:
    policy: AllocationPolicy
    decisions: tuple[AllocationDecision, ...]

    def __post_init__(self) -> None:
        if not self.decisions:
            raise ValueError("allocation requires at least one client decision")
        client_ids = tuple(decision.client_id for decision in self.decisions)
        if len(set(client_ids)) != len(client_ids):
            raise ValueError("allocation contains duplicate client decisions")

    def decision(self, client_id: ClientId) -> AllocationDecision:
        for decision in self.decisions:
            if decision.client_id == client_id:
                return decision
        raise KeyError(client_id.value)

    def total_weighted_cost(self, weights: FederationWeights) -> FalsePositiveBudget:
        cost = sum(
            weights.for_client(decision.client_id).value * decision.target_rate.value
            for decision in self.decisions
        )
        return FalsePositiveBudget(cost)

    def is_budget_feasible(
        self,
        weights: FederationWeights,
        budget: FalsePositiveBudget,
    ) -> bool:
        return (
            self.total_weighted_cost(weights).value
            <= budget.value + _FEASIBILITY_TOLERANCE
        )
