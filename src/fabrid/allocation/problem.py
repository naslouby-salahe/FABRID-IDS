from __future__ import annotations

from dataclasses import dataclass

from fabrid.allocation.contracts import (
    Allocation,
    AllocationDecision,
    FederationWeights,
)
from fabrid.allocation.frontier import (
    FallbackClientFrontier,
    FederationFrontier,
    FederationFrontierInputs,
    build_federation_frontier,
)
from fabrid.domain.enums import AllocationPolicy
from fabrid.domain.identifiers import ClientId
from fabrid.domain.population import ClientPopulation
from fabrid.domain.values import FalsePositiveBudget, TargetFalsePositiveRate
from fabrid.protocol.models import UtilityEligibility

_BUDGET_TOLERANCE = 1.0e-12


@dataclass(frozen=True, slots=True)
class FallbackDecisions:
    decisions: tuple[AllocationDecision, ...]

    def target_for(self, client_id: ClientId) -> TargetFalsePositiveRate | None:
        for decision in self.decisions:
            if decision.client_id == client_id:
                return decision.target_rate
        return None


@dataclass(frozen=True, slots=True)
class AllocationProblem:
    inputs: FederationFrontierInputs
    frontier: FederationFrontier
    weights: FederationWeights
    fallback: FallbackDecisions
    remaining_budget: FalsePositiveBudget


def _fallback_decisions(
    frontier: FederationFrontier,
    budget: FalsePositiveBudget,
    maximum_target_rate: TargetFalsePositiveRate,
) -> FallbackDecisions:
    target_rate = TargetFalsePositiveRate(
        min(budget.value, maximum_target_rate.value)
    )
    return FallbackDecisions(
        tuple(
            AllocationDecision(client_id=client.client_id, target_rate=target_rate)
            for client in frontier.clients
            if isinstance(client, FallbackClientFrontier)
        )
    )


def _remaining_budget(
    budget: FalsePositiveBudget,
    fallback: FallbackDecisions,
    weights: FederationWeights,
) -> FalsePositiveBudget:
    reserved = sum(
        weights.for_client(decision.client_id).value * decision.target_rate.value
        for decision in fallback.decisions
    )
    if reserved > budget.value + _BUDGET_TOLERANCE:
        raise ValueError("fallback reservation exceeds federation budget")
    return FalsePositiveBudget(max(0.0, budget.value - reserved))


def build_allocation_problem(
    inputs: FederationFrontierInputs,
    weights: FederationWeights,
    budget: FalsePositiveBudget,
    eligibility: UtilityEligibility,
    maximum_target_rate: TargetFalsePositiveRate,
) -> AllocationProblem:
    input_clients = {client.client_id for client in inputs.clients}
    weight_clients = {client.client_id for client in weights.clients}
    if input_clients != weight_clients:
        raise ValueError("allocation inputs and federation weights must share clients")
    frontier = build_federation_frontier(inputs, eligibility)
    fallback = _fallback_decisions(frontier, budget, maximum_target_rate)
    return AllocationProblem(
        inputs=inputs,
        frontier=frontier,
        weights=weights,
        fallback=fallback,
        remaining_budget=_remaining_budget(budget, fallback, weights),
    )


def merge_full_allocation(
    policy: AllocationPolicy,
    population: ClientPopulation,
    fallback: FallbackDecisions,
    eligible: Allocation | None,
) -> Allocation:
    decisions: list[AllocationDecision] = []
    for client_id in population.clients:
        fallback_target = fallback.target_for(client_id)
        if fallback_target is not None:
            decisions.append(AllocationDecision(client_id, fallback_target))
        elif eligible is not None:
            decisions.append(eligible.decision(client_id))
        else:
            raise ValueError(
                f"client {client_id.value} has neither fallback nor eligible allocation"
            )
    return Allocation(policy=policy, decisions=tuple(decisions))
