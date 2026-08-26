from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum

import numpy as np

from fabrid.config import (
    TIGHT_TOLERANCE,
    AllocationPolicy,
    AttackSplit,
    AttackSubtypeId,
    BenignSplit,
    ClientId,
    ClientWeight,
    DetectionUtility,
    FalsePositiveBudget,
    NonNegativeFloat,
    Probability,
    RowCount,
    TargetFalsePositiveRate,
    TruePositiveRate,
    UtilityEligibilityConfig,
    WeightGamma,
)
from fabrid.datasets.registry import ClientPopulation
from fabrid.detector.calibration import sorted_order_statistic_threshold
from fabrid.detector.scoring import ScorePartitionArtifact


class EligibilityStatus(StrEnum):
    ELIGIBLE = "eligible"
    FALLBACK = "fallback"


_NO_ELIGIBLE_CLIENTS = "allocation problem has no eligible clients"


@dataclass(frozen=True, slots=True)
class ClientUtilityPoint:
    target_rate: TargetFalsePositiveRate
    utility: DetectionUtility
    utility_variance: NonNegativeFloat

    def __post_init__(self) -> None:
        if self.target_rate < 0.0 or self.target_rate > 1.0:
            raise ValueError(f"target rate must be in [0, 1], got {self.target_rate}")
        if self.utility < 0.0 or self.utility > 1.0:
            raise ValueError(f"utility must be in [0, 1], got {self.utility}")
        if self.utility_variance < 0.0 or self.utility_variance > 1.0:
            raise ValueError(f"utility variance must be in [0, 1], got {self.utility_variance}")


@dataclass(frozen=True, slots=True)
class ClientUtilityCurve:
    client_id: ClientId
    points: tuple[ClientUtilityPoint, ...]

    def __post_init__(self) -> None:
        if not self.points:
            raise ValueError("client utility curve must contain at least one point")
        rates = tuple(point.target_rate for point in self.points)
        if rates != tuple(sorted(rates)):
            raise ValueError("client utility curve rates must be sorted")
        if len(set(rates)) != len(rates):
            raise ValueError("client utility curve rates must be unique")
        if abs(rates[0]) > TIGHT_TOLERANCE:
            raise ValueError("client utility curve must start at zero target rate")


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
        raise KeyError(client_id)


@dataclass(frozen=True, slots=True)
class ClientBudgetWeight:
    client_id: ClientId
    weight: ClientWeight

    def __post_init__(self) -> None:
        if self.weight < 0.0 or self.weight > 1.0:
            raise ValueError(f"client weight must be in [0, 1], got {self.weight}")


@dataclass(frozen=True, slots=True)
class AllocationWeights:
    clients: tuple[ClientBudgetWeight, ...]

    def __post_init__(self) -> None:
        if not self.clients:
            raise ValueError("allocation weights require at least one client")
        client_ids = tuple(client.client_id for client in self.clients)
        if len(set(client_ids)) != len(client_ids):
            raise ValueError("allocation weights contain duplicate clients")
        total = sum(client.weight for client in self.clients)
        if total > 1.0 + TIGHT_TOLERANCE:
            raise ValueError(f"allocation weights may not sum above one, got {total}")

    def for_client(self, client_id: ClientId) -> ClientWeight:
        for client in self.clients:
            if client.client_id == client_id:
                return client.weight
        raise KeyError(client_id)


@dataclass(frozen=True, slots=True)
class FederationWeights:
    allocation_weights: AllocationWeights

    def __post_init__(self) -> None:
        total = sum(client.weight for client in self.allocation_weights.clients)
        if abs(total - 1.0) > TIGHT_TOLERANCE:
            raise ValueError(f"federation weights must sum to one, got {total}")

    @property
    def clients(self) -> tuple[ClientBudgetWeight, ...]:
        return self.allocation_weights.clients

    def for_client(self, client_id: ClientId) -> ClientWeight:
        return self.allocation_weights.for_client(client_id)

    def subset(self, population: ClientPopulation) -> AllocationWeights:
        return AllocationWeights(
            tuple(
                ClientBudgetWeight(client_id, self.for_client(client_id))
                for client_id in population.clients
            )
        )


@dataclass(frozen=True, slots=True)
class AllocationDecision:
    client_id: ClientId
    target_rate: TargetFalsePositiveRate

    def __post_init__(self) -> None:
        if self.target_rate < 0.0 or self.target_rate > 1.0:
            raise ValueError(f"target rate must be in [0, 1], got {self.target_rate}")


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
        raise KeyError(client_id)

    def total_weighted_cost(self, weights: AllocationWeights) -> FalsePositiveBudget:
        return sum(
            weights.for_client(decision.client_id) * decision.target_rate
            for decision in self.decisions
        )

    def budget_feasibility(
        self,
        weights: AllocationWeights,
        budget: FalsePositiveBudget,
    ) -> bool:
        return self.total_weighted_cost(weights) <= budget + TIGHT_TOLERANCE


@dataclass(frozen=True, slots=True)
class FallbackDecisions:
    decisions: tuple[AllocationDecision, ...]

    def target_for(self, client_id: ClientId) -> TargetFalsePositiveRate | None:
        for decision in self.decisions:
            if decision.client_id == client_id:
                return decision.target_rate
        return None


@dataclass(frozen=True, slots=True)
class SubtypeConfusionCounts:
    true_positive: RowCount
    false_negative: RowCount

    def __post_init__(self) -> None:
        if self.true_positive < 0 or self.false_negative < 0:
            raise ValueError("subtype confusion counts must be non-negative")
        if self.true_positive + self.false_negative == 0:
            raise ValueError("subtype confusion counts must cover at least one attack row")

    @property
    def attack_rows(self) -> RowCount:
        return self.true_positive + self.false_negative

    def true_positive_rate(self) -> TruePositiveRate:
        return self.true_positive / self.attack_rows


@dataclass(frozen=True, slots=True)
class SubtypeConfusion:
    subtype: AttackSubtypeId
    counts: SubtypeConfusionCounts


@dataclass(frozen=True, slots=True)
class CandidateConfusions:
    target_rate: TargetFalsePositiveRate
    subtypes: tuple[SubtypeConfusion, ...]

    def __post_init__(self) -> None:
        if self.target_rate < 0.0 or self.target_rate > 1.0:
            raise ValueError(f"candidate target rate must be in [0, 1], got {self.target_rate}")
        subtype_ids = tuple(subtype.subtype for subtype in self.subtypes)
        if len(set(subtype_ids)) != len(subtype_ids):
            raise ValueError("candidate confusion data contains duplicate subtypes")

    def for_subtype(self, subtype: AttackSubtypeId) -> SubtypeConfusionCounts:
        for item in self.subtypes:
            if item.subtype == subtype:
                return item.counts
        raise KeyError(subtype)


@dataclass(frozen=True, slots=True)
class AttackSubtypeSelection:
    subtypes: tuple[AttackSubtypeId, ...]

    def __post_init__(self) -> None:
        if len(set(self.subtypes)) != len(self.subtypes):
            raise ValueError("attack subtype selection contains duplicates")

    def contains(self, subtype: AttackSubtypeId) -> bool:
        return subtype in self.subtypes


@dataclass(frozen=True, slots=True)
class ClientFrontierInputs:
    client_id: ClientId
    benign_frontier_scores: np.ndarray
    frontier_row_count: RowCount
    candidates: tuple[CandidateConfusions, ...]

    def __post_init__(self) -> None:
        if not self.candidates:
            raise ValueError("client frontier inputs require candidate data")
        if self.frontier_row_count <= 0:
            raise ValueError("client frontier row count must be positive")
        target_rates = tuple(candidate.target_rate for candidate in self.candidates)
        if target_rates != tuple(sorted(target_rates)):
            raise ValueError("frontier candidates must be sorted by target rate")
        if len(set(target_rates)) != len(target_rates):
            raise ValueError("frontier target rates must be unique")
        reference = self.candidates[0]
        reference_subtypes = tuple(subtype.subtype for subtype in reference.subtypes)
        reference_rows = tuple(subtype.counts.attack_rows for subtype in reference.subtypes)
        for candidate in self.candidates[1:]:
            candidate_subtypes = tuple(subtype.subtype for subtype in candidate.subtypes)
            candidate_rows = tuple(subtype.counts.attack_rows for subtype in candidate.subtypes)
            if candidate_subtypes != reference_subtypes or candidate_rows != reference_rows:
                raise ValueError(
                    "all frontier candidates must describe the same validation population"
                )


@dataclass(frozen=True, slots=True)
class FederationFrontierInputs:
    clients: tuple[ClientFrontierInputs, ...]

    def __post_init__(self) -> None:
        if not self.clients:
            raise ValueError("federation frontier inputs require at least one client")
        client_ids = tuple(client.client_id for client in self.clients)
        if len(set(client_ids)) != len(client_ids):
            raise ValueError("federation frontier inputs contain duplicate clients")


@dataclass(frozen=True, slots=True)
class EligibleClientFrontier:
    client_id: ClientId
    status: EligibilityStatus
    utility_curve: ClientUtilityCurve

    def __post_init__(self) -> None:
        if self.status is not EligibilityStatus.ELIGIBLE:
            raise ValueError("eligible client frontier must have ELIGIBLE status")


@dataclass(frozen=True, slots=True)
class FallbackClientFrontier:
    client_id: ClientId
    status: EligibilityStatus

    def __post_init__(self) -> None:
        if self.status is not EligibilityStatus.FALLBACK:
            raise ValueError("fallback client frontier must have FALLBACK status")


@dataclass(frozen=True, slots=True)
class FederationFrontier:
    clients: tuple[EligibleClientFrontier | FallbackClientFrontier, ...]
    fallback_rate: Probability

    def eligible_curves(self) -> ClientUtilityCurves | None:
        curves = tuple(
            client.utility_curve
            for client in self.clients
            if isinstance(client, EligibleClientFrontier)
        )
        return None if not curves else ClientUtilityCurves(curves)

    def eligible_population(self) -> ClientPopulation | None:
        clients = tuple(
            client.client_id
            for client in self.clients
            if isinstance(client, EligibleClientFrontier)
        )
        return None if not clients else ClientPopulation(clients)


@dataclass(frozen=True, slots=True)
class FrontierScoreArtifacts:
    benign_frontier: ScorePartitionArtifact
    attack_validation: ScorePartitionArtifact

    def __post_init__(self) -> None:
        if self.benign_frontier.split is not BenignSplit.FRONTIER:
            raise ValueError("frontier inputs require a BENIGN_FRONTIER artifact")
        if self.attack_validation.split is not AttackSplit.VALIDATION:
            raise ValueError("frontier inputs require an ATTACK_VALIDATION artifact")
        if self.benign_frontier.coordinate != self.attack_validation.coordinate:
            raise ValueError("frontier score artifacts must share one score coordinate")

    @property
    def client_id(self) -> ClientId:
        return self.benign_frontier.coordinate.client_id


@dataclass(frozen=True, slots=True)
class AllocationProblem:
    inputs: FederationFrontierInputs
    frontier: FederationFrontier
    weights: FederationWeights
    fallback: FallbackDecisions
    budget: FalsePositiveBudget
    remaining_budget: FalsePositiveBudget
    maximum_target_rate: TargetFalsePositiveRate

    @property
    def population(self) -> ClientPopulation:
        return ClientPopulation(tuple(client.client_id for client in self.inputs.clients))

    def require_eligible_curves(self) -> ClientUtilityCurves:
        curves = self.frontier.eligible_curves()
        if curves is None:
            raise ValueError(_NO_ELIGIBLE_CLIENTS)
        return curves

    def eligible_weights(self) -> AllocationWeights:
        population = self.frontier.eligible_population()
        if population is None:
            raise ValueError(_NO_ELIGIBLE_CLIENTS)
        return self.weights.subset(population)

    def frontier_row_counts(self) -> tuple[ClientRowCount, ...]:
        return tuple(
            ClientRowCount(
                client_id=client.client_id,
                row_count=client.frontier_row_count,
            )
            for client in self.inputs.clients
        )

    def with_eligible_curves(self, curves: ClientUtilityCurves) -> AllocationProblem:
        population = self.frontier.eligible_population()
        if population is None:
            raise ValueError(_NO_ELIGIBLE_CLIENTS)
        curve_ids = {curve.client_id for curve in curves.clients}
        if curve_ids != set(population.clients):
            raise ValueError("replacement utility curves must cover exactly the eligible clients")
        replaced: list[EligibleClientFrontier | FallbackClientFrontier] = []
        for client in self.frontier.clients:
            if isinstance(client, EligibleClientFrontier):
                new_curve = curves.for_client(client.client_id)
                previous_rates = tuple(point.target_rate for point in client.utility_curve.points)
                next_rates = tuple(point.target_rate for point in new_curve.points)
                if previous_rates != next_rates:
                    raise ValueError(
                        "replacement utility curves must keep the same target-rate grid"
                    )
                replaced.append(replace(client, utility_curve=new_curve))
            else:
                replaced.append(client)
        return AllocationProblem(
            inputs=self.inputs,
            frontier=replace(self.frontier, clients=tuple(replaced)),
            weights=self.weights,
            fallback=self.fallback,
            budget=self.budget,
            remaining_budget=self.remaining_budget,
            maximum_target_rate=self.maximum_target_rate,
        )


@dataclass(frozen=True, slots=True)
class SubtypeRowCount:
    subtype: AttackSubtypeId
    row_count: RowCount


def _subtype_rows(inputs: ClientFrontierInputs) -> tuple[SubtypeRowCount, ...]:
    return tuple(
        SubtypeRowCount(subtype=item.subtype, row_count=item.counts.attack_rows)
        for item in inputs.candidates[0].subtypes
    )


def build_client_frontier_inputs(
    frontier_artifacts: FrontierScoreArtifacts,
    alpha_grid: tuple[TargetFalsePositiveRate, ...],
    frontier_row_count: RowCount,
) -> ClientFrontierInputs:
    benign_frontier_scores = frontier_artifacts.benign_frontier.score_values()
    if benign_frontier_scores.size == 0:
        raise ValueError("frontier inputs require benign frontier scores")
    sorted_benign_scores = np.sort(benign_frontier_scores)
    subtype_items = frontier_artifacts.attack_validation.subtype_scores()
    if not subtype_items:
        raise ValueError("frontier inputs require attack-validation subtypes")
    sorted_subtype_scores = tuple((item.subtype, np.sort(item.scores)) for item in subtype_items)
    if any(scores.size == 0 for _, scores in sorted_subtype_scores):
        raise ValueError("frontier inputs contain an empty attack subtype")
    candidates: list[CandidateConfusions] = []
    for target_rate in alpha_grid:
        threshold = sorted_order_statistic_threshold(sorted_benign_scores, target_rate)
        subtype_confusions: list[SubtypeConfusion] = []
        for subtype, sorted_scores in sorted_subtype_scores:
            true_positive = sorted_scores.size - int(
                np.searchsorted(sorted_scores, threshold, side="right")
            )
            subtype_confusions.append(
                SubtypeConfusion(
                    subtype=subtype,
                    counts=SubtypeConfusionCounts(
                        true_positive=true_positive,
                        false_negative=int(sorted_scores.size) - true_positive,
                    ),
                )
            )
        candidates.append(
            CandidateConfusions(
                target_rate=target_rate,
                subtypes=tuple(subtype_confusions),
            )
        )
    return ClientFrontierInputs(
        client_id=frontier_artifacts.client_id,
        benign_frontier_scores=benign_frontier_scores,
        frontier_row_count=frontier_row_count,
        candidates=tuple(candidates),
    )


def eligible_subtypes(
    inputs: ClientFrontierInputs,
    guardrails: UtilityEligibilityConfig,
) -> AttackSubtypeSelection:
    return AttackSubtypeSelection(
        tuple(
            item.subtype
            for item in _subtype_rows(inputs)
            if item.row_count >= guardrails.minimum_rows_per_subtype
        )
    )


def client_eligibility(
    inputs: ClientFrontierInputs,
    guardrails: UtilityEligibilityConfig,
) -> EligibilityStatus:
    total_rows = sum(item.row_count for item in _subtype_rows(inputs))
    selection = eligible_subtypes(inputs, guardrails)
    if (
        total_rows < guardrails.minimum_attack_validation_rows
        or len(selection.subtypes) < guardrails.minimum_eligible_subtypes
    ):
        return EligibilityStatus.FALLBACK
    return EligibilityStatus.ELIGIBLE


def _client_utility(
    candidate: CandidateConfusions,
    selection: AttackSubtypeSelection,
) -> DetectionUtility:
    rates = tuple(
        item.counts.true_positive_rate()
        for item in candidate.subtypes
        if selection.contains(item.subtype)
    )
    if not rates:
        raise ValueError("client utility requires at least one eligible subtype")
    return sum(rates) / len(rates)


def client_utility_variance(
    candidate: CandidateConfusions,
    selection: AttackSubtypeSelection,
) -> NonNegativeFloat:
    counts = tuple(item.counts for item in candidate.subtypes if selection.contains(item.subtype))
    if not counts:
        raise ValueError("client utility variance requires at least one eligible subtype")
    subtype_count = len(counts)
    variance = sum(
        rate * (1.0 - rate) / item.attack_rows
        for item in counts
        for rate in (item.true_positive_rate(),)
    ) / (subtype_count * subtype_count)
    return variance


def build_client_frontier(
    inputs: ClientFrontierInputs,
    guardrails: UtilityEligibilityConfig,
) -> EligibleClientFrontier | FallbackClientFrontier:
    status = client_eligibility(inputs, guardrails)
    if status is EligibilityStatus.FALLBACK:
        return FallbackClientFrontier(inputs.client_id, status)
    selection = eligible_subtypes(inputs, guardrails)
    curve = ClientUtilityCurve(
        client_id=inputs.client_id,
        points=tuple(
            ClientUtilityPoint(
                target_rate=candidate.target_rate,
                utility=_client_utility(candidate, selection),
                utility_variance=client_utility_variance(candidate, selection),
            )
            for candidate in inputs.candidates
        ),
    )
    return EligibleClientFrontier(
        client_id=inputs.client_id,
        status=status,
        utility_curve=curve,
    )


def build_federation_frontier(
    inputs: FederationFrontierInputs,
    guardrails: UtilityEligibilityConfig,
) -> FederationFrontier:
    clients = tuple(build_client_frontier(client, guardrails) for client in inputs.clients)
    fallback_count = sum(1 for client in clients if isinstance(client, FallbackClientFrontier))
    return FederationFrontier(
        clients=clients,
        fallback_rate=fallback_count / len(clients),
    )


def _fallback_decisions(
    frontier: FederationFrontier,
    budget: FalsePositiveBudget,
    maximum_target_rate: TargetFalsePositiveRate,
) -> FallbackDecisions:
    target_rate = min(budget, maximum_target_rate)
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
        weights.for_client(decision.client_id) * decision.target_rate
        for decision in fallback.decisions
    )
    if reserved > budget + TIGHT_TOLERANCE:
        raise ValueError("fallback reservation exceeds federation budget")
    return max(0.0, budget - reserved)


def build_allocation_problem(
    inputs: FederationFrontierInputs,
    weights: FederationWeights,
    budget: FalsePositiveBudget,
    eligibility: UtilityEligibilityConfig,
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
        budget=budget,
        remaining_budget=_remaining_budget(budget, fallback, weights),
        maximum_target_rate=maximum_target_rate,
    )


def merge_full_allocation(
    policy: AllocationPolicy,
    problem: AllocationProblem,
    eligible: Allocation | None,
) -> Allocation:
    decisions: list[AllocationDecision] = []
    for client_id in problem.population.clients:
        fallback_target = problem.fallback.target_for(client_id)
        if fallback_target is not None:
            decisions.append(AllocationDecision(client_id, fallback_target))
        elif eligible is not None:
            decisions.append(eligible.decision(client_id))
        else:
            raise ValueError(f"client {client_id} has neither fallback nor eligible allocation")
    return Allocation(policy=policy, decisions=tuple(decisions))


def equal_client_weights(population: ClientPopulation) -> FederationWeights:
    weight = 1.0 / population.size
    return FederationWeights(
        AllocationWeights(
            tuple(
                ClientBudgetWeight(client_id=client_id, weight=weight)
                for client_id in population.clients
            )
        )
    )


@dataclass(frozen=True, slots=True)
class ClientRowCount:
    client_id: ClientId
    row_count: RowCount


def dataset_count_weights(
    population: ClientPopulation,
    counts: tuple[ClientRowCount, ...],
) -> FederationWeights:
    count_by_client = {item.client_id: item.row_count for item in counts}
    if set(count_by_client) != set(population.clients):
        raise ValueError("dataset-count weights must cover exactly the federation population")
    if len(count_by_client) != len(counts):
        raise ValueError("dataset-count weights contain duplicate clients")
    if any(count <= 0 for count in count_by_client.values()):
        raise ValueError("dataset-count proxy requires strictly positive client counts")
    total = sum(count_by_client.values())
    return FederationWeights(
        AllocationWeights(
            tuple(
                ClientBudgetWeight(
                    client_id=client_id,
                    weight=count_by_client[client_id] / total,
                )
                for client_id in population.clients
            )
        )
    )


def weight_gamma_transform(
    weights: FederationWeights,
    gamma: WeightGamma,
) -> FederationWeights:
    if any(client.weight <= 0.0 for client in weights.clients):
        raise ValueError("gamma reweighting requires strictly positive weights")
    powered = tuple(client.weight**gamma for client in weights.clients)
    total = sum(powered)
    return FederationWeights(
        AllocationWeights(
            tuple(
                ClientBudgetWeight(client_id=client.client_id, weight=value / total)
                for client, value in zip(weights.clients, powered, strict=True)
            )
        )
    )
