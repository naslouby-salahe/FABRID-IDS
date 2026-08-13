from __future__ import annotations

from dataclasses import dataclass

from fabrid.allocation.contracts import (
    Allocation,
    AllocationDecision,
    AllocationWeights,
    ClientBudgetWeight,
    FederationWeights,
)
from fabrid.allocation.equal_fpr import allocate_equal_fpr
from fabrid.allocation.fabrid_macro import allocate_fabrid_macro
from fabrid.allocation.fabrid_minimax import allocate_fabrid_minimax
from fabrid.allocation.frontier import (
    FallbackClientFrontier,
    FederationFrontier,
    FederationFrontierInputs,
    build_federation_frontier,
)
from fabrid.allocation.frontier_inputs import (
    FrontierScoreArtifacts,
    build_client_frontier_inputs,
)
from fabrid.allocation.greedy import allocate_greedy
from fabrid.allocation.solver import (
    SolverEvidence,
    SolverInvalidError,
    not_applicable_solver_evidence,
)
from fabrid.domain.coordinates import (
    AllocationCoordinate,
    ExperimentCoordinate,
    ScoreCoordinate,
)
from fabrid.domain.enums import (
    AllocationPolicy,
    DatasetId,
    ExperimentId,
    ExperimentVariantId,
    SolverStatus,
    WeightMode,
)
from fabrid.domain.identifiers import CampaignId, ClientId
from fabrid.domain.population import ClientPopulation
from fabrid.domain.values import (
    ClientWeight,
    DetectorSeed,
    FalsePositiveBudget,
    Probability,
    TargetFalsePositiveRate,
)
from fabrid.evaluation.evaluator import (
    ClientEvaluationArtifacts,
    EvaluationProvenance,
    FederationEvaluationArtifacts,
    PolicyEvaluationResult,
    evaluate_allocation,
)
from fabrid.evaluation.results import (
    ClientResultRecord,
    ExcludedPolicyEvaluation,
    PolicyEvaluation,
    SeedBudgetEvaluation,
)
from fabrid.pipeline.context import PipelinePaths
from fabrid.pipeline.score_loading import load_client_scores
from fabrid.protocol.models import BudgetLevel, FabridProtocol

_BUDGET_TOLERANCE = 1.0e-12


@dataclass(frozen=True, slots=True)
class LoadedClientScores:
    client_id: ClientId
    frontier: FrontierScoreArtifacts
    evaluation: ClientEvaluationArtifacts


@dataclass(frozen=True, slots=True)
class LoadedSeedScores:
    clients: tuple[LoadedClientScores, ...]

    def __post_init__(self) -> None:
        if not self.clients:
            raise ValueError("loaded seed scores require at least one client")
        client_ids = tuple(client.client_id for client in self.clients)
        if len(set(client_ids)) != len(client_ids):
            raise ValueError("loaded seed scores contain duplicate clients")

    @property
    def population(self) -> ClientPopulation:
        return ClientPopulation(tuple(client.client_id for client in self.clients))

    @property
    def evaluation_artifacts(self) -> FederationEvaluationArtifacts:
        return FederationEvaluationArtifacts(
            tuple(client.evaluation for client in self.clients)
        )


@dataclass(frozen=True, slots=True)
class FallbackDecisions:
    decisions: tuple[AllocationDecision, ...]

    def target_for(self, client_id: ClientId) -> TargetFalsePositiveRate | None:
        for decision in self.decisions:
            if decision.client_id == client_id:
                return decision.target_rate
        return None


@dataclass(frozen=True, slots=True)
class CompletedPolicyRun:
    allocation: Allocation
    solver: SolverEvidence
    result: PolicyEvaluationResult


@dataclass(frozen=True, slots=True)
class ExcludedPolicyRun:
    result: ExcludedPolicyEvaluation


PolicyRun = CompletedPolicyRun | ExcludedPolicyRun


@dataclass(frozen=True, slots=True)
class SeedBudgetRun:
    evaluation: SeedBudgetEvaluation
    records: tuple[ClientResultRecord, ...]
    policy_runs: tuple[PolicyRun, ...]


def equal_client_weights(population: ClientPopulation) -> FederationWeights:
    weight = ClientWeight(1.0 / population.size.value)
    return FederationWeights(
        AllocationWeights(
            tuple(
                ClientBudgetWeight(client_id=client_id, weight=weight)
                for client_id in population.clients
            )
        )
    )


def load_seed_scores(
    campaign_id: CampaignId,
    detector_seed: DetectorSeed,
    population: ClientPopulation,
    paths: PipelinePaths,
) -> LoadedSeedScores:
    clients: list[LoadedClientScores] = []
    for client_id in population.clients:
        coordinate = ScoreCoordinate(
            dataset_id=DatasetId.NBAIOT,
            detector_seed=detector_seed,
            client_id=client_id,
        )
        scores = load_client_scores(campaign_id, coordinate, paths)
        clients.append(
            LoadedClientScores(
                client_id=client_id,
                frontier=FrontierScoreArtifacts(
                    benign_frontier=scores.benign_frontier,
                    attack_validation=scores.attack_validation,
                ),
                evaluation=ClientEvaluationArtifacts(
                    final_calibration=scores.benign_final_cal,
                    benign_test=scores.benign_test,
                    attack_test=scores.attack_test,
                ),
            )
        )
    return LoadedSeedScores(tuple(clients))


def _build_frontier(
    scores: LoadedSeedScores,
    protocol: FabridProtocol,
) -> FederationFrontier:
    return build_federation_frontier(
        FederationFrontierInputs(
            tuple(
                build_client_frontier_inputs(client.frontier, protocol.alpha_grid)
                for client in scores.clients
            )
        ),
        protocol.utility_eligibility,
    )


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
    full_weights: FederationWeights,
) -> FalsePositiveBudget:
    reserved = sum(
        full_weights.for_client(decision.client_id).value * decision.target_rate.value
        for decision in fallback.decisions
    )
    if reserved > budget.value + _BUDGET_TOLERANCE:
        raise ValueError("fallback reservation exceeds federation budget")
    return FalsePositiveBudget(max(0.0, budget.value - reserved))


def _merge_full_allocation(
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


def _evaluate_policy(
    coordinate: ExperimentCoordinate,
    allocation: Allocation,
    scores: LoadedSeedScores,
    weights: FederationWeights,
    solver: SolverEvidence,
    provenance: EvaluationProvenance,
    fallback_rate: Probability,
) -> CompletedPolicyRun:
    if not allocation.is_budget_feasible(
        weights.allocation_weights,
        coordinate.budget,
    ):
        raise ValueError(
            f"allocation {allocation.policy.value} violates the federation budget"
        )
    result = evaluate_allocation(
        coordinate=AllocationCoordinate(coordinate, allocation.policy),
        allocation=allocation,
        artifacts=scores.evaluation_artifacts,
        weights=weights,
        solver=solver,
        provenance=provenance,
        fallback_rate=fallback_rate,
    )
    return CompletedPolicyRun(
        allocation=allocation,
        solver=solver,
        result=result,
    )


def _run_greedy(
    coordinate: ExperimentCoordinate,
    scores: LoadedSeedScores,
    frontier: FederationFrontier,
    weights: FederationWeights,
    fallback: FallbackDecisions,
    remaining_budget: FalsePositiveBudget,
    protocol: FabridProtocol,
    provenance: EvaluationProvenance,
) -> CompletedPolicyRun:
    eligible_population = frontier.eligible_population()
    eligible_curves = frontier.eligible_curves()
    eligible_allocation = None
    if eligible_population is not None and eligible_curves is not None:
        eligible_allocation = allocate_greedy(
            eligible_curves,
            weights.subset(eligible_population),
            remaining_budget,
            protocol.alpha_grid.maximum,
        )
    full_allocation = _merge_full_allocation(
        AllocationPolicy.GREEDY,
        scores.population,
        fallback,
        eligible_allocation,
    )
    return _evaluate_policy(
        coordinate,
        full_allocation,
        scores,
        weights,
        not_applicable_solver_evidence(),
        provenance,
        frontier.fallback_rate,
    )


def _run_optimized(
    coordinate: ExperimentCoordinate,
    policy: AllocationPolicy,
    scores: LoadedSeedScores,
    frontier: FederationFrontier,
    weights: FederationWeights,
    fallback: FallbackDecisions,
    remaining_budget: FalsePositiveBudget,
    protocol: FabridProtocol,
    provenance: EvaluationProvenance,
) -> PolicyRun:
    eligible_population = frontier.eligible_population()
    eligible_curves = frontier.eligible_curves()
    if eligible_population is None or eligible_curves is None:
        full_fallback = _merge_full_allocation(
            policy,
            scores.population,
            fallback,
            eligible=None,
        )
        return _evaluate_policy(
            coordinate,
            full_fallback,
            scores,
            weights,
            not_applicable_solver_evidence(),
            provenance,
            frontier.fallback_rate,
        )

    eligible_weights = weights.subset(eligible_population)
    try:
        optimized = (
            allocate_fabrid_macro(
                eligible_curves,
                eligible_weights,
                remaining_budget,
                protocol.solver,
            )
            if policy is AllocationPolicy.FABRID_MACRO
            else allocate_fabrid_minimax(
                eligible_curves,
                eligible_weights,
                remaining_budget,
                protocol.solver,
            )
        )
    except SolverInvalidError as error:
        return ExcludedPolicyRun(
            ExcludedPolicyEvaluation(
                policy=policy,
                status=SolverStatus.SOLVER_INVALID,
                reason=error.reason,
            )
        )

    full_allocation = _merge_full_allocation(
        policy,
        scores.population,
        fallback,
        optimized.allocation,
    )
    return _evaluate_policy(
        coordinate,
        full_allocation,
        scores,
        weights,
        optimized.solver,
        provenance,
        frontier.fallback_rate,
    )


def run_seed_budget(
    campaign_id: CampaignId,
    detector_seed: DetectorSeed,
    budget_level: BudgetLevel,
    scores: LoadedSeedScores,
    protocol: FabridProtocol,
    provenance: EvaluationProvenance,
) -> SeedBudgetRun:
    coordinate = ExperimentCoordinate(
        campaign_id=campaign_id,
        experiment_id=ExperimentId.MATCHED_BUDGET,
        variant_id=ExperimentVariantId.PRIMARY,
        dataset_id=DatasetId.NBAIOT,
        detector_seed=detector_seed,
        budget_id=budget_level.budget_id,
        budget=budget_level.value,
        weight_mode=WeightMode.EQUAL_CLIENT,
    )
    weights = equal_client_weights(scores.population)
    frontier = _build_frontier(scores, protocol)
    fallback = _fallback_decisions(
        frontier,
        budget_level.value,
        protocol.alpha_grid.maximum,
    )
    remaining_budget = _remaining_budget(budget_level.value, fallback, weights)

    equal_fpr = _evaluate_policy(
        coordinate,
        allocate_equal_fpr(
            scores.population,
            budget_level.value,
            protocol.alpha_grid.maximum,
        ),
        scores,
        weights,
        not_applicable_solver_evidence(),
        provenance,
        frontier.fallback_rate,
    )
    greedy = _run_greedy(
        coordinate,
        scores,
        frontier,
        weights,
        fallback,
        remaining_budget,
        protocol,
        provenance,
    )
    fabrid_macro = _run_optimized(
        coordinate,
        AllocationPolicy.FABRID_MACRO,
        scores,
        frontier,
        weights,
        fallback,
        remaining_budget,
        protocol,
        provenance,
    )
    fabrid_minimax = _run_optimized(
        coordinate,
        AllocationPolicy.FABRID_MINIMAX,
        scores,
        frontier,
        weights,
        fallback,
        remaining_budget,
        protocol,
        provenance,
    )

    runs: tuple[PolicyRun, ...] = (
        equal_fpr,
        greedy,
        fabrid_macro,
        fabrid_minimax,
    )
    policy_results: list[PolicyEvaluation] = []
    records: list[ClientResultRecord] = []
    for run in runs:
        if isinstance(run, CompletedPolicyRun):
            policy_results.append(run.result.summary)
            records.extend(run.result.records)
        else:
            policy_results.append(run.result)

    return SeedBudgetRun(
        evaluation=SeedBudgetEvaluation(
            experiment=coordinate,
            policies=tuple(policy_results),
        ),
        records=tuple(records),
        policy_runs=runs,
    )
