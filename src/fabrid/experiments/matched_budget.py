from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from pydantic import BaseModel, ConfigDict, TypeAdapter

from fabrid.allocation.baselines import (
    ClientPooledValidation,
    FederationPooledValidation,
    select_pooled_shared_threshold,
)
from fabrid.allocation.optimization import (
    AllocationDecisionSnapshot,
    AllocationSnapshot,
    SolverEvidence,
    SolverStatus,
    not_applicable_solver_evidence,
)
from fabrid.allocation.policies import (
    allocate_common_rate_macro,
    allocate_equal_alert,
    allocate_equal_fpr,
    allocate_fabrid_cvar,
    allocate_fabrid_macro,
    allocate_fabrid_macro_without_rate_minimization,
    allocate_greedy,
)
from fabrid.allocation.problem import (
    Allocation,
    AllocationDecision,
    AllocationProblem,
    ClientUtilityCurve,
    FederationFrontierInputs,
    FederationWeights,
    FrontierScoreArtifacts,
    build_allocation_problem,
    build_client_frontier_inputs,
    merge_full_allocation,
)
from fabrid.artifacts.json import digest_text, write_typed_json
from fabrid.artifacts.parquet import write_parquet_models
from fabrid.artifacts.paths import (
    AllocationCoordinate,
    ArtifactPaths,
    ExperimentCoordinate,
    ScoreCoordinate,
)
from fabrid.config import (
    AllocationPolicy,
    AnalysisArtifactId,
    ArtifactDigest,
    AttackSplit,
    BenignSplit,
    BitCount,
    ByteCount,
    ClientId,
    DatasetId,
    DetectorSeed,
    FabridConfig,
    FailureReason,
    PayloadSizingConfig,
    Probability,
    TargetFalsePositiveRate,
    Threshold,
)
from fabrid.datasets.registry import ClientPopulation
from fabrid.detector.calibration import (
    FinalCalibrationDecision,
    FinalCalibrationInputs,
    FinalCalibrationResults,
    calibrate_final_thresholds,
)
from fabrid.detector.scoring import (
    ScorePartitionArtifact,
    load_score_partition,
    require_exclusive_sample_ids,
)
from fabrid.errors import SolverInvalidError
from fabrid.evaluation.metrics import (
    ClientFalseAlerts,
    ClientFalsePositiveRate,
    ClientMacroRecall,
    ClientResultRecord,
    CompletedPolicyEvaluation,
    ConfusionCounts,
    EvaluationProvenance,
    ExcludedPolicyEvaluation,
    SeedBudgetEvaluation,
    assert_policy_auroc_invariant,
    compute_auprc,
    compute_auroc,
    evaluate_allocation,
    evaluate_client_threshold,
    summarize_policy,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ClientEvaluationArtifacts:
    client_id: ClientId
    final_calibration: ScorePartitionArtifact
    benign_test: ScorePartitionArtifact
    attack_test: ScorePartitionArtifact

    def __post_init__(self) -> None:
        if self.final_calibration.split is not BenignSplit.FINAL_CAL:
            raise ValueError("final calibration artifact must be a final-calibration split")
        if self.benign_test.split is not BenignSplit.TEST:
            raise ValueError("benign test artifact must be a benign test split")
        if self.attack_test.split is not AttackSplit.TEST:
            raise ValueError("attack test artifact must be an attack test split")
        coordinates = (
            self.final_calibration.coordinate,
            self.benign_test.coordinate,
            self.attack_test.coordinate,
        )
        if any(coordinate.client_id != self.client_id for coordinate in coordinates):
            raise ValueError("evaluation artifacts must belong to the client")


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
        require_exclusive_sample_ids(
            tuple(
                artifact
                for client in self.clients
                for artifact in (
                    client.frontier.benign_frontier,
                    client.frontier.attack_validation,
                    client.evaluation.final_calibration,
                    client.evaluation.benign_test,
                    client.evaluation.attack_test,
                )
            )
        )

    def for_client(self, client_id: ClientId) -> LoadedClientScores:
        for client in self.clients:
            if client.client_id == client_id:
                return client
        raise KeyError(client_id)

    def digest(self) -> ArtifactDigest:
        return digest_text(
            tuple(
                artifact.digest()
                for client in self.clients
                for artifact in (
                    client.frontier.benign_frontier,
                    client.frontier.attack_validation,
                    client.evaluation.final_calibration,
                    client.evaluation.benign_test,
                    client.evaluation.attack_test,
                )
            )
        )

    @property
    def population(self) -> ClientPopulation:
        return ClientPopulation(tuple(client.client_id for client in self.clients))


def load_seed_scores(
    paths: ArtifactPaths,
    dataset_id: DatasetId,
    seed: DetectorSeed,
    population: ClientPopulation,
) -> LoadedSeedScores:
    loaded: list[LoadedClientScores] = []
    for client_id in population.clients:
        coordinate = ScoreCoordinate(dataset_id=dataset_id, detector_seed=seed, client_id=client_id)
        frontier = FrontierScoreArtifacts(
            benign_frontier=load_score_partition(
                paths.score_path(coordinate, BenignSplit.FRONTIER),
                coordinate,
                BenignSplit.FRONTIER,
            ),
            attack_validation=load_score_partition(
                paths.score_path(coordinate, AttackSplit.VALIDATION),
                coordinate,
                AttackSplit.VALIDATION,
            ),
        )
        evaluation = ClientEvaluationArtifacts(
            client_id=client_id,
            final_calibration=load_score_partition(
                paths.score_path(coordinate, BenignSplit.FINAL_CAL),
                coordinate,
                BenignSplit.FINAL_CAL,
            ),
            benign_test=load_score_partition(
                paths.score_path(coordinate, BenignSplit.TEST),
                coordinate,
                BenignSplit.TEST,
            ),
            attack_test=load_score_partition(
                paths.score_path(coordinate, AttackSplit.TEST),
                coordinate,
                AttackSplit.TEST,
            ),
        )
        loaded.append(
            LoadedClientScores(client_id=client_id, frontier=frontier, evaluation=evaluation)
        )
    return LoadedSeedScores(tuple(loaded))


def build_frontier_inputs(
    loaded: LoadedSeedScores,
    alpha_grid: tuple[TargetFalsePositiveRate, ...],
) -> FederationFrontierInputs:
    return FederationFrontierInputs(
        clients=tuple(
            build_client_frontier_inputs(
                client.frontier,
                alpha_grid,
                frontier_row_count=client.frontier.benign_frontier.row_count,
            )
            for client in loaded.clients
        )
    )


@dataclass(frozen=True, slots=True)
class CompletedPolicyRun:
    allocation: Allocation
    solver: SolverEvidence
    evaluation: CompletedPolicyEvaluation
    records: tuple[ClientResultRecord, ...]


class MatchedEqualRateSelection(BaseModel):
    model_config = ConfigDict(frozen=True)
    reference_policy: AllocationPolicy
    matched_policy: AllocationPolicy
    reference_calibration_fpr: Probability
    selected_common_rate: TargetFalsePositiveRate
    selected_calibration_fpr: Probability
    absolute_calibration_fpr_difference: Probability


class EqualRateCalibrationFrontierRow(BaseModel):
    model_config = ConfigDict(frozen=True)
    reference_policy: AllocationPolicy
    matched_policy: AllocationPolicy
    reference_calibration_fpr: Probability
    common_target_rate: TargetFalsePositiveRate
    common_calibration_fpr: Probability
    absolute_calibration_fpr_difference: Probability
    development_macro_recall: Probability
    selected: bool


@dataclass(frozen=True, slots=True)
class ExcludedPolicyRun:
    reason: FailureReason
    status: SolverStatus


@dataclass(frozen=True, slots=True)
class _PolicyRunContribution:
    runs: tuple[CompletedPolicyRun | ExcludedPolicyRun, ...]
    records: tuple[ClientResultRecord, ...]
    evaluations: tuple[CompletedPolicyEvaluation | ExcludedPolicyEvaluation, ...]

    def plus(self, other: _PolicyRunContribution) -> _PolicyRunContribution:
        return _PolicyRunContribution(
            runs=self.runs + other.runs,
            records=self.records + other.records,
            evaluations=self.evaluations + other.evaluations,
        )

    @staticmethod
    def completed(
        allocation: Allocation,
        solver: SolverEvidence,
        evaluation: CompletedPolicyEvaluation,
        policy_records: tuple[ClientResultRecord, ...],
    ) -> _PolicyRunContribution:
        return _PolicyRunContribution(
            runs=(
                CompletedPolicyRun(
                    allocation=allocation,
                    solver=solver,
                    evaluation=evaluation,
                    records=policy_records,
                ),
            ),
            records=policy_records,
            evaluations=(evaluation,),
        )

    @staticmethod
    def excluded(
        policy: AllocationPolicy, reason: FailureReason, status: SolverStatus
    ) -> _PolicyRunContribution:
        return _PolicyRunContribution(
            runs=(ExcludedPolicyRun(reason=reason, status=status),),
            records=(),
            evaluations=(ExcludedPolicyEvaluation(policy=policy, status=status, reason=reason),),
        )


def _evaluated_contribution(
    coordinate: AllocationCoordinate,
    allocation: Allocation,
    solver: SolverEvidence,
    loaded: LoadedSeedScores,
    weights: FederationWeights,
    provenance: EvaluationProvenance,
    fallback_rate: Probability,
    finite_sample_confidence: Probability | None = None,
) -> _PolicyRunContribution:
    evaluation, policy_records = _run_policy(
        coordinate,
        allocation,
        solver,
        loaded,
        weights,
        provenance,
        fallback_rate,
        finite_sample_confidence,
    )
    return _PolicyRunContribution.completed(allocation, solver, evaluation, policy_records)


@dataclass(frozen=True, slots=True)
class SeedBudgetRun:
    evaluation: SeedBudgetEvaluation
    records: tuple[ClientResultRecord, ...]
    policy_runs: tuple[CompletedPolicyRun | ExcludedPolicyRun, ...]
    matched_equal_rate_selections: tuple[MatchedEqualRateSelection, ...] = ()
    equal_rate_calibration_frontier: tuple[EqualRateCalibrationFrontierRow, ...] = ()


def _run_policy(
    coordinate: AllocationCoordinate,
    allocation: Allocation,
    solver: SolverEvidence,
    loaded: LoadedSeedScores,
    weights: FederationWeights,
    provenance: EvaluationProvenance,
    fallback_rate: Probability,
    finite_sample_confidence: Probability | None = None,
) -> tuple[CompletedPolicyEvaluation, tuple[ClientResultRecord, ...]]:
    calibration = tuple(client.evaluation.final_calibration for client in loaded.clients)
    benign_test = tuple(client.evaluation.benign_test for client in loaded.clients)
    attack_test = tuple(client.evaluation.attack_test for client in loaded.clients)
    return evaluate_allocation(
        coordinate,
        allocation,
        calibration,
        benign_test,
        attack_test,
        weights,
        solver,
        fallback_rate,
        provenance,
        finite_sample_confidence,
    )


def _empirical_fpr(scores: np.ndarray, threshold: Threshold) -> Probability:
    if scores.size == 0:
        return 0.0
    return float(np.count_nonzero(scores > threshold) / scores.size)


def _evaluate_absolute_threshold(
    coordinate: AllocationCoordinate,
    threshold: Threshold,
    loaded: LoadedSeedScores,
    weights: FederationWeights,
    provenance: EvaluationProvenance,
    fallback_rate: Probability,
) -> tuple[CompletedPolicyEvaluation, tuple[ClientResultRecord, ...]]:
    client_recalls: list[ClientMacroRecall] = []
    client_rates: list[ClientFalsePositiveRate] = []
    client_confusions: list[ConfusionCounts] = []
    client_alerts: list[ClientFalseAlerts] = []
    records: list[ClientResultRecord] = []
    benign_scores: list[np.ndarray] = []
    attack_scores: list[np.ndarray] = []
    for client in loaded.clients:
        outcome = evaluate_client_threshold(
            coordinate,
            client.client_id,
            threshold,
            _empirical_fpr(client.evaluation.benign_test.score_values(), threshold),
            0,
            None,
            client.evaluation.benign_test,
            client.evaluation.attack_test,
            weights.for_client(client.client_id),
            not_applicable_solver_evidence(),
            provenance,
        )
        client_recalls.append(outcome.macro_recall)
        client_rates.append(outcome.false_positive_rate)
        client_confusions.append(outcome.confusion)
        client_alerts.append(outcome.false_alerts)
        records.extend(outcome.records)
        benign_scores.append(client.evaluation.benign_test.score_values())
        attack_scores.append(client.evaluation.attack_test.score_values())
    summary = summarize_policy(
        coordinate,
        tuple(client_recalls),
        tuple(client_rates),
        weights,
        fallback_rate,
        None,
        tuple(client_confusions),
        tuple(client_alerts),
        compute_auroc(np.concatenate(benign_scores), np.concatenate(attack_scores)),
        compute_auprc(np.concatenate(benign_scores), np.concatenate(attack_scores)),
    )
    return summary, tuple(records)


_NO_ELIGIBLE_CLIENTS: FailureReason = "no eligible clients; policy excluded"


def _equal_fpr_contribution(
    coordinate: ExperimentCoordinate,
    problem: AllocationProblem,
    loaded: LoadedSeedScores,
    weights: FederationWeights,
    provenance: EvaluationProvenance,
) -> _PolicyRunContribution:
    return _evaluated_contribution(
        AllocationCoordinate(coordinate, AllocationPolicy.EQ_FPR),
        allocate_equal_fpr(problem),
        not_applicable_solver_evidence(),
        loaded,
        weights,
        provenance,
        problem.frontier.fallback_rate,
    )


def _final_calibration(
    allocation: Allocation,
    loaded: LoadedSeedScores,
) -> FinalCalibrationResults:
    return calibrate_final_thresholds(
        decisions=tuple(
            FinalCalibrationDecision(client_id=decision.client_id, target_rate=decision.target_rate)
            for decision in allocation.decisions
        ),
        inputs=FinalCalibrationInputs(
            clients=tuple(client.evaluation.final_calibration for client in loaded.clients)
        ),
    )


def _calibration_federation_fpr(
    allocation: Allocation,
    loaded: LoadedSeedScores,
    weights: FederationWeights,
) -> Probability:
    calibration = _final_calibration(allocation, loaded)
    return sum(
        weights.for_client(result.client_id) * result.reachability.attained_rate
        for result in calibration.clients
    )


def _development_macro_recall(allocation: Allocation, loaded: LoadedSeedScores) -> Probability:
    calibration = _final_calibration(allocation, loaded)
    recalls: list[Probability] = []
    for client in loaded.clients:
        threshold = calibration.for_client(client.client_id).threshold
        subtype_scores = client.frontier.attack_validation.subtype_scores()
        if not subtype_scores:
            raise ValueError(f"client {client.client_id} has no ATTACK_VALIDATION subtype rows")
        recalls.append(
            sum(
                float(np.count_nonzero(item.scores > threshold) / item.scores.size)
                for item in subtype_scores
            )
            / len(subtype_scores)
        )
    return sum(recalls) / len(recalls)


def _matched_equal_fpr_contribution(
    coordinate: ExperimentCoordinate,
    reference: Allocation,
    matched_policy: AllocationPolicy,
    problem: AllocationProblem,
    loaded: LoadedSeedScores,
    weights: FederationWeights,
    provenance: EvaluationProvenance,
    alpha_grid: tuple[TargetFalsePositiveRate, ...],
) -> tuple[
    _PolicyRunContribution,
    MatchedEqualRateSelection,
    tuple[EqualRateCalibrationFrontierRow, ...],
]:
    reference_rate = _calibration_federation_fpr(reference, loaded, weights)
    candidates = tuple(
        (
            rate,
            Allocation(
                policy=matched_policy,
                decisions=tuple(
                    AllocationDecision(client_id=client_id, target_rate=rate)
                    for client_id in loaded.population.clients
                ),
            ),
        )
        for rate in alpha_grid
    )
    candidate_rates = tuple(
        (rate, allocation, _calibration_federation_fpr(allocation, loaded, weights))
        for rate, allocation in candidates
    )
    selected_rate, selected_allocation, selected_calibration_fpr = min(
        candidate_rates,
        key=lambda candidate: (
            abs(candidate[2] - reference_rate),
            candidate[0],
        ),
    )
    selection = MatchedEqualRateSelection(
        reference_policy=reference.policy,
        matched_policy=matched_policy,
        reference_calibration_fpr=reference_rate,
        selected_common_rate=selected_rate,
        selected_calibration_fpr=selected_calibration_fpr,
        absolute_calibration_fpr_difference=abs(selected_calibration_fpr - reference_rate),
    )
    frontier = tuple(
        EqualRateCalibrationFrontierRow(
            reference_policy=reference.policy,
            matched_policy=matched_policy,
            reference_calibration_fpr=reference_rate,
            common_target_rate=rate,
            common_calibration_fpr=calibration_fpr,
            absolute_calibration_fpr_difference=abs(calibration_fpr - reference_rate),
            development_macro_recall=_development_macro_recall(allocation, loaded),
            selected=rate == selected_rate,
        )
        for rate, allocation, calibration_fpr in candidate_rates
    )
    return (
        _evaluated_contribution(
            AllocationCoordinate(coordinate, matched_policy),
            selected_allocation,
            not_applicable_solver_evidence(),
            loaded,
            weights,
            provenance,
            problem.frontier.fallback_rate,
        ),
        selection,
        frontier,
    )


def _macro_contribution(
    coordinate: ExperimentCoordinate,
    problem: AllocationProblem,
    loaded: LoadedSeedScores,
    weights: FederationWeights,
    provenance: EvaluationProvenance,
    config: FabridConfig,
) -> _PolicyRunContribution:
    try:
        optimized = allocate_fabrid_macro(problem, config.solver)
    except SolverInvalidError as error:
        return _PolicyRunContribution.excluded(
            AllocationPolicy.FABRID_MACRO, str(error), SolverStatus.SOLVER_INVALID
        )
    return _evaluated_contribution(
        AllocationCoordinate(coordinate, AllocationPolicy.FABRID_MACRO),
        merge_full_allocation(AllocationPolicy.FABRID_MACRO, problem, optimized.allocation),
        optimized.solver,
        loaded,
        weights,
        provenance,
        problem.frontier.fallback_rate,
    )


def _cvar_contribution(
    coordinate: ExperimentCoordinate,
    problem: AllocationProblem,
    loaded: LoadedSeedScores,
    weights: FederationWeights,
    provenance: EvaluationProvenance,
    config: FabridConfig,
) -> _PolicyRunContribution:
    try:
        optimized = allocate_fabrid_cvar(problem, config.solver)
    except SolverInvalidError as error:
        return _PolicyRunContribution.excluded(
            AllocationPolicy.FABRID_CVAR, str(error), SolverStatus.SOLVER_INVALID
        )
    return _evaluated_contribution(
        AllocationCoordinate(coordinate, AllocationPolicy.FABRID_CVAR),
        merge_full_allocation(AllocationPolicy.FABRID_CVAR, problem, optimized.allocation),
        optimized.solver,
        loaded,
        weights,
        provenance,
        problem.frontier.fallback_rate,
    )


def _finite_safe_macro_contribution(
    coordinate: ExperimentCoordinate,
    problem: AllocationProblem,
    loaded: LoadedSeedScores,
    weights: FederationWeights,
    provenance: EvaluationProvenance,
    config: FabridConfig,
) -> _PolicyRunContribution:
    try:
        optimized = allocate_fabrid_macro(problem, config.solver)
    except SolverInvalidError as error:
        return _PolicyRunContribution.excluded(
            AllocationPolicy.FABRID_MACRO_FINITE_SAFE, str(error), SolverStatus.SOLVER_INVALID
        )
    return _evaluated_contribution(
        AllocationCoordinate(coordinate, AllocationPolicy.FABRID_MACRO_FINITE_SAFE),
        merge_full_allocation(
            AllocationPolicy.FABRID_MACRO_FINITE_SAFE, problem, optimized.allocation
        ),
        optimized.solver,
        loaded,
        weights,
        provenance,
        problem.frontier.fallback_rate,
        config.calibration.finite_sample_confidence,
    )


def _macro_without_rate_minimization_contribution(
    coordinate: ExperimentCoordinate,
    problem: AllocationProblem,
    loaded: LoadedSeedScores,
    weights: FederationWeights,
    provenance: EvaluationProvenance,
    config: FabridConfig,
) -> _PolicyRunContribution:
    try:
        optimized = allocate_fabrid_macro_without_rate_minimization(problem, config.solver)
    except SolverInvalidError as error:
        return _PolicyRunContribution.excluded(
            AllocationPolicy.FABRID_MACRO_NO_RATE_MINIMIZATION,
            str(error),
            SolverStatus.SOLVER_INVALID,
        )
    return _evaluated_contribution(
        AllocationCoordinate(coordinate, AllocationPolicy.FABRID_MACRO_NO_RATE_MINIMIZATION),
        merge_full_allocation(
            AllocationPolicy.FABRID_MACRO_NO_RATE_MINIMIZATION, problem, optimized.allocation
        ),
        optimized.solver,
        loaded,
        weights,
        provenance,
        problem.frontier.fallback_rate,
    )


def _common_rate_macro_contribution(
    coordinate: ExperimentCoordinate,
    problem: AllocationProblem,
    loaded: LoadedSeedScores,
    weights: FederationWeights,
    provenance: EvaluationProvenance,
) -> _PolicyRunContribution:
    try:
        allocation = allocate_common_rate_macro(problem)
    except ValueError as error:
        return _PolicyRunContribution.excluded(
            AllocationPolicy.COMMON_RATE_MACRO, str(error), SolverStatus.SOLVER_INVALID
        )
    return _evaluated_contribution(
        AllocationCoordinate(coordinate, AllocationPolicy.COMMON_RATE_MACRO),
        merge_full_allocation(AllocationPolicy.COMMON_RATE_MACRO, problem, allocation),
        not_applicable_solver_evidence(),
        loaded,
        weights,
        provenance,
        problem.frontier.fallback_rate,
    )


def _finite_safe_cvar_contribution(
    coordinate: ExperimentCoordinate,
    problem: AllocationProblem,
    loaded: LoadedSeedScores,
    weights: FederationWeights,
    provenance: EvaluationProvenance,
    config: FabridConfig,
) -> _PolicyRunContribution:
    try:
        optimized = allocate_fabrid_cvar(problem, config.solver)
    except SolverInvalidError as error:
        return _PolicyRunContribution.excluded(
            AllocationPolicy.FABRID_CVAR_FINITE_SAFE, str(error), SolverStatus.SOLVER_INVALID
        )
    return _evaluated_contribution(
        AllocationCoordinate(coordinate, AllocationPolicy.FABRID_CVAR_FINITE_SAFE),
        merge_full_allocation(
            AllocationPolicy.FABRID_CVAR_FINITE_SAFE, problem, optimized.allocation
        ),
        optimized.solver,
        loaded,
        weights,
        provenance,
        problem.frontier.fallback_rate,
        config.calibration.finite_sample_confidence,
    )


def _equal_alert_contribution(
    coordinate: ExperimentCoordinate,
    problem: AllocationProblem,
    loaded: LoadedSeedScores,
    weights: FederationWeights,
    provenance: EvaluationProvenance,
    config: FabridConfig,
) -> _PolicyRunContribution:
    if len({client.weight for client in weights.clients}) <= 1:
        return _PolicyRunContribution(runs=(), records=(), evaluations=())
    return _evaluated_contribution(
        AllocationCoordinate(coordinate, AllocationPolicy.EQ_ALERT),
        allocate_equal_alert(problem, config.solver),
        not_applicable_solver_evidence(),
        loaded,
        weights,
        provenance,
        problem.frontier.fallback_rate,
    )


def _eligible_or_excluded_contribution(
    coordinate: ExperimentCoordinate,
    problem: AllocationProblem,
    loaded: LoadedSeedScores,
    weights: FederationWeights,
    provenance: EvaluationProvenance,
    config: FabridConfig,
) -> _PolicyRunContribution:
    if problem.frontier.eligible_curves() is None:
        contribution = _PolicyRunContribution(runs=(), records=(), evaluations=())
        for policy in (
            AllocationPolicy.GREEDY,
            AllocationPolicy.FABRID_MACRO,
            AllocationPolicy.FABRID_CVAR,
            AllocationPolicy.FABRID_MACRO_FINITE_SAFE,
            AllocationPolicy.FABRID_CVAR_FINITE_SAFE,
            AllocationPolicy.FABRID_MACRO_NO_RATE_MINIMIZATION,
            AllocationPolicy.COMMON_RATE_MACRO,
            AllocationPolicy.EQ_ALERT,
        ):
            contribution = contribution.plus(
                _PolicyRunContribution.excluded(
                    policy, _NO_ELIGIBLE_CLIENTS, SolverStatus.NOT_APPLICABLE
                )
            )
        return contribution
    greedy = _evaluated_contribution(
        AllocationCoordinate(coordinate, AllocationPolicy.GREEDY),
        merge_full_allocation(AllocationPolicy.GREEDY, problem, allocate_greedy(problem)),
        not_applicable_solver_evidence(),
        loaded,
        weights,
        provenance,
        problem.frontier.fallback_rate,
    )
    optimized = (
        _macro_contribution(coordinate, problem, loaded, weights, provenance, config)
        .plus(_cvar_contribution(coordinate, problem, loaded, weights, provenance, config))
        .plus(
            _finite_safe_macro_contribution(
                coordinate, problem, loaded, weights, provenance, config
            )
        )
        .plus(
            _finite_safe_cvar_contribution(coordinate, problem, loaded, weights, provenance, config)
        )
        .plus(
            _macro_without_rate_minimization_contribution(
                coordinate, problem, loaded, weights, provenance, config
            )
        )
        .plus(_common_rate_macro_contribution(coordinate, problem, loaded, weights, provenance))
    )
    return greedy.plus(optimized).plus(
        _equal_alert_contribution(coordinate, problem, loaded, weights, provenance, config)
    )


def _pooled_shared_contribution(
    coordinate: ExperimentCoordinate,
    problem: AllocationProblem,
    loaded: LoadedSeedScores,
    weights: FederationWeights,
    provenance: EvaluationProvenance,
) -> _PolicyRunContribution:
    validation = FederationPooledValidation(
        clients=tuple(
            ClientPooledValidation(
                client_id=client.client_id,
                benign_frontier=client.frontier.benign_frontier,
                attack_validation=client.frontier.attack_validation,
            )
            for client in loaded.clients
        )
    )
    pooled = select_pooled_shared_threshold(validation, problem)
    evaluation, pooled_records = _evaluate_absolute_threshold(
        AllocationCoordinate(coordinate, AllocationPolicy.POOLED_SHARED),
        pooled.threshold,
        loaded,
        weights,
        provenance,
        problem.frontier.fallback_rate,
    )
    return _PolicyRunContribution.completed(
        Allocation(
            policy=AllocationPolicy.POOLED_SHARED,
            decisions=tuple(
                AllocationDecision(client_id=client_id, target_rate=pooled.federation_fpr)
                for client_id in loaded.population.clients
            ),
        ),
        not_applicable_solver_evidence(),
        evaluation,
        pooled_records,
    )


def run_seed_budget(
    coordinate: ExperimentCoordinate,
    loaded: LoadedSeedScores,
    config: FabridConfig,
    provenance: EvaluationProvenance,
    weights: FederationWeights,
) -> SeedBudgetRun:
    problem = build_allocation_problem(
        build_frontier_inputs(loaded, config.alpha_grid),
        weights,
        coordinate.budget,
        config.utility_eligibility,
        config.maximum_target_rate,
    )
    contribution = (
        _equal_fpr_contribution(
            coordinate,
            problem,
            loaded,
            weights,
            provenance,
        )
        .plus(
            _eligible_or_excluded_contribution(
                coordinate, problem, loaded, weights, provenance, config
            )
        )
        .plus(_pooled_shared_contribution(coordinate, problem, loaded, weights, provenance))
    )
    selections: list[MatchedEqualRateSelection] = []
    equal_rate_frontier: list[EqualRateCalibrationFrontierRow] = []
    references = {
        run.allocation.policy: run.allocation
        for run in contribution.runs
        if isinstance(run, CompletedPolicyRun)
    }
    for reference_policy, matched_policy in (
        (AllocationPolicy.FABRID_MACRO, AllocationPolicy.EQ_FPR_MATCHED_MACRO),
        (AllocationPolicy.FABRID_CVAR, AllocationPolicy.EQ_FPR_MATCHED_CVAR),
    ):
        reference = references.get(reference_policy)
        if reference is None:
            continue
        matched, selection, frontier = _matched_equal_fpr_contribution(
            coordinate,
            reference,
            matched_policy,
            problem,
            loaded,
            weights,
            provenance,
            config.alpha_grid,
        )
        contribution = contribution.plus(matched)
        selections.append(selection)
        equal_rate_frontier.extend(frontier)
    completed = tuple(
        evaluation
        for evaluation in contribution.evaluations
        if isinstance(evaluation, CompletedPolicyEvaluation)
    )
    if completed:
        assert_policy_auroc_invariant(completed)
    return SeedBudgetRun(
        evaluation=SeedBudgetEvaluation(experiment=coordinate, policies=contribution.evaluations),
        records=contribution.records,
        policy_runs=contribution.runs,
        matched_equal_rate_selections=tuple(selections),
        equal_rate_calibration_frontier=tuple(equal_rate_frontier),
    )


def persist_seed_budget(paths: ArtifactPaths, run: SeedBudgetRun) -> None:
    coordinate = run.evaluation.experiment
    write_parquet_models(paths.result_path(coordinate), run.records)
    write_typed_json(
        run.evaluation,
        TypeAdapter(SeedBudgetEvaluation),
        paths.evaluation_summary_path(coordinate),
    )
    if run.equal_rate_calibration_frontier:
        write_parquet_models(
            paths.analysis_path(coordinate, AnalysisArtifactId.EQUAL_RATE_CALIBRATION_FRONTIER),
            run.equal_rate_calibration_frontier,
        )
    for policy_run in run.policy_runs:
        if not isinstance(policy_run, CompletedPolicyRun):
            continue
        snapshot = AllocationSnapshot(
            policy=policy_run.allocation.policy,
            decisions=tuple(
                AllocationDecisionSnapshot(
                    client_id=decision.client_id, target_rate=decision.target_rate
                )
                for decision in policy_run.allocation.decisions
            ),
            solver=policy_run.solver,
        )
        write_typed_json(
            snapshot,
            TypeAdapter(AllocationSnapshot),
            paths.allocation_path(AllocationCoordinate(coordinate, policy_run.allocation.policy)),
        )


@dataclass(frozen=True, slots=True)
class PayloadMeasurement(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    client_id: ClientId
    logical_bytes: ByteCount
    serialized_utility_bytes: ByteCount
    server_response_bits: BitCount


def measure_client_payload(
    curve: ClientUtilityCurve,
    sizing: PayloadSizingConfig,
) -> PayloadMeasurement:
    utilities = np.asarray(tuple(point.utility for point in curve.points), dtype=np.float32)
    if utilities.size != sizing.alpha_candidate_count:
        raise ValueError(
            "utility curve size must match the protocol candidate count "
            f"({utilities.size} != {sizing.alpha_candidate_count})"
        )
    logical = sizing.metadata_envelope_bytes + int(utilities.nbytes)
    return PayloadMeasurement(
        client_id=curve.client_id,
        logical_bytes=logical,
        serialized_utility_bytes=int(utilities.nbytes),
        server_response_bits=sizing.server_response_bits,
    )
