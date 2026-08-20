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
    allocate_equal_alert,
    allocate_equal_fpr,
    allocate_fabrid_macro,
    allocate_fabrid_minimax,
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
                calibration_row_count=(
                    client.frontier.benign_frontier.row_count
                    + client.evaluation.final_calibration.row_count
                ),
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
) -> _PolicyRunContribution:
    evaluation, policy_records = _run_policy(
        coordinate,
        allocation,
        solver,
        loaded,
        weights,
        provenance,
        fallback_rate,
    )
    return _PolicyRunContribution.completed(allocation, solver, evaluation, policy_records)


@dataclass(frozen=True, slots=True)
class SeedBudgetRun:
    evaluation: SeedBudgetEvaluation
    records: tuple[ClientResultRecord, ...]
    policy_runs: tuple[CompletedPolicyRun | ExcludedPolicyRun, ...]


def _run_policy(
    coordinate: AllocationCoordinate,
    allocation: Allocation,
    solver: SolverEvidence,
    loaded: LoadedSeedScores,
    weights: FederationWeights,
    provenance: EvaluationProvenance,
    fallback_rate: Probability,
) -> tuple[CompletedPolicyEvaluation, tuple[ClientResultRecord, ...]]:
    calibration = tuple(
        artifact
        for client in loaded.clients
        for artifact in (client.frontier.benign_frontier, client.evaluation.final_calibration)
    )
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


def oracle_loaded(loaded: LoadedSeedScores) -> LoadedSeedScores:
    clients = tuple(
        LoadedClientScores(
            client_id=client.client_id,
            frontier=FrontierScoreArtifacts(
                benign_frontier=client.frontier.benign_frontier,
                attack_validation=ScorePartitionArtifact(
                    coordinate=client.evaluation.attack_test.coordinate,
                    split=AttackSplit.VALIDATION,
                    records=tuple(
                        record.relabeled(f"oracle|{record.sample_id}")
                        for record in client.evaluation.attack_test.records
                    ),
                ),
            ),
            evaluation=client.evaluation,
        )
        for client in loaded.clients
    )
    return LoadedSeedScores(clients=clients)


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


def _minimax_contribution(
    coordinate: ExperimentCoordinate,
    problem: AllocationProblem,
    loaded: LoadedSeedScores,
    weights: FederationWeights,
    provenance: EvaluationProvenance,
    config: FabridConfig,
) -> _PolicyRunContribution:
    try:
        optimized = allocate_fabrid_minimax(problem, config.solver)
    except SolverInvalidError as error:
        return _PolicyRunContribution.excluded(
            AllocationPolicy.FABRID_MINIMAX, str(error), SolverStatus.SOLVER_INVALID
        )
    return _evaluated_contribution(
        AllocationCoordinate(coordinate, AllocationPolicy.FABRID_MINIMAX),
        merge_full_allocation(AllocationPolicy.FABRID_MINIMAX, problem, optimized.allocation),
        optimized.solver,
        loaded,
        weights,
        provenance,
        problem.frontier.fallback_rate,
    )


def _oracle_contribution(
    coordinate: ExperimentCoordinate,
    loaded: LoadedSeedScores,
    oracle_scores: LoadedSeedScores,
    weights: FederationWeights,
    provenance: EvaluationProvenance,
    config: FabridConfig,
) -> _PolicyRunContribution:
    oracle_problem = build_allocation_problem(
        build_frontier_inputs(oracle_scores, config.alpha_grid),
        weights,
        coordinate.budget,
        config.utility_eligibility,
        config.maximum_target_rate,
    )
    if oracle_problem.frontier.eligible_curves() is None:
        return _PolicyRunContribution.excluded(
            AllocationPolicy.TEST_ORACLE, _NO_ELIGIBLE_CLIENTS, SolverStatus.NOT_APPLICABLE
        )
    try:
        optimized = allocate_fabrid_macro(oracle_problem, config.solver)
    except SolverInvalidError as error:
        return _PolicyRunContribution.excluded(
            AllocationPolicy.TEST_ORACLE, str(error), SolverStatus.SOLVER_INVALID
        )
    return _evaluated_contribution(
        AllocationCoordinate(coordinate, AllocationPolicy.TEST_ORACLE),
        merge_full_allocation(AllocationPolicy.TEST_ORACLE, oracle_problem, optimized.allocation),
        optimized.solver,
        loaded,
        weights,
        provenance,
        oracle_problem.frontier.fallback_rate,
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
    oracle_scores: LoadedSeedScores,
    weights: FederationWeights,
    provenance: EvaluationProvenance,
    config: FabridConfig,
) -> _PolicyRunContribution:
    if problem.frontier.eligible_curves() is None:
        contribution = _PolicyRunContribution(runs=(), records=(), evaluations=())
        for policy in (
            AllocationPolicy.GREEDY,
            AllocationPolicy.FABRID_MACRO,
            AllocationPolicy.FABRID_MINIMAX,
            AllocationPolicy.EQ_ALERT,
            AllocationPolicy.TEST_ORACLE,
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
    optimized = _macro_contribution(coordinate, problem, loaded, weights, provenance, config).plus(
        _minimax_contribution(coordinate, problem, loaded, weights, provenance, config)
    )
    return (
        greedy.plus(optimized)
        .plus(_oracle_contribution(coordinate, loaded, oracle_scores, weights, provenance, config))
        .plus(_equal_alert_contribution(coordinate, problem, loaded, weights, provenance, config))
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
    oracle_scores: LoadedSeedScores,
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
                coordinate, problem, loaded, oracle_scores, weights, provenance, config
            )
        )
        .plus(_pooled_shared_contribution(coordinate, problem, loaded, weights, provenance))
    )
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
    )


def persist_seed_budget(paths: ArtifactPaths, run: SeedBudgetRun) -> None:
    coordinate = run.evaluation.experiment
    write_parquet_models(paths.result_path(coordinate), run.records)
    write_typed_json(
        run.evaluation,
        TypeAdapter(SeedBudgetEvaluation),
        paths.evaluation_summary_path(coordinate),
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
