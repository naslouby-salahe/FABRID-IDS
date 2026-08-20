from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator
from scipy import stats

from fabrid.allocation.optimization import SolverEvidence, SolverStatus
from fabrid.allocation.problem import (
    Allocation,
    FederationWeights,
    SubtypeConfusion,
    SubtypeConfusionCounts,
)
from fabrid.artifacts.paths import AllocationCoordinate, ExperimentCoordinate
from fabrid.config import (
    TIGHT_TOLERANCE,
    AllocationPolicy,
    ArtifactDigest,
    AttackSubtypeId,
    BalancedAccuracy,
    BudgetId,
    BudgetUsageRatio,
    ClientId,
    ClientWeight,
    CoefficientOfVariation,
    DatasetId,
    DetectionUtility,
    DetectorSeed,
    ExperimentId,
    F1Score,
    FailureReason,
    FalseAlertCount,
    FalsePositiveBudget,
    FalsePositiveRate,
    GitCommit,
    MacroRecall,
    Probability,
    ReplicateIndex,
    RowCount,
    SolverGap,
    SolverObjective,
    SolverRuntimeMilliseconds,
    TargetFalsePositiveRate,
    Threshold,
    TruePositiveRate,
    UtilityDispersion,
    WeightMode,
    WorstClientRecall,
)
from fabrid.detector.calibration import (
    FinalCalibrationDecision,
    FinalCalibrationInputs,
    calibrate_final_thresholds,
)
from fabrid.detector.scoring import ScorePartitionArtifact


class MetricId(StrEnum):
    MACRO_RECALL = "macro_recall"
    WORST_CLIENT_RECALL = "worst_client_recall"
    FEDERATION_FPR = "federation_fpr"
    BUDGET_USAGE_RATIO = "budget_usage_ratio"


@dataclass(frozen=True, slots=True)
class EvaluationProvenance:
    model_sha256: ArtifactDigest
    score_sha256: ArtifactDigest
    split_sha256: ArtifactDigest
    feature_sha256: ArtifactDigest
    protocol_sha256: ArtifactDigest
    git_commit: GitCommit


class ConfusionCounts(BaseModel):
    model_config = ConfigDict(frozen=True)
    true_positive: RowCount
    false_negative: RowCount
    false_positive: RowCount
    true_negative: RowCount


class ClientResultRecord(BaseModel):
    model_config = ConfigDict(frozen=True)
    experiment_id: ExperimentId
    dataset_id: DatasetId
    seed: DetectorSeed
    budget_id: BudgetId
    budget_value: FalsePositiveBudget
    weight_mode: WeightMode
    policy: AllocationPolicy
    client_id: ClientId
    alpha_selected: TargetFalsePositiveRate
    threshold: Threshold
    calibration_n: RowCount
    nominal_weight: ClientWeight
    realized_weight: ClientWeight
    n_benign_test: RowCount
    n_attack_test: RowCount
    attack_subtype: AttackSubtypeId | None
    tp: RowCount
    fn: RowCount
    fp: RowCount
    tn: RowCount
    fpr: FalsePositiveRate
    tpr: TruePositiveRate
    macro_attack_recall: MacroRecall
    false_alert_count: FalseAlertCount
    solver_status: SolverStatus
    solver_objective: SolverObjective | None
    solver_gap: SolverGap | None
    solver_runtime_ms: SolverRuntimeMilliseconds | None
    model_sha256: ArtifactDigest
    score_sha256: ArtifactDigest
    split_sha256: ArtifactDigest
    feature_sha256: ArtifactDigest
    protocol_sha256: ArtifactDigest
    git_commit: GitCommit

    @model_validator(mode="after")
    def _validate(self) -> ClientResultRecord:
        if self.attack_subtype is None and self.n_attack_test > 0:
            raise ValueError("attack result rows must carry an attack subtype")
        return self


class FprDispersion(BaseModel):
    model_config = ConfigDict(frozen=True)
    median: FalsePositiveRate
    interquartile_range: FalsePositiveRate
    minimum: FalsePositiveRate
    maximum: FalsePositiveRate
    coefficient_of_variation: CoefficientOfVariation | None

    @model_validator(mode="after")
    def _validate(self) -> FprDispersion:
        if self.coefficient_of_variation is not None and self.coefficient_of_variation < 0.0:
            raise ValueError("coefficient of variation must be non-negative")
        return self


class UtilityHeterogeneity(BaseModel):
    model_config = ConfigDict(frozen=True)
    seed: DetectorSeed
    candidate_alpha: Probability
    dispersion: UtilityDispersion
    aggregate: UtilityDispersion


class StabilityReplicate(BaseModel):
    model_config = ConfigDict(frozen=True)
    seed: DetectorSeed
    budget_id: BudgetId
    replicate_index: ReplicateIndex
    client_id: ClientId
    alpha_selected: TargetFalsePositiveRate


class ClientStabilitySummary(BaseModel):
    model_config = ConfigDict(frozen=True)
    seed: DetectorSeed
    budget_id: BudgetId
    client_id: ClientId
    modal_alpha: TargetFalsePositiveRate
    modal_frequency: Probability
    median_alpha: TargetFalsePositiveRate
    percentile_5: TargetFalsePositiveRate
    percentile_95: TargetFalsePositiveRate
    instability: Probability


class UtilityCurveRow(BaseModel):
    model_config = ConfigDict(frozen=True)
    client_id: ClientId
    target_rate: TargetFalsePositiveRate
    utility: DetectionUtility


class CompletedPolicyEvaluation(BaseModel):
    model_config = ConfigDict(frozen=True)
    policy: AllocationPolicy
    macro_recall: MacroRecall
    worst_client_recall: WorstClientRecall
    federation_fpr: FalsePositiveRate
    budget_usage: BudgetUsageRatio | None
    fallback_rate: Probability
    fpr_dispersion: FprDispersion
    solver_runtime_ms: SolverRuntimeMilliseconds | None
    pooled_recall: Probability
    macro_f1: F1Score
    balanced_accuracy: BalancedAccuracy
    auroc: Probability
    auprc: Probability
    false_alert_gini: Probability
    budget_violation: BudgetUsageRatio | None

    @model_validator(mode="after")
    def _validate(self) -> CompletedPolicyEvaluation:
        if (self.budget_usage is None) != (self.budget_violation is None):
            raise ValueError("budget_violation is None if and only if budget_usage is None")
        if self.budget_usage is not None and self.budget_violation is not None:
            expected = budget_violation_ratio(self.budget_usage)
            if expected is None or abs(self.budget_violation - expected) >= TIGHT_TOLERANCE:
                raise ValueError("budget_violation must equal the excess of usage over one")
        return self


class ExcludedPolicyEvaluation(BaseModel):
    model_config = ConfigDict(frozen=True)
    policy: AllocationPolicy
    status: SolverStatus
    reason: FailureReason

    @model_validator(mode="after")
    def _validate(self) -> ExcludedPolicyEvaluation:
        if self.status is SolverStatus.OPTIMAL:
            raise ValueError("excluded policy evaluation may not have optimal solver status")
        return self


PolicyEvaluation = CompletedPolicyEvaluation | ExcludedPolicyEvaluation


class SeedBudgetEvaluation(BaseModel):
    model_config = ConfigDict(frozen=True)
    experiment: ExperimentCoordinate
    policies: tuple[PolicyEvaluation, ...]

    @model_validator(mode="after")
    def _validate(self) -> SeedBudgetEvaluation:
        policy_ids = tuple(policy.policy for policy in self.policies)
        if len(set(policy_ids)) != len(policy_ids):
            raise ValueError("seed-budget evaluation contains duplicate policies")
        return self

    def policy(self, policy: AllocationPolicy) -> PolicyEvaluation:
        for evaluation in self.policies:
            if evaluation.policy is policy:
                return evaluation
        raise KeyError(policy.value)


class ClientOperatingPoint(BaseModel):
    model_config = ConfigDict(frozen=True)
    client_id: ClientId
    policy: AllocationPolicy
    target_rate: TargetFalsePositiveRate
    threshold: Threshold
    false_positive_rate: FalsePositiveRate
    macro_attack_recall: MacroRecall


@dataclass(frozen=True, slots=True)
class SubtypeRecall:
    subtype: AttackSubtypeId
    rate: TruePositiveRate


@dataclass(frozen=True, slots=True)
class ClientMacroRecall:
    client_id: ClientId
    recall: MacroRecall


@dataclass(frozen=True, slots=True)
class ClientFalsePositiveRate:
    client_id: ClientId
    rate: FalsePositiveRate


@dataclass(frozen=True, slots=True)
class ClientFalseAlerts:
    client_id: ClientId
    count: FalseAlertCount


def client_macro_recall(subtypes: tuple[SubtypeRecall, ...]) -> MacroRecall:
    if not subtypes:
        raise ValueError("client macro recall requires at least one attack subtype")
    subtype_ids = tuple(subtype.subtype for subtype in subtypes)
    if len(set(subtype_ids)) != len(subtype_ids):
        raise ValueError("client macro recall contains duplicate attack subtypes")
    return sum(subtype.rate for subtype in subtypes) / len(subtypes)


def federation_macro_recall(clients: tuple[ClientMacroRecall, ...]) -> MacroRecall:
    if not clients:
        raise ValueError("federation macro recall requires at least one client")
    client_ids = tuple(client.client_id for client in clients)
    if len(set(client_ids)) != len(client_ids):
        raise ValueError("federation macro recall contains duplicate clients")
    return sum(client.recall for client in clients) / len(clients)


def worst_client_recall(clients: tuple[ClientMacroRecall, ...]) -> WorstClientRecall:
    if not clients:
        raise ValueError("worst-client recall requires at least one client")
    return min(client.recall for client in clients)


def federation_fpr(
    client_rates: tuple[ClientFalsePositiveRate, ...],
    weights: FederationWeights,
) -> FalsePositiveRate:
    if not client_rates:
        raise ValueError("federation FPR requires at least one client")
    rate_clients = {client.client_id for client in client_rates}
    weight_clients = {client.client_id for client in weights.clients}
    if rate_clients != weight_clients:
        raise ValueError("client FPRs and federation weights must cover the same clients")
    return sum(weights.for_client(client.client_id) * client.rate for client in client_rates)


def budget_usage_ratio(
    federation_rate: FalsePositiveRate,
    budget: FalsePositiveBudget,
) -> BudgetUsageRatio | None:
    if abs(budget) <= TIGHT_TOLERANCE:
        return None
    return federation_rate / budget


def budget_violation_ratio(usage: BudgetUsageRatio | None) -> BudgetUsageRatio | None:
    if usage is None:
        return None
    return max(0.0, usage - 1.0)


def fpr_dispersion(
    client_rates: tuple[ClientFalsePositiveRate, ...],
) -> FprDispersion:
    if not client_rates:
        raise ValueError("FPR dispersion requires at least one client")
    values = np.asarray(tuple(client.rate for client in client_rates), dtype=np.float64)
    mean = float(np.mean(values))
    standard_deviation = float(np.std(values))
    coefficient = None if abs(mean) <= TIGHT_TOLERANCE else standard_deviation / mean
    return FprDispersion(
        median=float(np.percentile(values, 50)),
        interquartile_range=float(np.percentile(values, 75) - np.percentile(values, 25)),
        minimum=float(np.min(values)),
        maximum=float(np.max(values)),
        coefficient_of_variation=coefficient,
    )


def false_alert_gini(clients: tuple[ClientFalseAlerts, ...]) -> Probability:
    if not clients:
        raise ValueError("false-alert Gini requires at least one client")
    counts = tuple(client.count for client in clients)
    total = sum(counts)
    if total == 0:
        return 0.0
    client_count = len(counts)
    absolute_differences = sum(abs(left - right) for left in counts for right in counts)
    return absolute_differences / (2.0 * client_count * total)


def pooled_recall(confusions: tuple[ConfusionCounts, ...]) -> Probability:
    total_true_positive = sum(entry.true_positive for entry in confusions)
    total_false_negative = sum(entry.false_negative for entry in confusions)
    denominator = total_true_positive + total_false_negative
    if denominator == 0:
        raise ValueError("pooled recall requires at least one attack row")
    return total_true_positive / denominator


def _client_f1(confusion: ConfusionCounts) -> F1Score:
    precision_denominator = confusion.true_positive + confusion.false_positive
    recall_denominator = confusion.true_positive + confusion.false_negative
    if precision_denominator == 0 or recall_denominator == 0:
        return 0.0
    precision = confusion.true_positive / precision_denominator
    recall = confusion.true_positive / recall_denominator
    if precision + recall <= TIGHT_TOLERANCE:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def macro_f1(confusions: tuple[ConfusionCounts, ...]) -> MacroRecall:
    if not confusions:
        raise ValueError("macro F1 requires at least one client confusion")
    return sum(_client_f1(entry) for entry in confusions) / len(confusions)


def _client_balanced_accuracy(confusion: ConfusionCounts) -> BalancedAccuracy:
    attack_denominator = confusion.true_positive + confusion.false_negative
    benign_denominator = confusion.true_negative + confusion.false_positive
    if attack_denominator == 0 or benign_denominator == 0:
        raise ValueError("balanced accuracy requires attack and benign rows for every client")
    sensitivity = confusion.true_positive / attack_denominator
    specificity = confusion.true_negative / benign_denominator
    return (sensitivity + specificity) / 2.0


def balanced_accuracy(confusions: tuple[ConfusionCounts, ...]) -> Probability:
    if not confusions:
        raise ValueError("balanced accuracy requires at least one client confusion")
    return sum(_client_balanced_accuracy(entry) for entry in confusions) / len(confusions)


def compute_auroc(benign_scores: np.ndarray, attack_scores: np.ndarray) -> Probability:
    if benign_scores.size == 0 or attack_scores.size == 0:
        raise ValueError("AUROC requires benign and attack scores")
    scores = np.concatenate((benign_scores, attack_scores))
    attack_mask = np.concatenate(
        (
            np.zeros(benign_scores.size, dtype=np.bool_),
            np.ones(attack_scores.size, dtype=np.bool_),
        )
    )
    ranks = stats.rankdata(scores)
    positive_count = attack_scores.size
    negative_count = benign_scores.size
    positive_rank_sum = float(np.sum(ranks[attack_mask]))
    area = (positive_rank_sum - positive_count * (positive_count + 1) / 2) / (
        positive_count * negative_count
    )
    return float(area)


def compute_auprc(benign_scores: np.ndarray, attack_scores: np.ndarray) -> Probability:
    if benign_scores.size == 0 or attack_scores.size == 0:
        raise ValueError("AUPRC requires benign and attack scores")
    scores = np.concatenate((benign_scores, attack_scores))
    attack_mask = np.concatenate(
        (
            np.zeros(benign_scores.size, dtype=np.bool_),
            np.ones(attack_scores.size, dtype=np.bool_),
        )
    )
    order = np.argsort(-scores, kind="stable")
    sorted_attacks = attack_mask[order]
    cumulative_true_positive = np.cumsum(sorted_attacks)
    cumulative_false_positive = np.cumsum(~sorted_attacks)
    recall = cumulative_true_positive / attack_scores.size
    precision = cumulative_true_positive / (cumulative_true_positive + cumulative_false_positive)
    recall_with_origin = np.concatenate((np.asarray((0.0,)), recall))
    precision_with_origin = np.concatenate((np.asarray((1.0,)), precision))
    return float(np.trapezoid(precision_with_origin, recall_with_origin))


@dataclass(frozen=True, slots=True)
class MetricArea:
    policy: AllocationPolicy
    area: Probability


def assert_auroc_invariant(
    areas: tuple[MetricArea, ...],
    tolerance: Probability = TIGHT_TOLERANCE,
) -> None:
    if not areas:
        raise ValueError("AUROC invariance requires at least one policy")
    policies = tuple(item.policy for item in areas)
    if len(set(policies)) != len(policies):
        raise ValueError("AUROC invariance evidence contains duplicate policies")
    values = tuple(item.area for item in areas)
    if max(values) - min(values) >= tolerance:
        raise ValueError("AUROC invariant violated across frozen-score policies")


def assert_policy_auroc_invariant(
    evaluations: tuple[CompletedPolicyEvaluation, ...],
) -> None:
    assert_auroc_invariant(
        tuple(
            MetricArea(policy=evaluation.policy, area=evaluation.auroc)
            for evaluation in evaluations
        )
    )


@dataclass(frozen=True, slots=True)
class ClientEvaluationOutcome:
    macro_recall: ClientMacroRecall
    false_positive_rate: ClientFalsePositiveRate
    confusion: ConfusionCounts
    false_alerts: ClientFalseAlerts
    records: tuple[ClientResultRecord, ...]


def evaluate_client_threshold(
    coordinate: AllocationCoordinate,
    client_id: ClientId,
    threshold: Threshold,
    alpha_selected: TargetFalsePositiveRate,
    calibration_count: RowCount,
    benign_test: ScorePartitionArtifact,
    attack_test: ScorePartitionArtifact,
    client_weight: ClientWeight,
    solver: SolverEvidence,
    provenance: EvaluationProvenance,
) -> ClientEvaluationOutcome:
    benign_scores = benign_test.score_values()
    false_positive_count = int(np.count_nonzero(benign_scores > threshold))
    true_negative_count = benign_scores.size - false_positive_count
    client_fpr = 0.0 if benign_scores.size == 0 else false_positive_count / benign_scores.size
    subtype_scores = attack_test.subtype_scores()
    if not subtype_scores:
        raise ValueError(f"client {client_id} has no ATTACK_TEST subtype rows")
    subtype_recalls: list[SubtypeRecall] = []
    subtype_confusions: list[SubtypeConfusion] = []
    for item in subtype_scores:
        scores = item.scores
        true_positive = int(np.count_nonzero(scores > threshold))
        false_negative = scores.size - true_positive
        subtype_recalls.append(
            SubtypeRecall(
                subtype=item.subtype,
                rate=0.0 if scores.size == 0 else true_positive / scores.size,
            )
        )
        subtype_confusions.append(
            SubtypeConfusion(
                subtype=item.subtype,
                counts=SubtypeConfusionCounts(
                    true_positive=true_positive, false_negative=false_negative
                ),
            )
        )
    macro_recall = client_macro_recall(tuple(subtype_recalls))
    records: list[ClientResultRecord] = []
    for confusion in subtype_confusions:
        attack_test_count = confusion.counts.true_positive + confusion.counts.false_negative
        tpr = 0.0 if attack_test_count == 0 else confusion.counts.true_positive / attack_test_count
        records.append(
            ClientResultRecord(
                experiment_id=coordinate.experiment.experiment_id,
                dataset_id=coordinate.experiment.dataset_id,
                seed=coordinate.experiment.detector_seed,
                budget_id=coordinate.experiment.budget_id,
                budget_value=coordinate.experiment.budget,
                weight_mode=coordinate.experiment.weight_mode,
                policy=coordinate.policy,
                client_id=client_id,
                alpha_selected=alpha_selected,
                threshold=threshold,
                calibration_n=calibration_count,
                nominal_weight=client_weight,
                realized_weight=client_weight,
                n_benign_test=benign_scores.size,
                n_attack_test=attack_test_count,
                attack_subtype=confusion.subtype,
                tp=confusion.counts.true_positive,
                fn=confusion.counts.false_negative,
                fp=false_positive_count,
                tn=true_negative_count,
                fpr=client_fpr,
                tpr=tpr,
                macro_attack_recall=macro_recall,
                false_alert_count=false_positive_count,
                solver_status=solver.status,
                solver_objective=solver.final_objective,
                solver_gap=solver.final_gap,
                solver_runtime_ms=solver.total_runtime,
                model_sha256=provenance.model_sha256,
                score_sha256=provenance.score_sha256,
                split_sha256=provenance.split_sha256,
                feature_sha256=provenance.feature_sha256,
                protocol_sha256=provenance.protocol_sha256,
                git_commit=provenance.git_commit,
            )
        )
    return ClientEvaluationOutcome(
        macro_recall=ClientMacroRecall(client_id=client_id, recall=macro_recall),
        false_positive_rate=ClientFalsePositiveRate(client_id=client_id, rate=client_fpr),
        confusion=ConfusionCounts(
            true_positive=sum(subtype.counts.true_positive for subtype in subtype_confusions),
            false_negative=sum(subtype.counts.false_negative for subtype in subtype_confusions),
            false_positive=false_positive_count,
            true_negative=true_negative_count,
        ),
        false_alerts=ClientFalseAlerts(client_id=client_id, count=false_positive_count),
        records=tuple(records),
    )


def mean(values: tuple[Probability, ...]) -> Probability:
    if not values:
        raise ValueError("mean requires at least one value")
    return sum(values) / len(values)


def summarize_policy(
    coordinate: AllocationCoordinate,
    client_recalls: tuple[ClientMacroRecall, ...],
    client_rates: tuple[ClientFalsePositiveRate, ...],
    weights: FederationWeights,
    fallback_rate: Probability,
    solver_runtime_ms: SolverRuntimeMilliseconds | None,
    client_confusions: tuple[ConfusionCounts, ...],
    client_alerts: tuple[ClientFalseAlerts, ...],
    auroc: Probability,
    auprc: Probability,
) -> CompletedPolicyEvaluation:
    if len(client_confusions) != len(client_recalls):
        raise ValueError("client confusions must cover the same clients as client recalls")
    if tuple(alert.client_id for alert in client_alerts) != tuple(
        recall.client_id for recall in client_recalls
    ):
        raise ValueError("client false alerts must cover the same clients as client recalls")
    federation_rate = federation_fpr(client_rates, weights)
    budget_usage = budget_usage_ratio(federation_rate, coordinate.experiment.budget)
    return CompletedPolicyEvaluation(
        policy=coordinate.policy,
        macro_recall=federation_macro_recall(client_recalls),
        worst_client_recall=worst_client_recall(client_recalls),
        federation_fpr=federation_rate,
        budget_usage=budget_usage,
        fallback_rate=fallback_rate,
        fpr_dispersion=fpr_dispersion(client_rates),
        solver_runtime_ms=solver_runtime_ms,
        pooled_recall=pooled_recall(client_confusions),
        macro_f1=macro_f1(client_confusions),
        balanced_accuracy=balanced_accuracy(client_confusions),
        auroc=auroc,
        auprc=auprc,
        false_alert_gini=false_alert_gini(client_alerts),
        budget_violation=budget_violation_ratio(budget_usage),
    )


def evaluate_allocation(
    coordinate: AllocationCoordinate,
    allocation: Allocation,
    calibration_artifacts: tuple[ScorePartitionArtifact, ...],
    benign_test_artifacts: tuple[ScorePartitionArtifact, ...],
    attack_test_artifacts: tuple[ScorePartitionArtifact, ...],
    weights: FederationWeights,
    solver: SolverEvidence,
    fallback_rate: Probability,
    provenance: EvaluationProvenance,
) -> tuple[CompletedPolicyEvaluation, tuple[ClientResultRecord, ...]]:
    if coordinate.policy is not allocation.policy:
        raise ValueError("allocation coordinate and allocation policy must match")
    allocation_clients = {decision.client_id for decision in allocation.decisions}
    artifact_clients = {artifact.coordinate.client_id for artifact in calibration_artifacts}
    benign_clients = {artifact.coordinate.client_id for artifact in benign_test_artifacts}
    attack_clients = {artifact.coordinate.client_id for artifact in attack_test_artifacts}
    weight_clients = {client.client_id for client in weights.clients}
    if not (
        allocation_clients == artifact_clients == benign_clients == attack_clients == weight_clients
    ):
        raise ValueError(
            "allocation, calibration, test artifacts, and weights must cover one client set"
        )
    final_calibration = calibrate_final_thresholds(
        decisions=tuple(
            FinalCalibrationDecision(
                client_id=decision.client_id,
                target_rate=decision.target_rate,
            )
            for decision in allocation.decisions
        ),
        inputs=FinalCalibrationInputs(clients=calibration_artifacts),
    )
    client_macro_recalls: list[ClientMacroRecall] = []
    client_false_positive_rates: list[ClientFalsePositiveRate] = []
    client_confusions: list[ConfusionCounts] = []
    client_alerts: list[ClientFalseAlerts] = []
    records: list[ClientResultRecord] = []
    per_client_calibration = {result.client_id: result for result in final_calibration.clients}
    artifacts_by_client = {
        artifact.coordinate.client_id: artifact for artifact in benign_test_artifacts
    }
    attacks_by_client = {
        artifact.coordinate.client_id: artifact for artifact in attack_test_artifacts
    }
    for decision in allocation.decisions:
        client_id = decision.client_id
        calibration = per_client_calibration[client_id]
        outcome = evaluate_client_threshold(
            coordinate,
            client_id,
            calibration.threshold,
            calibration.target_rate,
            calibration.calibration_count,
            artifacts_by_client[client_id],
            attacks_by_client[client_id],
            weights.for_client(client_id),
            solver,
            provenance,
        )
        client_macro_recalls.append(outcome.macro_recall)
        client_false_positive_rates.append(outcome.false_positive_rate)
        client_confusions.append(outcome.confusion)
        client_alerts.append(outcome.false_alerts)
        records.extend(outcome.records)
    benign_scores = np.concatenate(
        tuple(artifact.score_values() for artifact in benign_test_artifacts)
    )
    attack_scores = np.concatenate(
        tuple(artifact.score_values() for artifact in attack_test_artifacts)
    )
    summary = summarize_policy(
        coordinate,
        tuple(client_macro_recalls),
        tuple(client_false_positive_rates),
        weights,
        fallback_rate,
        solver.total_runtime,
        tuple(client_confusions),
        tuple(client_alerts),
        compute_auroc(benign_scores, attack_scores),
        compute_auprc(benign_scores, attack_scores),
    )
    return summary, tuple(records)
