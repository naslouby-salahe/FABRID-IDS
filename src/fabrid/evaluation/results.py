from __future__ import annotations

from dataclasses import dataclass

from fabrid.allocation.solver import SolverEvidence
from fabrid.domain.coordinates import AllocationCoordinate, ExperimentCoordinate
from fabrid.domain.enums import AllocationPolicy, SolverStatus
from fabrid.domain.identifiers import AttackSubtypeId, ClientId, FailureReason
from fabrid.domain.provenance import ExperimentProvenance
from fabrid.domain.values import (
    BudgetUsageRatio,
    ClientWeight,
    FalseAlertCount,
    FalsePositiveRate,
    MacroRecall,
    Probability,
    RowCount,
    TargetFalsePositiveRate,
    Threshold,
    TruePositiveRate,
    WorstClientRecall,
)


@dataclass(frozen=True, slots=True)
class ConfusionCounts:
    true_positive: RowCount
    false_negative: RowCount
    false_positive: RowCount
    true_negative: RowCount


@dataclass(frozen=True, slots=True)
class DetectionMetrics:
    false_positive_rate: FalsePositiveRate
    true_positive_rate: TruePositiveRate
    macro_attack_recall: MacroRecall
    false_alert_count: FalseAlertCount


@dataclass(frozen=True, slots=True)
class ClientWeights:
    nominal: ClientWeight
    realized: ClientWeight


@dataclass(frozen=True, slots=True)
class CalibrationOutcome:
    target_rate: TargetFalsePositiveRate
    threshold: Threshold
    calibration_count: RowCount


@dataclass(frozen=True, slots=True)
class ClientResultRecord:
    allocation: AllocationCoordinate
    client_id: ClientId
    calibration: CalibrationOutcome
    weights: ClientWeights
    benign_test_count: RowCount
    attack_test_count: RowCount
    attack_subtype: AttackSubtypeId | None
    confusion: ConfusionCounts
    metrics: DetectionMetrics
    solver: SolverEvidence
    provenance: ExperimentProvenance


@dataclass(frozen=True, slots=True)
class CompletedPolicyEvaluation:
    policy: AllocationPolicy
    macro_recall: MacroRecall
    worst_client_recall: WorstClientRecall
    federation_fpr: FalsePositiveRate
    budget_usage: BudgetUsageRatio | None
    fallback_rate: Probability


@dataclass(frozen=True, slots=True)
class ExcludedPolicyEvaluation:
    policy: AllocationPolicy
    status: SolverStatus
    reason: FailureReason

    def __post_init__(self) -> None:
        if self.status is SolverStatus.OPTIMAL:
            raise ValueError("excluded policy evaluation may not have optimal solver status")


PolicyEvaluation = CompletedPolicyEvaluation | ExcludedPolicyEvaluation


@dataclass(frozen=True, slots=True)
class SeedBudgetEvaluation:
    experiment: ExperimentCoordinate
    policies: tuple[PolicyEvaluation, ...]

    def __post_init__(self) -> None:
        policy_ids = tuple(policy.policy for policy in self.policies)
        if len(set(policy_ids)) != len(policy_ids):
            raise ValueError("seed-budget evaluation contains duplicate policies")

    def policy(self, policy: AllocationPolicy) -> PolicyEvaluation:
        for evaluation in self.policies:
            if evaluation.policy is policy:
                return evaluation
        raise KeyError(policy.value)
