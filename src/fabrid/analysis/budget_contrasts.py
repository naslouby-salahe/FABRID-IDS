from __future__ import annotations

from dataclasses import dataclass

from fabrid.analysis.contrasts import Contrast, build_contrast
from fabrid.domain.enums import AllocationPolicy, BudgetId, EvidenceAvailability, MetricId
from fabrid.domain.identifiers import FailureReason
from fabrid.domain.values import AnalysisSeed, FalsePositiveBudget, RowCount
from fabrid.evaluation.results import CompletedPolicyEvaluation, SeedBudgetEvaluation
from fabrid.protocol.models import BudgetLevel, FabridProtocol


@dataclass(frozen=True, slots=True)
class AvailableBudgetContrast:
    budget_id: BudgetId
    budget: FalsePositiveBudget
    availability: EvidenceAvailability
    contrast: Contrast

    def __post_init__(self) -> None:
        if self.availability is not EvidenceAvailability.AVAILABLE:
            raise ValueError("available budget contrast must be marked AVAILABLE")


@dataclass(frozen=True, slots=True)
class UnavailableBudgetContrast:
    budget_id: BudgetId
    budget: FalsePositiveBudget
    availability: EvidenceAvailability
    reason: FailureReason

    def __post_init__(self) -> None:
        if self.availability is not EvidenceAvailability.INSUFFICIENT_EVIDENCE:
            raise ValueError(
                "unavailable budget contrast must be marked INSUFFICIENT_EVIDENCE"
            )


BudgetContrast = AvailableBudgetContrast | UnavailableBudgetContrast


def budget_results(
    evaluations: tuple[SeedBudgetEvaluation, ...],
    level: BudgetLevel,
) -> tuple[SeedBudgetEvaluation, ...]:
    return tuple(
        evaluation
        for evaluation in evaluations
        if evaluation.experiment.budget_id is level.budget_id
    )


def paired_seed_count(
    evaluations: tuple[SeedBudgetEvaluation, ...],
    treatment: AllocationPolicy,
    baseline: AllocationPolicy,
) -> RowCount:
    paired = 0
    for evaluation in evaluations:
        treatment_result = evaluation.policy(treatment)
        baseline_result = evaluation.policy(baseline)
        if isinstance(treatment_result, CompletedPolicyEvaluation) and isinstance(
            baseline_result,
            CompletedPolicyEvaluation,
        ):
            paired += 1
    return RowCount(paired)


def analyze_budget_contrast(
    evaluations: tuple[SeedBudgetEvaluation, ...],
    level: BudgetLevel,
    treatment: AllocationPolicy,
    baseline: AllocationPolicy,
    metric: MetricId,
    protocol: FabridProtocol,
    analysis_seed: AnalysisSeed,
) -> BudgetContrast:
    results = budget_results(evaluations, level)
    if paired_seed_count(results, treatment, baseline).value == 0:
        return UnavailableBudgetContrast(
            budget_id=level.budget_id,
            budget=level.value,
            availability=EvidenceAvailability.INSUFFICIENT_EVIDENCE,
            reason=FailureReason(
                f"no paired seed evidence for {treatment.value} vs {baseline.value}"
            ),
        )
    return AvailableBudgetContrast(
        budget_id=level.budget_id,
        budget=level.value,
        availability=EvidenceAvailability.AVAILABLE,
        contrast=build_contrast(
            results=results,
            treatment=treatment,
            baseline=baseline,
            metric=metric,
            bootstrap_resamples=protocol.statistics.bootstrap_resamples,
            bootstrap_seed=analysis_seed,
            bootstrap_confidence=protocol.statistics.bootstrap_confidence,
        ),
    )
