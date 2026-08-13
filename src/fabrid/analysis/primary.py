from __future__ import annotations

from dataclasses import dataclass

from fabrid.analysis.contrasts import Contrast, build_contrast
from fabrid.analysis.multiplicity import HolmResult, holm_correction
from fabrid.domain.enums import (
    AllocationPolicy,
    BudgetId,
    EvidenceAvailability,
    ExperimentId,
    ExperimentVariantId,
    MetricId,
    PrimaryContrastId,
)
from fabrid.domain.identifiers import FailureReason
from fabrid.domain.values import AnalysisSeed, FalsePositiveBudget, RowCount
from fabrid.evaluation.results import (
    CompletedPolicyEvaluation,
    SeedBudgetEvaluation,
)
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


@dataclass(frozen=True, slots=True)
class BudgetHolmResult:
    budget_id: BudgetId
    result: HolmResult


@dataclass(frozen=True, slots=True)
class PrimaryContrastFamily:
    contrast_id: PrimaryContrastId
    treatment: AllocationPolicy
    baseline: AllocationPolicy
    metric: MetricId
    availability: EvidenceAvailability
    budgets: tuple[BudgetContrast, ...]
    holm: tuple[BudgetHolmResult, ...] | None


@dataclass(frozen=True, slots=True)
class PrimaryInference:
    macro_recall: PrimaryContrastFamily
    worst_client_recall: PrimaryContrastFamily


def _validate_primary_evaluations(
    evaluations: tuple[SeedBudgetEvaluation, ...],
) -> None:
    if not evaluations:
        raise ValueError("primary inference requires matched-budget evaluations")
    for evaluation in evaluations:
        experiment = evaluation.experiment
        if experiment.experiment_id is not ExperimentId.MATCHED_BUDGET:
            raise ValueError("primary inference may use only MATCHED_BUDGET evaluations")
        if experiment.variant_id is not ExperimentVariantId.PRIMARY:
            raise ValueError("primary inference may use only the PRIMARY experiment variant")


def _budget_results(
    evaluations: tuple[SeedBudgetEvaluation, ...],
    level: BudgetLevel,
) -> tuple[SeedBudgetEvaluation, ...]:
    return tuple(
        evaluation
        for evaluation in evaluations
        if evaluation.experiment.budget_id is level.budget_id
    )


def _paired_seed_count(
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


def _budget_contrast(
    evaluations: tuple[SeedBudgetEvaluation, ...],
    level: BudgetLevel,
    treatment: AllocationPolicy,
    baseline: AllocationPolicy,
    metric: MetricId,
    protocol: FabridProtocol,
    analysis_seed: AnalysisSeed,
) -> BudgetContrast:
    results = _budget_results(evaluations, level)
    paired_count = _paired_seed_count(results, treatment, baseline)
    if paired_count.value == 0:
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


def _family(
    evaluations: tuple[SeedBudgetEvaluation, ...],
    protocol: FabridProtocol,
    contrast_id: PrimaryContrastId,
    treatment: AllocationPolicy,
    baseline: AllocationPolicy,
    metric: MetricId,
) -> PrimaryContrastFamily:
    budgets = tuple(
        _budget_contrast(
            evaluations=evaluations,
            level=level,
            treatment=treatment,
            baseline=baseline,
            metric=metric,
            protocol=protocol,
            analysis_seed=AnalysisSeed(index),
        )
        for index, level in enumerate(protocol.budgets)
    )
    available = tuple(
        budget for budget in budgets if isinstance(budget, AvailableBudgetContrast)
    )
    if len(available) != protocol.statistics.holm_family_size.value:
        return PrimaryContrastFamily(
            contrast_id=contrast_id,
            treatment=treatment,
            baseline=baseline,
            metric=metric,
            availability=EvidenceAvailability.INSUFFICIENT_EVIDENCE,
            budgets=budgets,
            holm=None,
        )

    corrected = holm_correction(
        tuple(budget.contrast.sign_flip.p_value for budget in available),
        protocol.statistics.significance,
    )
    return PrimaryContrastFamily(
        contrast_id=contrast_id,
        treatment=treatment,
        baseline=baseline,
        metric=metric,
        availability=EvidenceAvailability.AVAILABLE,
        budgets=budgets,
        holm=tuple(
            BudgetHolmResult(
                budget_id=budget.budget_id,
                result=result,
            )
            for budget, result in zip(available, corrected, strict=True)
        ),
    )


def analyze_primary_inference(
    evaluations: tuple[SeedBudgetEvaluation, ...],
    protocol: FabridProtocol,
) -> PrimaryInference:
    _validate_primary_evaluations(evaluations)
    return PrimaryInference(
        macro_recall=_family(
            evaluations=evaluations,
            protocol=protocol,
            contrast_id=PrimaryContrastId.FABRID_MACRO_VS_EQ_FPR_MACRO_RECALL,
            treatment=AllocationPolicy.FABRID_MACRO,
            baseline=AllocationPolicy.EQ_FPR,
            metric=MetricId.MACRO_RECALL,
        ),
        worst_client_recall=_family(
            evaluations=evaluations,
            protocol=protocol,
            contrast_id=(
                PrimaryContrastId.FABRID_MINIMAX_VS_EQ_FPR_WORST_CLIENT_RECALL
            ),
            treatment=AllocationPolicy.FABRID_MINIMAX,
            baseline=AllocationPolicy.EQ_FPR,
            metric=MetricId.WORST_CLIENT_RECALL,
        ),
    )
