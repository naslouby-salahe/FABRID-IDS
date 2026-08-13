from __future__ import annotations

from dataclasses import dataclass

from fabrid.analysis.budget_contrasts import (
    AvailableBudgetContrast,
    BudgetContrast,
    analyze_budget_contrast,
)
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
from fabrid.domain.values import AnalysisSeed
from fabrid.evaluation.results import SeedBudgetEvaluation
from fabrid.protocol.models import FabridProtocol


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


def _family(
    evaluations: tuple[SeedBudgetEvaluation, ...],
    protocol: FabridProtocol,
    contrast_id: PrimaryContrastId,
    treatment: AllocationPolicy,
    baseline: AllocationPolicy,
    metric: MetricId,
) -> PrimaryContrastFamily:
    budgets = tuple(
        analyze_budget_contrast(
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
