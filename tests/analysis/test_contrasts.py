from __future__ import annotations

import pytest

from fabrid.analysis.contrasts import build_contrast
from fabrid.domain.coordinates import ExperimentCoordinate
from fabrid.domain.enums import (
    AllocationPolicy,
    BudgetId,
    DatasetId,
    ExperimentId,
    ExperimentVariantId,
    MetricId,
    SolverStatus,
    WeightMode,
)
from fabrid.domain.identifiers import CampaignId, FailureReason
from fabrid.domain.values import (
    AnalysisSeed,
    BudgetUsageRatio,
    DetectorSeed,
    FalsePositiveBudget,
    FalsePositiveRate,
    MacroRecall,
    Probability,
    RowCount,
    WorstClientRecall,
)
from fabrid.evaluation.results import (
    CompletedPolicyEvaluation,
    ExcludedPolicyEvaluation,
    PolicyEvaluation,
    SeedBudgetEvaluation,
)


def _coordinate(seed: DetectorSeed) -> ExperimentCoordinate:
    return ExperimentCoordinate(
        campaign_id=CampaignId("test-campaign"),
        experiment_id=ExperimentId.MATCHED_BUDGET,
        variant_id=ExperimentVariantId.PRIMARY,
        dataset_id=DatasetId.NBAIOT,
        detector_seed=seed,
        budget_id=BudgetId.FALSE_POSITIVE_0P010,
        budget=FalsePositiveBudget(0.01),
        weight_mode=WeightMode.EQUAL_CLIENT,
    )


def _completed(
    policy: AllocationPolicy,
    macro_recall: MacroRecall,
    worst_client_recall: WorstClientRecall,
) -> CompletedPolicyEvaluation:
    return CompletedPolicyEvaluation(
        policy=policy,
        macro_recall=macro_recall,
        worst_client_recall=worst_client_recall,
        federation_fpr=FalsePositiveRate(0.01),
        budget_usage=BudgetUsageRatio(1.0),
        fallback_rate=Probability(0.0),
    )


def _evaluation(
    seed: DetectorSeed,
    policies: tuple[PolicyEvaluation, ...],
) -> SeedBudgetEvaluation:
    return SeedBudgetEvaluation(experiment=_coordinate(seed), policies=policies)


def test_macro_recall_contrast_pairs_available_detector_seeds() -> None:
    first = _evaluation(
        DetectorSeed(0),
        (
            _completed(AllocationPolicy.EQ_FPR, MacroRecall(0.5), WorstClientRecall(0.4)),
            _completed(AllocationPolicy.FABRID_MACRO, MacroRecall(0.6), WorstClientRecall(0.45)),
        ),
    )
    second = _evaluation(
        DetectorSeed(1),
        (
            _completed(AllocationPolicy.EQ_FPR, MacroRecall(0.5), WorstClientRecall(0.4)),
            _completed(AllocationPolicy.FABRID_MACRO, MacroRecall(0.55), WorstClientRecall(0.42)),
        ),
    )

    contrast = build_contrast(
        results=(first, second),
        treatment=AllocationPolicy.FABRID_MACRO,
        baseline=AllocationPolicy.EQ_FPR,
        metric=MetricId.MACRO_RECALL,
        bootstrap_resamples=RowCount(1_000),
        bootstrap_seed=AnalysisSeed(0),
        bootstrap_confidence=Probability(0.95),
    )

    assert tuple(value.value for value in contrast.paired_differences) == pytest.approx((0.1, 0.05))
    assert contrast.included_seeds == (DetectorSeed(0), DetectorSeed(1))
    assert contrast.excluded_seeds == ()
    assert contrast.sign_flip.enumerated_assignments == RowCount(4)


def test_solver_invalid_treatment_is_explicitly_excluded() -> None:
    available = _evaluation(
        DetectorSeed(0),
        (
            _completed(AllocationPolicy.EQ_FPR, MacroRecall(0.5), WorstClientRecall(0.4)),
            _completed(AllocationPolicy.FABRID_MACRO, MacroRecall(0.6), WorstClientRecall(0.45)),
        ),
    )
    excluded = _evaluation(
        DetectorSeed(1),
        (
            _completed(AllocationPolicy.EQ_FPR, MacroRecall(0.5), WorstClientRecall(0.4)),
            ExcludedPolicyEvaluation(
                policy=AllocationPolicy.FABRID_MACRO,
                status=SolverStatus.SOLVER_INVALID,
                reason=FailureReason("solver-invalid"),
            ),
        ),
    )

    contrast = build_contrast(
        results=(available, excluded),
        treatment=AllocationPolicy.FABRID_MACRO,
        baseline=AllocationPolicy.EQ_FPR,
        metric=MetricId.MACRO_RECALL,
        bootstrap_resamples=RowCount(1_000),
        bootstrap_seed=AnalysisSeed(0),
        bootstrap_confidence=Probability(0.95),
    )

    assert contrast.included_seeds == (DetectorSeed(0),)
    assert contrast.excluded_seeds == (DetectorSeed(1),)
    assert len(contrast.paired_differences) == 1
