from __future__ import annotations

from fabrid.domain.coordinates import ExperimentCoordinate
from fabrid.domain.enums import (
    AllocationPolicy,
    BudgetId,
    DatasetId,
    EvidenceAvailability,
    ExperimentId,
    ExperimentVariantId,
    SolverStatus,
    WeightMode,
)
from fabrid.domain.identifiers import CampaignId, FailureReason
from fabrid.domain.values import (
    BudgetUsageRatio,
    DetectorSeed,
    FalsePositiveBudget,
    FalsePositiveRate,
    MacroRecall,
    Probability,
    WorstClientRecall,
)
from fabrid.evaluation.results import (
    CompletedPolicyEvaluation,
    ExcludedPolicyEvaluation,
    SeedBudgetEvaluation,
)
from fabrid.reporting.tables import (
    AvailableExperimentSummaryRow,
    UnavailableExperimentSummaryRow,
    build_experiment_summary_table,
)


def _coordinate(seed: int) -> ExperimentCoordinate:
    return ExperimentCoordinate(
        campaign_id=CampaignId("test-campaign"),
        experiment_id=ExperimentId.MATCHED_BUDGET,
        variant_id=ExperimentVariantId.PRIMARY,
        dataset_id=DatasetId.NBAIOT,
        detector_seed=DetectorSeed(seed),
        budget_id=BudgetId.FALSE_POSITIVE_0P010,
        budget=FalsePositiveBudget(0.01),
        weight_mode=WeightMode.EQUAL_CLIENT,
    )


def _completed(seed: int, macro: float, worst: float, bur: float) -> SeedBudgetEvaluation:
    return SeedBudgetEvaluation(
        experiment=_coordinate(seed),
        policies=(
            CompletedPolicyEvaluation(
                policy=AllocationPolicy.EQ_FPR,
                macro_recall=MacroRecall(macro),
                worst_client_recall=WorstClientRecall(worst),
                federation_fpr=FalsePositiveRate(0.01),
                budget_usage=BudgetUsageRatio(bur),
                fallback_rate=Probability(0.0),
            ),
        ),
    )


def test_experiment_summary_averages_completed_seed_evidence() -> None:
    rows = build_experiment_summary_table(
        (
            _completed(0, macro=0.8, worst=0.6, bur=1.0),
            _completed(1, macro=0.6, worst=0.4, bur=1.2),
        )
    )

    assert len(rows) == 1
    row = rows[0]
    assert isinstance(row, AvailableExperimentSummaryRow)
    assert row.availability is EvidenceAvailability.AVAILABLE
    assert row.seed_count.value == 2
    assert row.macro_recall.value == 0.7
    assert row.worst_client_recall.value == 0.5
    assert row.budget_usage is not None
    assert row.budget_usage.value == 1.1


def test_experiment_summary_preserves_solver_exclusion_as_unavailable() -> None:
    evaluation = SeedBudgetEvaluation(
        experiment=_coordinate(0),
        policies=(
            ExcludedPolicyEvaluation(
                policy=AllocationPolicy.FABRID_MACRO,
                status=SolverStatus.SOLVER_INVALID,
                reason=FailureReason("synthetic solver exclusion"),
            ),
        ),
    )

    rows = build_experiment_summary_table((evaluation,))

    assert len(rows) == 1
    row = rows[0]
    assert isinstance(row, UnavailableExperimentSummaryRow)
    assert row.availability is EvidenceAvailability.INSUFFICIENT_EVIDENCE
