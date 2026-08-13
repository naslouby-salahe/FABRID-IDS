from __future__ import annotations

from dataclasses import dataclass

from fabrid.domain.enums import (
    AllocationPolicy,
    BudgetId,
    EvidenceAvailability,
    ExperimentId,
    ExperimentVariantId,
)
from fabrid.domain.identifiers import FailureReason
from fabrid.domain.values import (
    BudgetUsageRatio,
    FalsePositiveBudget,
    FalsePositiveRate,
    MacroRecall,
    Probability,
    RowCount,
    WorstClientRecall,
)
from fabrid.evaluation.results import CompletedPolicyEvaluation, SeedBudgetEvaluation


@dataclass(frozen=True, slots=True)
class AvailableExperimentSummaryRow:
    experiment_id: ExperimentId
    variant_id: ExperimentVariantId
    budget_id: BudgetId
    budget: FalsePositiveBudget
    policy: AllocationPolicy
    availability: EvidenceAvailability
    seed_count: RowCount
    macro_recall: MacroRecall
    worst_client_recall: WorstClientRecall
    federation_fpr: FalsePositiveRate
    budget_usage: BudgetUsageRatio | None
    fallback_rate: Probability

    def __post_init__(self) -> None:
        if self.availability is not EvidenceAvailability.AVAILABLE:
            raise ValueError("available experiment summary row must be marked AVAILABLE")


@dataclass(frozen=True, slots=True)
class UnavailableExperimentSummaryRow:
    experiment_id: ExperimentId
    variant_id: ExperimentVariantId
    budget_id: BudgetId
    budget: FalsePositiveBudget
    policy: AllocationPolicy
    availability: EvidenceAvailability
    reason: FailureReason

    def __post_init__(self) -> None:
        if self.availability is not EvidenceAvailability.INSUFFICIENT_EVIDENCE:
            raise ValueError(
                "unavailable experiment summary row must be marked INSUFFICIENT_EVIDENCE"
            )


ExperimentSummaryRow = AvailableExperimentSummaryRow | UnavailableExperimentSummaryRow


def _mean(values: tuple[float, ...]) -> float:
    if not values:
        raise ValueError("mean requires at least one value")
    return sum(values) / len(values)


def _completed_results(
    evaluations: tuple[SeedBudgetEvaluation, ...],
    variant_id: ExperimentVariantId,
    budget_id: BudgetId,
    policy: AllocationPolicy,
) -> tuple[CompletedPolicyEvaluation, ...]:
    results: list[CompletedPolicyEvaluation] = []
    for evaluation in evaluations:
        experiment = evaluation.experiment
        if experiment.variant_id is not variant_id or experiment.budget_id is not budget_id:
            continue
        candidate = evaluation.policy(policy)
        if isinstance(candidate, CompletedPolicyEvaluation):
            results.append(candidate)
    return tuple(results)


def _coordinate_evaluation(
    evaluations: tuple[SeedBudgetEvaluation, ...],
    variant_id: ExperimentVariantId,
    budget_id: BudgetId,
) -> SeedBudgetEvaluation:
    for evaluation in evaluations:
        experiment = evaluation.experiment
        if experiment.variant_id is variant_id and experiment.budget_id is budget_id:
            return evaluation
    raise KeyError(f"missing evaluation coordinate {variant_id.value}/{budget_id.value}")


def _available_row(
    coordinate: SeedBudgetEvaluation,
    policy: AllocationPolicy,
    results: tuple[CompletedPolicyEvaluation, ...],
) -> AvailableExperimentSummaryRow:
    budget_usages = tuple(
        result.budget_usage.value
        for result in results
        if result.budget_usage is not None
    )
    return AvailableExperimentSummaryRow(
        experiment_id=coordinate.experiment.experiment_id,
        variant_id=coordinate.experiment.variant_id,
        budget_id=coordinate.experiment.budget_id,
        budget=coordinate.experiment.budget,
        policy=policy,
        availability=EvidenceAvailability.AVAILABLE,
        seed_count=RowCount(len(results)),
        macro_recall=MacroRecall(_mean(tuple(result.macro_recall.value for result in results))),
        worst_client_recall=WorstClientRecall(
            _mean(tuple(result.worst_client_recall.value for result in results))
        ),
        federation_fpr=FalsePositiveRate(
            _mean(tuple(result.federation_fpr.value for result in results))
        ),
        budget_usage=(
            None if not budget_usages else BudgetUsageRatio(_mean(budget_usages))
        ),
        fallback_rate=Probability(
            _mean(tuple(result.fallback_rate.value for result in results))
        ),
    )


def build_experiment_summary_table(
    evaluations: tuple[SeedBudgetEvaluation, ...],
) -> tuple[ExperimentSummaryRow, ...]:
    if not evaluations:
        raise ValueError("experiment summary table requires evaluations")

    experiment_ids = {evaluation.experiment.experiment_id for evaluation in evaluations}
    if len(experiment_ids) != 1:
        raise ValueError("experiment summary table may contain only one experiment identity")

    variant_ids = tuple(
        sorted(
            {evaluation.experiment.variant_id for evaluation in evaluations},
            key=lambda variant: variant.value,
        )
    )
    budget_ids = tuple(
        sorted(
            {evaluation.experiment.budget_id for evaluation in evaluations},
            key=lambda budget: budget.value,
        )
    )
    policies = tuple(
        sorted(
            {policy.policy for evaluation in evaluations for policy in evaluation.policies},
            key=lambda policy: policy.value,
        )
    )

    rows: list[ExperimentSummaryRow] = []
    for variant_id in variant_ids:
        for budget_id in budget_ids:
            coordinate = _coordinate_evaluation(evaluations, variant_id, budget_id)
            for policy in policies:
                results = _completed_results(evaluations, variant_id, budget_id, policy)
                if results:
                    rows.append(_available_row(coordinate, policy, results))
                else:
                    rows.append(
                        UnavailableExperimentSummaryRow(
                            experiment_id=coordinate.experiment.experiment_id,
                            variant_id=variant_id,
                            budget_id=budget_id,
                            budget=coordinate.experiment.budget,
                            policy=policy,
                            availability=EvidenceAvailability.INSUFFICIENT_EVIDENCE,
                            reason=FailureReason(
                                "no completed seed evaluation for this policy coordinate"
                            ),
                        )
                    )
    return tuple(rows)
