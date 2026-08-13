from __future__ import annotations

from dataclasses import dataclass

from fabrid.analysis.bootstrap import BootstrapResult, paired_bootstrap_ci
from fabrid.analysis.sign_flip import SignFlipResult, exact_sign_flip_test
from fabrid.domain.enums import AllocationPolicy, MetricId
from fabrid.domain.values import (
    AnalysisSeed,
    DetectorSeed,
    MetricDifference,
    Probability,
    RowCount,
)
from fabrid.evaluation.results import (
    CompletedPolicyEvaluation,
    ExcludedPolicyEvaluation,
    SeedBudgetEvaluation,
)


@dataclass(frozen=True, slots=True)
class Contrast:
    treatment: AllocationPolicy
    baseline: AllocationPolicy
    metric: MetricId
    paired_differences: tuple[MetricDifference, ...]
    included_seeds: tuple[DetectorSeed, ...]
    excluded_seeds: tuple[DetectorSeed, ...]
    sign_flip: SignFlipResult
    bootstrap: BootstrapResult


def _metric_value(
    evaluation: CompletedPolicyEvaluation,
    metric: MetricId,
) -> float | None:
    if metric is MetricId.MACRO_RECALL:
        return evaluation.macro_recall.value
    if metric is MetricId.WORST_CLIENT_RECALL:
        return evaluation.worst_client_recall.value
    if metric is MetricId.FEDERATION_FPR:
        return evaluation.federation_fpr.value
    if metric is MetricId.BUDGET_USAGE_RATIO:
        return None if evaluation.budget_usage is None else evaluation.budget_usage.value
    raise ValueError(f"unsupported contrast metric {metric.value}")


def build_contrast(
    results: tuple[SeedBudgetEvaluation, ...],
    treatment: AllocationPolicy,
    baseline: AllocationPolicy,
    metric: MetricId,
    bootstrap_resamples: RowCount,
    bootstrap_seed: AnalysisSeed,
    bootstrap_confidence: Probability,
) -> Contrast:
    differences: list[MetricDifference] = []
    included: list[DetectorSeed] = []
    excluded: list[DetectorSeed] = []

    for result in results:
        seed = result.experiment.detector_seed
        treatment_result = result.policy(treatment)
        baseline_result = result.policy(baseline)
        if isinstance(treatment_result, ExcludedPolicyEvaluation) or isinstance(
            baseline_result, ExcludedPolicyEvaluation
        ):
            excluded.append(seed)
            continue

        treatment_value = _metric_value(treatment_result, metric)
        baseline_value = _metric_value(baseline_result, metric)
        if treatment_value is None or baseline_value is None:
            excluded.append(seed)
            continue

        differences.append(MetricDifference(treatment_value - baseline_value))
        included.append(seed)

    if not differences:
        raise ValueError(
            f"no seed has paired {treatment.value} and {baseline.value} values for {metric.value}"
        )

    paired = tuple(differences)
    return Contrast(
        treatment=treatment,
        baseline=baseline,
        metric=metric,
        paired_differences=paired,
        included_seeds=tuple(included),
        excluded_seeds=tuple(excluded),
        sign_flip=exact_sign_flip_test(paired),
        bootstrap=paired_bootstrap_ci(
            paired_differences=paired,
            resamples=bootstrap_resamples,
            seed=bootstrap_seed,
            confidence=bootstrap_confidence,
        ),
    )
