from __future__ import annotations

import itertools
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from fabrid.config import (
    TIGHT_TOLERANCE,
    AllocationPolicy,
    AnalysisSeed,
    BudgetId,
    BudgetLevel,
    DetectorSeed,
    ExperimentalUnit,
    MetricDifference,
    NonNegativeInt,
    Probability,
    RowCount,
    StatisticsConfig,
)
from fabrid.evaluation.metrics import (
    CompletedPolicyEvaluation,
    ExcludedPolicyEvaluation,
    MetricId,
    SeedBudgetEvaluation,
)

HypothesisIndex = NonNegativeInt


class HypothesisDecision(StrEnum):
    REJECT = "reject"
    RETAIN = "retain"


@dataclass(frozen=True, slots=True)
class SignFlipResult:
    observed_mean_difference: MetricDifference
    p_value: Probability
    enumerated_assignments: RowCount


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    mean_difference: MetricDifference
    median_difference: MetricDifference
    confidence_interval_low: MetricDifference
    confidence_interval_high: MetricDifference
    confidence: Probability
    resamples: RowCount


@dataclass(frozen=True, slots=True)
class HolmResult:
    index: HypothesisIndex
    p_value: Probability
    adjusted_p_value: Probability
    decision: HypothesisDecision


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


@dataclass(frozen=True, slots=True)
class BudgetContrast:
    budget_id: BudgetId
    contrast: Contrast


@dataclass(frozen=True, slots=True)
class PrimaryInference:
    macro_recall: tuple[BudgetContrast, ...]
    worst_client_recall: tuple[BudgetContrast, ...]

    def holm_macro(self, alpha: Probability) -> tuple[HolmResult, ...]:
        return holm_correction(
            tuple(
                budget_contrast.contrast.sign_flip.p_value for budget_contrast in self.macro_recall
            ),
            alpha,
        )

    def holm_worst_client(self, alpha: Probability) -> tuple[HolmResult, ...]:
        return holm_correction(
            tuple(
                budget_contrast.contrast.sign_flip.p_value
                for budget_contrast in self.worst_client_recall
            ),
            alpha,
        )


def exact_sign_flip_test(
    paired_differences: tuple[MetricDifference, ...],
) -> SignFlipResult:
    if not paired_differences:
        raise ValueError("exact sign-flip test requires paired differences")
    observed_mean = sum(paired_differences) / len(paired_differences)
    observed_absolute = abs(observed_mean)
    extreme_count = 0
    assignment_count = 0
    for signs in itertools.product((1.0, -1.0), repeat=len(paired_differences)):
        flipped_mean = sum(
            sign * difference for sign, difference in zip(signs, paired_differences, strict=True)
        ) / len(paired_differences)
        if abs(flipped_mean) >= observed_absolute - TIGHT_TOLERANCE:
            extreme_count += 1
        assignment_count += 1
    return SignFlipResult(
        observed_mean_difference=observed_mean,
        p_value=extreme_count / assignment_count,
        enumerated_assignments=assignment_count,
    )


def paired_bootstrap_ci(
    paired_differences: tuple[MetricDifference, ...],
    statistics: StatisticsConfig,
    seed: AnalysisSeed,
) -> BootstrapResult:
    if not paired_differences:
        raise ValueError("paired bootstrap requires at least one paired difference")
    resamples = statistics.bootstrap_resamples
    confidence = statistics.bootstrap_confidence
    if resamples < 1:
        raise ValueError("paired bootstrap requires at least one resample")
    if not 0.0 < confidence < 1.0:
        raise ValueError("bootstrap confidence must be strictly between zero and one")
    raw = np.asarray(tuple(paired_differences), dtype=np.float64)
    rng = np.random.default_rng(seed)
    resample_indices = rng.integers(0, raw.shape[0], size=(resamples, raw.shape[0]))
    resample_means: np.ndarray = raw[resample_indices].mean(axis=1)
    lower_quantile = (1.0 - confidence) / 2.0
    upper_quantile = 1.0 - lower_quantile
    return BootstrapResult(
        mean_difference=float(np.mean(raw)),
        median_difference=float(np.median(raw)),
        confidence_interval_low=float(np.quantile(resample_means, lower_quantile)),
        confidence_interval_high=float(np.quantile(resample_means, upper_quantile)),
        confidence=confidence,
        resamples=resamples,
    )


def holm_correction(
    p_values: tuple[Probability, ...],
    alpha: Probability,
) -> tuple[HolmResult, ...]:
    if not p_values:
        raise ValueError("Holm correction requires at least one p-value")
    if not 0.0 < alpha < 1.0:
        raise ValueError("Holm alpha must be strictly between zero and one")
    family_size = len(p_values)
    ordered_indices = tuple(sorted(range(family_size), key=lambda index: p_values[index]))
    adjusted_values = [0.0] * family_size
    running_maximum = 0.0
    for rank, index in enumerate(ordered_indices):
        multiplier = family_size - rank
        candidate = min(1.0, p_values[index] * multiplier)
        running_maximum = max(running_maximum, candidate)
        adjusted_values[index] = running_maximum
    decisions = [HypothesisDecision.RETAIN] * family_size
    still_rejecting = True
    for rank, index in enumerate(ordered_indices):
        threshold = alpha / (family_size - rank)
        still_rejecting = still_rejecting and p_values[index] <= threshold
        decisions[index] = (
            HypothesisDecision.REJECT if still_rejecting else HypothesisDecision.RETAIN
        )
    return tuple(
        HolmResult(
            index=index,
            p_value=p_values[index],
            adjusted_p_value=adjusted_values[index],
            decision=decisions[index],
        )
        for index in range(family_size)
    )


def _metric_value(
    evaluation: CompletedPolicyEvaluation,
    metric: MetricId,
) -> Probability | None:
    if metric is MetricId.MACRO_RECALL:
        return evaluation.macro_recall
    if metric is MetricId.WORST_CLIENT_RECALL:
        return evaluation.worst_client_recall
    if metric is MetricId.FEDERATION_FPR:
        return evaluation.federation_fpr
    if metric is MetricId.BUDGET_USAGE_RATIO:
        return evaluation.budget_usage
    raise ValueError(f"unsupported contrast metric {metric.value}")


def build_contrast(
    evaluations: tuple[SeedBudgetEvaluation, ...],
    treatment: AllocationPolicy,
    baseline: AllocationPolicy,
    metric: MetricId,
    statistics: StatisticsConfig,
    bootstrap_seed: AnalysisSeed,
) -> Contrast:
    differences: list[MetricDifference] = []
    included: list[DetectorSeed] = []
    excluded: list[DetectorSeed] = []
    for evaluation in evaluations:
        seed = evaluation.experiment.detector_seed
        treatment_result = evaluation.policy(treatment)
        baseline_result = evaluation.policy(baseline)
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
        differences.append(treatment_value - baseline_value)
        included.append(seed)
    if not differences:
        raise ValueError(
            f"no seed has paired {treatment.value} and {baseline.value} values for {metric.value}"
        )
    if 2 ** len(differences) > statistics.sign_flip_enumeration:
        raise ValueError("exact sign-flip enumeration exceeds the configured bound")
    paired = tuple(differences)
    return Contrast(
        treatment=treatment,
        baseline=baseline,
        metric=metric,
        paired_differences=paired,
        included_seeds=tuple(included),
        excluded_seeds=tuple(excluded),
        sign_flip=exact_sign_flip_test(paired),
        bootstrap=paired_bootstrap_ci(paired, statistics, bootstrap_seed),
    )


def analyze_budget_contrast(
    evaluations: tuple[SeedBudgetEvaluation, ...],
    level: BudgetLevel,
    treatment: AllocationPolicy,
    baseline: AllocationPolicy,
    metric: MetricId,
    statistics: StatisticsConfig,
    analysis_seed: AnalysisSeed,
) -> BudgetContrast:
    budget_evaluations = tuple(
        evaluation
        for evaluation in evaluations
        if evaluation.experiment.budget_id is level.budget_id
    )
    if statistics.experimental_unit is not ExperimentalUnit.DETECTOR_SEED:
        raise ValueError("paired contrasts require the detector seed as the experimental unit")
    return BudgetContrast(
        budget_id=level.budget_id,
        contrast=build_contrast(
            evaluations=budget_evaluations,
            treatment=treatment,
            baseline=baseline,
            metric=metric,
            statistics=statistics,
            bootstrap_seed=analysis_seed,
        ),
    )


def analyze_primary_inference(
    evaluations: tuple[SeedBudgetEvaluation, ...],
    budgets: tuple[BudgetLevel, ...],
    statistics: StatisticsConfig,
) -> PrimaryInference:
    if statistics.holm_family_size != len(budgets):
        raise ValueError("Holm family size must match the number of primary budgets")
    macro_recall = tuple(
        analyze_budget_contrast(
            evaluations=evaluations,
            level=level,
            treatment=AllocationPolicy.FABRID_MACRO,
            baseline=AllocationPolicy.EQ_FPR,
            metric=MetricId.MACRO_RECALL,
            statistics=statistics,
            analysis_seed=budget_index,
        )
        for budget_index, level in enumerate(budgets)
    )
    worst_client_recall = tuple(
        analyze_budget_contrast(
            evaluations=evaluations,
            level=level,
            treatment=AllocationPolicy.FABRID_MINIMAX,
            baseline=AllocationPolicy.EQ_FPR,
            metric=MetricId.WORST_CLIENT_RECALL,
            statistics=statistics,
            analysis_seed=budget_index,
        )
        for budget_index, level in enumerate(budgets)
    )
    return PrimaryInference(
        macro_recall=macro_recall,
        worst_client_recall=worst_client_recall,
    )


def analyze_minimax_macro_guardrail(
    evaluations: tuple[SeedBudgetEvaluation, ...],
    budgets: tuple[BudgetLevel, ...],
    statistics: StatisticsConfig,
) -> tuple[BudgetContrast, ...]:
    seed_offset = statistics.holm_family_size
    return tuple(
        analyze_budget_contrast(
            evaluations=evaluations,
            level=level,
            treatment=AllocationPolicy.FABRID_MINIMAX,
            baseline=AllocationPolicy.EQ_FPR,
            metric=MetricId.MACRO_RECALL,
            statistics=statistics,
            analysis_seed=seed_offset + budget_index,
        )
        for budget_index, level in enumerate(budgets)
    )
