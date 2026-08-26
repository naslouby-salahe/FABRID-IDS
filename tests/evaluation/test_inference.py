from __future__ import annotations

import pytest

from fabrid.allocation.optimization import SolverStatus
from fabrid.artifacts.paths import ExperimentCoordinate
from fabrid.config import (
    AllocationPolicy,
    BudgetId,
    BudgetLevel,
    DatasetId,
    ExperimentalUnit,
    ExperimentId,
    ExperimentVariantId,
    StatisticsConfig,
    WeightMode,
)
from fabrid.evaluation.inference import (
    HypothesisDecision,
    analyze_primary_inference,
    exact_sign_flip_test,
    holm_correction,
    paired_bootstrap_ci,
)
from fabrid.evaluation.metrics import (
    CompletedPolicyEvaluation,
    ExcludedPolicyEvaluation,
    FprDispersion,
    MetricId,
    SeedBudgetEvaluation,
    budget_violation_ratio,
)


def _statistics(*, family_size: int = 1, resamples: int = 32) -> StatisticsConfig:
    return StatisticsConfig(
        experimental_unit=ExperimentalUnit.DETECTOR_SEED,
        sign_flip_enumeration=1024,
        significance=0.05,
        holm_family_size=family_size,
        bootstrap_resamples=resamples,
        bootstrap_confidence=0.95,
    )


def _dispersion() -> FprDispersion:
    return FprDispersion(
        median=0.01,
        interquartile_range=0.0,
        minimum=0.01,
        maximum=0.01,
        coefficient_of_variation=0.0,
    )


def _completed(
    policy: AllocationPolicy,
    macro_recall: float,
    worst_client_recall: float,
    *,
    budget_usage: float | None = 1.0,
) -> CompletedPolicyEvaluation:
    return CompletedPolicyEvaluation(
        policy=policy,
        macro_recall=macro_recall,
        worst_client_recall=worst_client_recall,
        federation_fpr=0.01,
        budget_usage=budget_usage,
        fallback_rate=0.0,
        fpr_dispersion=_dispersion(),
        solver_runtime_ms=None,
        pooled_recall=macro_recall,
        macro_f1=macro_recall,
        balanced_accuracy=macro_recall,
        auroc=0.9,
        auprc=0.8,
        false_alert_gini=0.0,
        budget_violation=budget_violation_ratio(budget_usage),
    )


def _evaluation(
    seed: int,
    budget_id: BudgetId,
    budget: float,
    policies: tuple[CompletedPolicyEvaluation | ExcludedPolicyEvaluation, ...],
) -> SeedBudgetEvaluation:
    return SeedBudgetEvaluation(
        experiment=ExperimentCoordinate(
            experiment_id=ExperimentId.MATCHED_BUDGET,
            variant_id=ExperimentVariantId.PRIMARY,
            dataset_id=DatasetId.NBAIOT,
            detector_seed=seed,
            budget_id=budget_id,
            budget=budget,
            weight_mode=WeightMode.EQUAL_CLIENT,
        ),
        policies=policies,
    )


def test_exact_sign_flip_two_sided_enumeration() -> None:
    two_equal = exact_sign_flip_test((1.0, 1.0))
    assert two_equal.observed_mean_difference == 1.0
    assert two_equal.enumerated_assignments == 4
    assert two_equal.p_value == 0.5
    cancelled = exact_sign_flip_test((2.0, 0.0))
    assert cancelled.observed_mean_difference == 1.0
    assert cancelled.p_value == 1.0
    singleton = exact_sign_flip_test((1.0,))
    assert singleton.enumerated_assignments == 2
    assert singleton.p_value == 1.0


def test_exact_sign_flip_rejects_empty_pairs() -> None:
    with pytest.raises(ValueError, match="paired differences"):
        exact_sign_flip_test(())


def test_holm_rejects_prefix_and_monotone_adjusts() -> None:
    results = holm_correction((0.01, 0.04, 0.03), 0.05)
    assert tuple(result.p_value for result in results) == (0.01, 0.04, 0.03)
    assert results[0].adjusted_p_value == pytest.approx(0.03)
    assert results[1].adjusted_p_value == pytest.approx(0.06)
    assert results[2].adjusted_p_value == pytest.approx(0.06)
    assert results[0].decision is HypothesisDecision.REJECT
    assert results[1].decision is HypothesisDecision.RETAIN
    assert results[2].decision is HypothesisDecision.RETAIN


def test_paired_bootstrap_is_seed_deterministic() -> None:
    statistics = _statistics()
    first = paired_bootstrap_ci((0.02, 0.04, -0.01), statistics, seed=3)
    second = paired_bootstrap_ci((0.02, 0.04, -0.01), statistics, seed=3)
    third = paired_bootstrap_ci((0.02, 0.04, -0.01), statistics, seed=4)
    assert first == second
    assert first.mean_difference == pytest.approx(sum((0.02, 0.04, -0.01)) / 3)
    assert first.resamples == statistics.bootstrap_resamples
    assert first.confidence == statistics.bootstrap_confidence
    assert first.confidence_interval_low != third.confidence_interval_low


def test_primary_inference_pairs_per_seed_and_budget() -> None:
    budget = BudgetLevel(budget_id=BudgetId.FALSE_POSITIVE_0P005, value=0.005)
    evaluations = (
        _evaluation(
            0,
            budget.budget_id,
            budget.value,
            (
                _completed(AllocationPolicy.EQ_FPR, 0.80, 0.50),
                _completed(AllocationPolicy.FABRID_MACRO, 0.82, 0.50),
                _completed(AllocationPolicy.FABRID_CVAR, 0.80, 0.56),
            ),
        ),
        _evaluation(
            1,
            budget.budget_id,
            budget.value,
            (
                _completed(AllocationPolicy.EQ_FPR, 0.80, 0.50),
                _completed(AllocationPolicy.FABRID_MACRO, 0.84, 0.50),
                _completed(AllocationPolicy.FABRID_CVAR, 0.79, 0.54),
            ),
        ),
        _evaluation(
            2,
            budget.budget_id,
            budget.value,
            (
                _completed(AllocationPolicy.EQ_FPR, 0.80, 0.50),
                ExcludedPolicyEvaluation(
                    policy=AllocationPolicy.FABRID_MACRO,
                    status=SolverStatus.SOLVER_INVALID,
                    reason="infeasible",
                ),
                _completed(AllocationPolicy.FABRID_CVAR, 0.80, 0.55),
            ),
        ),
    )
    primary = analyze_primary_inference(evaluations, (budget,), _statistics())
    macro = primary.macro_recall[0].contrast
    assert macro.metric is MetricId.MACRO_RECALL
    assert macro.treatment is AllocationPolicy.FABRID_MACRO
    assert macro.baseline is AllocationPolicy.EQ_FPR
    assert macro.paired_differences == pytest.approx((0.02, 0.04))
    assert macro.included_seeds == (0, 1)
    assert macro.excluded_seeds == (2,)
    assert macro.bootstrap.mean_difference == pytest.approx(0.03)
    worst = primary.worst_client_recall[0].contrast
    assert worst.metric is MetricId.WORST_CLIENT_RECALL
    assert worst.treatment is AllocationPolicy.FABRID_CVAR
    assert worst.paired_differences == pytest.approx((0.06, 0.04, 0.05))
    assert worst.included_seeds == (0, 1, 2)


def test_primary_inference_requires_matching_holm_family() -> None:
    budget = BudgetLevel(budget_id=BudgetId.FALSE_POSITIVE_0P005, value=0.005)
    evaluation = _evaluation(
        0,
        budget.budget_id,
        budget.value,
        (
            _completed(AllocationPolicy.EQ_FPR, 0.80, 0.50),
            _completed(AllocationPolicy.FABRID_MACRO, 0.82, 0.50),
            _completed(AllocationPolicy.FABRID_CVAR, 0.80, 0.56),
        ),
    )
    statistics = _statistics(family_size=5)
    with pytest.raises(ValueError, match="Holm family size"):
        analyze_primary_inference((evaluation,), (budget,), statistics)
