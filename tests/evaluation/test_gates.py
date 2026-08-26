from __future__ import annotations

import pytest

from fabrid.artifacts.paths import ExperimentCoordinate
from fabrid.config import (
    AllocationPolicy,
    BudgetComplianceConfig,
    BudgetId,
    BudgetLevel,
    DatasetId,
    EvidenceAvailability,
    ExperimentalUnit,
    ExperimentId,
    ExperimentVariantId,
    FabridCvarGateConfig,
    FabridMacroGateConfig,
    GateStatus,
    PracticalGateConfig,
    StatisticsConfig,
    WeightMode,
)
from fabrid.evaluation.gates import (
    AvailableBudgetCompliance,
    AvailableFabridCvarGate,
    AvailableFabridMacroGate,
    UnavailableBudgetCompliance,
    UnavailableFabridCvarGate,
    UnavailableFabridMacroGate,
    analyze_budget_compliance,
    analyze_practical_gates,
    evaluate_fabrid_cvar_gate,
    evaluate_fabrid_macro_gate,
)
from fabrid.evaluation.gates import (
    EvidenceAvailability as EvaluationEvidenceAvailability,
)
from fabrid.evaluation.gates import (
    GateStatus as EvaluationGateStatus,
)
from fabrid.evaluation.inference import (
    BootstrapResult,
    BudgetContrast,
    Contrast,
    SignFlipResult,
    analyze_cvar_macro_guardrail,
)
from fabrid.evaluation.metrics import (
    CompletedPolicyEvaluation,
    FprDispersion,
    MetricId,
    SeedBudgetEvaluation,
    budget_violation_ratio,
)


def test_gate_enums_are_the_config_enums() -> None:
    assert EvaluationGateStatus is GateStatus
    assert EvaluationEvidenceAvailability is EvidenceAvailability


def _gates() -> PracticalGateConfig:
    return PracticalGateConfig(
        fabrid_macro=FabridMacroGateConfig(
            minimum_macro_recall=0.95,
            minimum_passing_budgets=3,
            total_budgets=5,
        ),
        fabrid_cvar=FabridCvarGateConfig(
            minimum_worst_client_recall=0.8,
            maximum_macro_recall_loss=2.0,
            minimum_passing_budgets=3,
            total_budgets=5,
        ),
        budget_compliance=BudgetComplianceConfig(
            maximum_median_usage=1.05,
            seed_usage_limit=1.1,
            minimum_seed_fraction_below_limit=0.9,
            evaluated_budgets=_BUDGETS,
        ),
    )


def _statistics(*, family_size: int = 5) -> StatisticsConfig:
    return StatisticsConfig(
        experimental_unit=ExperimentalUnit.DETECTOR_SEED,
        sign_flip_enumeration=1024,
        significance=0.05,
        holm_family_size=family_size,
        bootstrap_resamples=16,
        bootstrap_confidence=0.95,
    )


def _budget_contrast(
    budget_id: BudgetId,
    mean_difference: float,
    *,
    treatment: AllocationPolicy,
    metric: MetricId,
) -> BudgetContrast:
    return BudgetContrast(
        budget_id=budget_id,
        contrast=Contrast(
            treatment=treatment,
            baseline=AllocationPolicy.EQ_FPR,
            metric=metric,
            paired_differences=(mean_difference,),
            included_seeds=(0,),
            excluded_seeds=(),
            sign_flip=SignFlipResult(
                observed_mean_difference=mean_difference,
                p_value=1.0,
                enumerated_assignments=2,
            ),
            bootstrap=BootstrapResult(
                mean_difference=mean_difference,
                median_difference=mean_difference,
                confidence_interval_low=mean_difference,
                confidence_interval_high=mean_difference,
                confidence=0.95,
                resamples=1,
            ),
        ),
    )


_BUDGETS = (
    BudgetId.FALSE_POSITIVE_0P001,
    BudgetId.FALSE_POSITIVE_0P0025,
    BudgetId.FALSE_POSITIVE_0P005,
    BudgetId.FALSE_POSITIVE_0P010,
    BudgetId.FALSE_POSITIVE_0P020,
)


def test_fabrid_macro_gate_counts_absolute_recall() -> None:
    levels = tuple(
        BudgetLevel(budget_id=budget_id, value=value)
        for budget_id, value in zip(_BUDGETS, (0.001, 0.0025, 0.005, 0.01, 0.02), strict=True)
    )
    passing = evaluate_fabrid_macro_gate(
        tuple(
            _seed_evaluation(0, level, usage=1.0, macro_gain=0.16, worst_gain=0.0)
            for level in levels
        ),
        _gates(),
    )
    assert isinstance(passing, AvailableFabridMacroGate)
    assert passing.availability is EvidenceAvailability.AVAILABLE
    assert passing.status is GateStatus.PASS
    assert passing.budgets_passing == 5
    assert passing.differences[0].macro_recall == pytest.approx(0.96)
    failing = evaluate_fabrid_macro_gate(
        tuple(
            _seed_evaluation(0, level, usage=1.0, macro_gain=0.14, worst_gain=0.0)
            for level in levels
        ),
        _gates(),
    )
    assert isinstance(failing, AvailableFabridMacroGate)
    assert failing.status is GateStatus.FAIL
    assert failing.budgets_passing == 0


def test_fabrid_macro_gate_requires_every_budget() -> None:
    level = BudgetLevel(budget_id=BudgetId.FALSE_POSITIVE_0P005, value=0.005)
    result = evaluate_fabrid_macro_gate(
        (_seed_evaluation(0, level, usage=1.0, macro_gain=0.0, worst_gain=0.0),),
        _gates(),
    )
    assert isinstance(result, UnavailableFabridMacroGate)
    assert result.availability is EvidenceAvailability.INSUFFICIENT_EVIDENCE


def test_fabrid_cvar_gate_requires_absolute_worst_client_and_macro_guardrail() -> None:
    worst_gains = (0.35, 0.32, 0.31, 0.29, 0.10)
    levels = tuple(
        BudgetLevel(budget_id=budget_id, value=value)
        for budget_id, value in zip(_BUDGETS, (0.001, 0.0025, 0.005, 0.01, 0.02), strict=True)
    )
    evaluations = tuple(
        _seed_evaluation(0, level, usage=1.0, macro_gain=0.0, worst_gain=worst_gain)
        for level, worst_gain in zip(levels, worst_gains, strict=True)
    )
    guardrail = tuple(
        _budget_contrast(
            budget_id,
            difference,
            treatment=AllocationPolicy.FABRID_CVAR,
            metric=MetricId.MACRO_RECALL,
        )
        for budget_id, difference in zip(_BUDGETS, (0.00, -0.02, -0.019, 0.00, 0.00), strict=True)
    )
    passing = evaluate_fabrid_cvar_gate(evaluations, guardrail, _gates())
    assert isinstance(passing, AvailableFabridCvarGate)
    assert passing.availability is EvidenceAvailability.AVAILABLE
    assert passing.status is GateStatus.PASS
    assert passing.budgets_passing == 3
    assert passing.differences[0].worst_client_recall == pytest.approx(0.85)
    failing = evaluate_fabrid_cvar_gate(
        evaluations,
        tuple(
            _budget_contrast(
                budget_id,
                -0.021,
                treatment=AllocationPolicy.FABRID_CVAR,
                metric=MetricId.MACRO_RECALL,
            )
            for budget_id in _BUDGETS
        ),
        _gates(),
    )
    assert isinstance(failing, AvailableFabridCvarGate)
    assert failing.status is GateStatus.FAIL
    assert failing.budgets_passing == 0


def test_fabrid_cvar_gate_requires_complete_evidence() -> None:
    result = evaluate_fabrid_cvar_gate((), (), _gates())
    assert isinstance(result, UnavailableFabridCvarGate)
    assert result.availability is EvidenceAvailability.INSUFFICIENT_EVIDENCE


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
    *,
    macro_recall: float = 0.8,
    worst_client_recall: float = 0.5,
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


def _seed_evaluation(
    seed: int,
    level: BudgetLevel,
    *,
    usage: float,
    macro_gain: float,
    worst_gain: float,
) -> SeedBudgetEvaluation:
    return SeedBudgetEvaluation(
        experiment=ExperimentCoordinate(
            experiment_id=ExperimentId.MATCHED_BUDGET,
            variant_id=ExperimentVariantId.PRIMARY,
            dataset_id=DatasetId.NBAIOT,
            detector_seed=seed,
            budget_id=level.budget_id,
            budget=level.value,
            weight_mode=WeightMode.EQUAL_CLIENT,
        ),
        policies=(
            _completed(AllocationPolicy.EQ_FPR, budget_usage=usage),
            _completed(AllocationPolicy.GREEDY, budget_usage=usage),
            _completed(
                AllocationPolicy.FABRID_MACRO,
                macro_recall=0.80 + macro_gain,
                budget_usage=usage,
            ),
            _completed(
                AllocationPolicy.FABRID_CVAR,
                worst_client_recall=0.50 + worst_gain,
                budget_usage=usage,
            ),
        ),
    )


def test_budget_compliance_uses_median_and_seed_fraction() -> None:
    level = BudgetLevel(budget_id=BudgetId.FALSE_POSITIVE_0P005, value=0.005)
    usages = (1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.09)
    evaluations = tuple(
        _seed_evaluation(seed, level, usage=usage, macro_gain=0.0, worst_gain=0.0)
        for seed, usage in enumerate(usages)
    )
    results = analyze_budget_compliance(evaluations, (level,), _gates(), expected_seed_count=10)
    fabrid_macro = next(
        result
        for result in results
        if isinstance(result, AvailableBudgetCompliance)
        and result.policy is AllocationPolicy.FABRID_MACRO
    )
    assert fabrid_macro.status is GateStatus.PASS
    assert fabrid_macro.median_usage == pytest.approx(1.00)
    assert fabrid_macro.fraction_within_seed_limit == pytest.approx(1.0)
    assert fabrid_macro.maximum_usage == pytest.approx(1.09)


def test_budget_compliance_fails_when_median_exceeds_limit() -> None:
    level = BudgetLevel(budget_id=BudgetId.FALSE_POSITIVE_0P005, value=0.005)
    evaluations = tuple(
        _seed_evaluation(seed, level, usage=1.06, macro_gain=0.0, worst_gain=0.0)
        for seed in range(10)
    )
    results = analyze_budget_compliance(evaluations, (level,), _gates(), expected_seed_count=10)
    statuses = {
        result.policy: result.status
        for result in results
        if isinstance(result, AvailableBudgetCompliance)
    }
    assert set(statuses.values()) == {GateStatus.FAIL}


def test_budget_compliance_requires_every_seed() -> None:
    level = BudgetLevel(budget_id=BudgetId.FALSE_POSITIVE_0P005, value=0.005)
    evaluation = _seed_evaluation(0, level, usage=1.0, macro_gain=0.0, worst_gain=0.0)
    results = analyze_budget_compliance((evaluation,), (level,), _gates(), expected_seed_count=10)
    assert all(isinstance(result, UnavailableBudgetCompliance) for result in results)
    assert all(
        result.availability is EvidenceAvailability.INSUFFICIENT_EVIDENCE for result in results
    )


def test_analyze_practical_gates_composes_primary_and_guardrail() -> None:
    levels = tuple(
        BudgetLevel(budget_id=budget_id, value=value)
        for budget_id, value in zip(_BUDGETS, (0.001, 0.0025, 0.005, 0.01, 0.02), strict=True)
    )
    evaluations = tuple(
        _seed_evaluation(seed, level, usage=1.0, macro_gain=0.16, worst_gain=0.31)
        for seed in range(2)
        for level in levels
    )
    statistics = _statistics()
    analysis = analyze_practical_gates(
        evaluations,
        levels,
        statistics,
        _gates(),
        expected_seed_count=2,
    )
    assert isinstance(analysis.fabrid_macro, AvailableFabridMacroGate)
    assert analysis.fabrid_macro.status is GateStatus.PASS
    assert isinstance(analysis.fabrid_cvar, AvailableFabridCvarGate)
    assert analysis.fabrid_cvar.status is GateStatus.PASS
    guardrail = analyze_cvar_macro_guardrail(evaluations, levels, statistics)
    assert all(
        budget_contrast.contrast.bootstrap.mean_difference == pytest.approx(0.0)
        for budget_contrast in guardrail
    )
