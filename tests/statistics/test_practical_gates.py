from __future__ import annotations

import pytest

from fabrid.config.protocol import (
    BudgetComplianceThresholds,
    FabridMacroGate,
    FabridMinimaxGate,
)
from fabrid.experiments.main_experiment import SeedBudgetResult
from fabrid.schemas.allocation import AllocationPolicy
from fabrid.statistics.contrasts import macro_recall_contrast, worst_client_recall_contrast
from fabrid.statistics.practical_gates import (
    evaluate_budget_compliance,
    evaluate_fabrid_macro_gate,
    evaluate_fabrid_minimax_gate,
)

_MACRO_GATE = FabridMacroGate(min_delta_macro_recall_pp=2.0, min_budgets_passing=3, of_budgets=5)
_MINIMAX_GATE = FabridMinimaxGate(
    min_delta_worst_client_recall_pp=5.0,
    max_macro_recall_loss_pp=2.0,
    min_budgets_passing=3,
    of_budgets=5,
)
_COMPLIANCE = BudgetComplianceThresholds(
    median_bur_leq=1.05, seeds_with_bur_leq_1_10_min_fraction=0.9
)


def _contrast_from_deltas(deltas: tuple[float, ...]):
    results = tuple(
        SeedBudgetResult(
            seed=seed,
            budget=0.01,
            fallback_rate=0.0,
            macro_recall_by_policy={
                AllocationPolicy.EQ_FPR: 0.5,
                AllocationPolicy.FABRID_MACRO: 0.5 + delta,
            },
        )
        for seed, delta in enumerate(deltas)
    )
    return macro_recall_contrast(
        results,
        AllocationPolicy.FABRID_MACRO,
        AllocationPolicy.EQ_FPR,
        bootstrap_resamples=200,
        bootstrap_seed=0,
    )


def test_fabrid_macro_gate_passes_when_enough_budgets_clear_threshold() -> None:
    passing = _contrast_from_deltas((0.03,) * 5)
    failing = _contrast_from_deltas((0.0,) * 5)
    contrasts = (passing, passing, passing, failing, failing)
    result = evaluate_fabrid_macro_gate(contrasts, _MACRO_GATE)
    assert result.passed
    assert result.budgets_passing == 3


def test_fabrid_macro_gate_fails_when_too_few_budgets_clear_threshold() -> None:
    passing = _contrast_from_deltas((0.03,) * 5)
    failing = _contrast_from_deltas((0.0,) * 5)
    contrasts = (passing, passing, failing, failing, failing)
    result = evaluate_fabrid_macro_gate(contrasts, _MACRO_GATE)
    assert not result.passed
    assert result.budgets_passing == 2


def test_fabrid_macro_gate_requires_exact_budget_count() -> None:
    contrast = _contrast_from_deltas((0.03,) * 5)
    with pytest.raises(ValueError, match="expected exactly"):
        evaluate_fabrid_macro_gate((contrast, contrast), _MACRO_GATE)


def _worst_client_contrast_from_deltas(deltas: tuple[float, ...]):
    results = tuple(
        SeedBudgetResult(
            seed=seed,
            budget=0.01,
            fallback_rate=0.0,
            worst_client_recall_by_policy={
                AllocationPolicy.EQ_FPR: 0.3,
                AllocationPolicy.FABRID_MINIMAX: 0.3 + delta,
            },
        )
        for seed, delta in enumerate(deltas)
    )
    return worst_client_recall_contrast(
        results,
        AllocationPolicy.FABRID_MINIMAX,
        AllocationPolicy.EQ_FPR,
        bootstrap_resamples=200,
        bootstrap_seed=0,
    )


def test_fabrid_minimax_gate_passes_when_both_conditions_hold() -> None:
    good_worst = _worst_client_contrast_from_deltas((0.08,) * 5)
    good_macro = _contrast_from_deltas((0.0,) * 5)  # 0 loss, within -2pp allowance
    result = evaluate_fabrid_minimax_gate((good_worst,) * 5, (good_macro,) * 5, _MINIMAX_GATE)
    assert result.passed
    assert result.budgets_passing == 5


def test_fabrid_minimax_gate_fails_when_macro_recall_drops_too_much() -> None:
    good_worst = _worst_client_contrast_from_deltas((0.08,) * 5)
    bad_macro = _contrast_from_deltas((-0.05,) * 5)  # -5pp loss, exceeds -2pp allowance
    result = evaluate_fabrid_minimax_gate((good_worst,) * 5, (bad_macro,) * 5, _MINIMAX_GATE)
    assert not result.passed
    assert result.budgets_passing == 0


def test_budget_compliance_passes_within_thresholds() -> None:
    bur_by_seed = (1.0, 1.02, 1.03, 1.01, 1.0, 1.04, 1.0, 1.02, 1.01, 1.0)
    result = evaluate_budget_compliance(bur_by_seed, _COMPLIANCE)
    assert result.passed
    assert result.median_bur == pytest.approx(1.01)
    assert result.fraction_seeds_bur_leq_1_10 == pytest.approx(1.0)
    assert result.max_bur == pytest.approx(1.04)


def test_budget_compliance_fails_on_high_median() -> None:
    bur_by_seed = (1.2,) * 10
    result = evaluate_budget_compliance(bur_by_seed, _COMPLIANCE)
    assert not result.passed


def test_budget_compliance_requires_at_least_one_seed() -> None:
    with pytest.raises(ValueError):
        evaluate_budget_compliance((), _COMPLIANCE)
