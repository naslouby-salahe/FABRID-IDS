"""Practical Success Gates (roadmap section 72) and Budget-Compliance Reporting (section 73):
preregistered engineering thresholds evaluated against real bootstrap contrast results, not
against hypothesis-test p-values alone.
"""

from __future__ import annotations

from dataclasses import dataclass

from fabrid.config.protocol import (
    BudgetComplianceThresholds,
    FabridMacroGate,
    FabridMinimaxGate,
)
from fabrid.statistics.contrasts import Contrast

_PERCENTAGE_POINTS_PER_UNIT = 100.0


@dataclass(frozen=True, slots=True)
class FabridMacroGateResult:
    passed: bool
    budgets_passing: int
    of_budgets: int
    delta_macro_recall_pp_by_budget: tuple[float, ...]


def evaluate_fabrid_macro_gate(
    contrasts_by_budget: tuple[Contrast, ...], gate: FabridMacroGate
) -> FabridMacroGateResult:
    """`contrasts_by_budget` must have exactly `gate.of_budgets` entries, one MacroRecall
    contrast (e.g. FABRID_MACRO - EQ_FPR) per primary budget.
    """
    if len(contrasts_by_budget) != gate.of_budgets:
        raise ValueError(
            f"expected exactly {gate.of_budgets} budget contrasts, got {len(contrasts_by_budget)}"
        )
    deltas_pp = tuple(
        c.bootstrap.mean_difference * _PERCENTAGE_POINTS_PER_UNIT for c in contrasts_by_budget
    )
    budgets_passing = sum(1 for delta in deltas_pp if delta >= gate.min_delta_macro_recall_pp)
    return FabridMacroGateResult(
        passed=budgets_passing >= gate.min_budgets_passing,
        budgets_passing=budgets_passing,
        of_budgets=gate.of_budgets,
        delta_macro_recall_pp_by_budget=deltas_pp,
    )


@dataclass(frozen=True, slots=True)
class FabridMinimaxGateResult:
    passed: bool
    budgets_passing: int
    of_budgets: int
    delta_worst_client_recall_pp_by_budget: tuple[float, ...]
    delta_macro_recall_pp_by_budget: tuple[float, ...]


def evaluate_fabrid_minimax_gate(
    worst_client_contrasts_by_budget: tuple[Contrast, ...],
    macro_recall_contrasts_by_budget: tuple[Contrast, ...],
    gate: FabridMinimaxGate,
) -> FabridMinimaxGateResult:
    """A budget passes if `Delta WorstClientRecall >= min_delta_worst_client_recall_pp` AND
    `Delta MacroRecall >= -max_macro_recall_loss_pp` (i.e. macro recall does not drop by more
    than the allowed loss). Both contrast tuples must align budget-for-budget with `gate.of_budgets`
    entries each.
    """
    if len(worst_client_contrasts_by_budget) != gate.of_budgets:
        raise ValueError(
            f"expected exactly {gate.of_budgets} worst-client-recall budget contrasts, got "
            f"{len(worst_client_contrasts_by_budget)}"
        )
    if len(macro_recall_contrasts_by_budget) != gate.of_budgets:
        raise ValueError(
            f"expected exactly {gate.of_budgets} macro-recall budget contrasts, got "
            f"{len(macro_recall_contrasts_by_budget)}"
        )
    worst_deltas_pp = tuple(
        c.bootstrap.mean_difference * _PERCENTAGE_POINTS_PER_UNIT
        for c in worst_client_contrasts_by_budget
    )
    macro_deltas_pp = tuple(
        c.bootstrap.mean_difference * _PERCENTAGE_POINTS_PER_UNIT
        for c in macro_recall_contrasts_by_budget
    )
    budgets_passing = sum(
        1
        for worst_delta, macro_delta in zip(worst_deltas_pp, macro_deltas_pp, strict=True)
        if worst_delta >= gate.min_delta_worst_client_recall_pp
        and macro_delta >= -gate.max_macro_recall_loss_pp
    )
    return FabridMinimaxGateResult(
        passed=budgets_passing >= gate.min_budgets_passing,
        budgets_passing=budgets_passing,
        of_budgets=gate.of_budgets,
        delta_worst_client_recall_pp_by_budget=worst_deltas_pp,
        delta_macro_recall_pp_by_budget=macro_deltas_pp,
    )


@dataclass(frozen=True, slots=True)
class BudgetComplianceResult:
    passed: bool
    median_bur: float
    fraction_seeds_bur_leq_1_10: float
    max_bur: float


def _median(values: tuple[float, ...]) -> float:
    sorted_values = sorted(values)
    n = len(sorted_values)
    midpoint = n // 2
    if n % 2 == 1:
        return sorted_values[midpoint]
    return (sorted_values[midpoint - 1] + sorted_values[midpoint]) / 2


def evaluate_budget_compliance(
    bur_by_seed: tuple[float, ...], thresholds: BudgetComplianceThresholds
) -> BudgetComplianceResult:
    if not bur_by_seed:
        raise ValueError("evaluate_budget_compliance requires at least one seed's BUR")
    median_bur = _median(bur_by_seed)
    fraction_leq_1_10 = sum(1 for bur in bur_by_seed if bur <= 1.10) / len(bur_by_seed)
    return BudgetComplianceResult(
        passed=(
            median_bur <= thresholds.median_bur_leq
            and fraction_leq_1_10 >= thresholds.seeds_with_bur_leq_1_10_min_fraction
        ),
        median_bur=median_bur,
        fraction_seeds_bur_leq_1_10=fraction_leq_1_10,
        max_bur=max(bur_by_seed),
    )
