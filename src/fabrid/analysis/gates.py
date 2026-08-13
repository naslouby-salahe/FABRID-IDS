from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fabrid.analysis.contrasts import Contrast
from fabrid.analysis.types import PercentagePointDifference
from fabrid.domain.enums import GateStatus
from fabrid.domain.values import BudgetUsageRatio, Probability, RowCount
from fabrid.protocol.models import (
    BudgetComplianceGate,
    FabridMacroGate,
    FabridMinimaxGate,
)

_PERCENTAGE_POINTS_PER_UNIT = 100.0


@dataclass(frozen=True, slots=True)
class FabridMacroGateResult:
    status: GateStatus
    budgets_passing: RowCount
    total_budgets: RowCount
    differences: tuple[PercentagePointDifference, ...]


@dataclass(frozen=True, slots=True)
class FabridMinimaxGateResult:
    status: GateStatus
    budgets_passing: RowCount
    total_budgets: RowCount
    worst_client_differences: tuple[PercentagePointDifference, ...]
    macro_recall_differences: tuple[PercentagePointDifference, ...]


@dataclass(frozen=True, slots=True)
class BudgetComplianceResult:
    status: GateStatus
    median_usage: BudgetUsageRatio
    fraction_within_seed_limit: Probability
    maximum_usage: BudgetUsageRatio


def _status(passed: bool) -> GateStatus:
    return GateStatus.PASS if passed else GateStatus.FAIL


def _percentage_point_differences(
    contrasts: tuple[Contrast, ...],
) -> tuple[PercentagePointDifference, ...]:
    return tuple(
        PercentagePointDifference(
            contrast.bootstrap.mean_difference.value * _PERCENTAGE_POINTS_PER_UNIT
        )
        for contrast in contrasts
    )


def evaluate_fabrid_macro_gate(
    contrasts: tuple[Contrast, ...],
    gate: FabridMacroGate,
) -> FabridMacroGateResult:
    if len(contrasts) != gate.total_budgets.value:
        raise ValueError(
            f"expected {gate.total_budgets.value} budget contrasts, got {len(contrasts)}"
        )
    differences = _percentage_point_differences(contrasts)
    passing = RowCount(
        sum(
            1
            for difference in differences
            if difference.value >= gate.minimum_macro_recall_gain.value
        )
    )
    return FabridMacroGateResult(
        status=_status(passing.value >= gate.minimum_passing_budgets.value),
        budgets_passing=passing,
        total_budgets=gate.total_budgets,
        differences=differences,
    )


def evaluate_fabrid_minimax_gate(
    worst_client_contrasts: tuple[Contrast, ...],
    macro_recall_contrasts: tuple[Contrast, ...],
    gate: FabridMinimaxGate,
) -> FabridMinimaxGateResult:
    if len(worst_client_contrasts) != gate.total_budgets.value:
        raise ValueError("worst-client contrast count does not match the protocol")
    if len(macro_recall_contrasts) != gate.total_budgets.value:
        raise ValueError("macro-recall contrast count does not match the protocol")

    worst_differences = _percentage_point_differences(worst_client_contrasts)
    macro_differences = _percentage_point_differences(macro_recall_contrasts)
    passing = RowCount(
        sum(
            1
            for worst_difference, macro_difference in zip(
                worst_differences,
                macro_differences,
                strict=True,
            )
            if worst_difference.value
            >= gate.minimum_worst_client_recall_gain.value
            and macro_difference.value >= -gate.maximum_macro_recall_loss.value
        )
    )
    return FabridMinimaxGateResult(
        status=_status(passing.value >= gate.minimum_passing_budgets.value),
        budgets_passing=passing,
        total_budgets=gate.total_budgets,
        worst_client_differences=worst_differences,
        macro_recall_differences=macro_differences,
    )


def evaluate_budget_compliance(
    usage_by_seed: tuple[BudgetUsageRatio, ...],
    gate: BudgetComplianceGate,
) -> BudgetComplianceResult:
    if not usage_by_seed:
        raise ValueError("budget compliance requires at least one seed")
    raw = np.asarray(tuple(usage.value for usage in usage_by_seed), dtype=np.float64)
    median_usage = BudgetUsageRatio(float(np.median(raw)))
    fraction_within_limit = Probability(
        float(np.mean(raw <= gate.seed_usage_limit.value))
    )
    maximum_usage = BudgetUsageRatio(float(np.max(raw)))
    return BudgetComplianceResult(
        status=_status(
            median_usage.value <= gate.maximum_median_usage.value
            and fraction_within_limit.value
            >= gate.minimum_seed_fraction_below_limit.value
        ),
        median_usage=median_usage,
        fraction_within_seed_limit=fraction_within_limit,
        maximum_usage=maximum_usage,
    )
