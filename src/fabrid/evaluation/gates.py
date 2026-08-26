from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, ConfigDict

from fabrid.config import (
    PERCENTAGE_POINTS_PER_UNIT,
    AllocationPolicy,
    BudgetId,
    BudgetLevel,
    BudgetUsageRatio,
    EvidenceAvailability,
    FailureReason,
    GateStatus,
    MetricDifference,
    PercentagePointDifference,
    PracticalGateConfig,
    Probability,
    RowCount,
    SeedCount,
    StatisticsConfig,
)
from fabrid.evaluation.inference import (
    BudgetContrast,
    analyze_cvar_macro_guardrail,
)
from fabrid.evaluation.metrics import CompletedPolicyEvaluation, SeedBudgetEvaluation

_BUDGET_COMPLIANCE_POLICIES = (
    AllocationPolicy.EQ_FPR,
    AllocationPolicy.GREEDY,
    AllocationPolicy.FABRID_MACRO,
    AllocationPolicy.FABRID_CVAR,
)


class BudgetDifference(BaseModel):
    model_config = ConfigDict(frozen=True)
    budget_id: BudgetId
    macro_recall: Probability


class CvarBudgetDifference(BaseModel):
    model_config = ConfigDict(frozen=True)
    budget_id: BudgetId
    worst_client_recall: Probability
    macro_recall_percentage_points: PercentagePointDifference


class AvailableFabridMacroGate(BaseModel):
    model_config = ConfigDict(frozen=True)
    availability: EvidenceAvailability
    status: GateStatus
    budgets_passing: RowCount
    total_budgets: RowCount
    differences: tuple[BudgetDifference, ...]


class UnavailableFabridMacroGate(BaseModel):
    model_config = ConfigDict(frozen=True)
    availability: EvidenceAvailability
    reason: FailureReason


FabridMacroGateResult = AvailableFabridMacroGate | UnavailableFabridMacroGate


class AvailableFabridCvarGate(BaseModel):
    model_config = ConfigDict(frozen=True)
    availability: EvidenceAvailability
    status: GateStatus
    budgets_passing: RowCount
    total_budgets: RowCount
    differences: tuple[CvarBudgetDifference, ...]


class UnavailableFabridCvarGate(BaseModel):
    model_config = ConfigDict(frozen=True)
    availability: EvidenceAvailability
    reason: FailureReason


FabridCvarGateResult = AvailableFabridCvarGate | UnavailableFabridCvarGate


class AvailableBudgetCompliance(BaseModel):
    model_config = ConfigDict(frozen=True)
    policy: AllocationPolicy
    budget_id: BudgetId
    availability: EvidenceAvailability
    status: GateStatus
    median_usage: BudgetUsageRatio
    fraction_within_seed_limit: Probability
    maximum_usage: BudgetUsageRatio
    seed_count: RowCount


class UnavailableBudgetCompliance(BaseModel):
    model_config = ConfigDict(frozen=True)
    policy: AllocationPolicy
    budget_id: BudgetId
    availability: EvidenceAvailability
    reason: FailureReason


BudgetComplianceResult = AvailableBudgetCompliance | UnavailableBudgetCompliance


class PracticalGateAnalysis(BaseModel):
    model_config = ConfigDict(frozen=True)
    fabrid_macro: FabridMacroGateResult
    fabrid_cvar: FabridCvarGateResult
    budget_compliance: tuple[BudgetComplianceResult, ...]


def _status(passed: bool) -> GateStatus:
    return GateStatus.PASS if passed else GateStatus.FAIL


def _percentage_point_difference(
    mean_difference: MetricDifference,
) -> PercentagePointDifference:
    return mean_difference * PERCENTAGE_POINTS_PER_UNIT


def evaluate_fabrid_macro_gate(
    evaluations: tuple[SeedBudgetEvaluation, ...],
    gates: PracticalGateConfig,
) -> FabridMacroGateResult:
    macro = gates.fabrid_macro
    macro_by_budget = _policy_metric_by_budget(
        evaluations, AllocationPolicy.FABRID_MACRO, _macro_recall_of
    )
    if len(macro_by_budget) != macro.total_budgets:
        return UnavailableFabridMacroGate(
            availability=EvidenceAvailability.INSUFFICIENT_EVIDENCE,
            reason="FABRID-Macro gate requires a macro recall at every budget",
        )
    differences = tuple(
        BudgetDifference(
            budget_id=budget_metric.budget_id,
            macro_recall=budget_metric.mean,
        )
        for budget_metric in macro_by_budget
    )
    passing = sum(
        1 for difference in differences if difference.macro_recall >= macro.minimum_macro_recall
    )
    return AvailableFabridMacroGate(
        availability=EvidenceAvailability.AVAILABLE,
        status=_status(passing >= macro.minimum_passing_budgets),
        budgets_passing=passing,
        total_budgets=macro.total_budgets,
        differences=differences,
    )


@dataclass(frozen=True, slots=True)
class _BudgetMetric:
    budget_id: BudgetId
    mean: Probability


def _policy_metric_by_budget(
    evaluations: tuple[SeedBudgetEvaluation, ...],
    policy: AllocationPolicy,
    metric: Callable[[CompletedPolicyEvaluation], Probability],
) -> tuple[_BudgetMetric, ...]:
    grouped: list[tuple[BudgetId, list[Probability]]] = []
    for evaluation in evaluations:
        completed = [
            item
            for item in evaluation.policies
            if isinstance(item, CompletedPolicyEvaluation) and item.policy is policy
        ]
        if not completed:
            raise ValueError(f"no completed {policy.value} policy in evaluation")
        for budget_id, values in grouped:
            if budget_id == evaluation.experiment.budget_id:
                values.append(metric(completed[0]))
                break
        else:
            grouped.append((evaluation.experiment.budget_id, [metric(completed[0])]))
    return tuple(
        _BudgetMetric(
            budget_id=budget_id,
            mean=sum(values) / len(values),
        )
        for budget_id, values in grouped
    )


def _worst_client_recall_of(evaluation: CompletedPolicyEvaluation) -> Probability:
    return evaluation.worst_client_recall


def _macro_recall_of(evaluation: CompletedPolicyEvaluation) -> Probability:
    return evaluation.macro_recall


def evaluate_fabrid_cvar_gate(
    evaluations: tuple[SeedBudgetEvaluation, ...],
    guardrail: tuple[BudgetContrast, ...],
    gates: PracticalGateConfig,
) -> FabridCvarGateResult:
    cvar = gates.fabrid_cvar
    if len(guardrail) != cvar.total_budgets:
        return UnavailableFabridCvarGate(
            availability=EvidenceAvailability.INSUFFICIENT_EVIDENCE,
            reason="FABRID-CVaR gate requires all worst-client and MacroRecall effects",
        )
    worst_by_budget = _policy_metric_by_budget(
        evaluations, AllocationPolicy.FABRID_CVAR, _worst_client_recall_of
    )
    if len(worst_by_budget) != cvar.total_budgets:
        return UnavailableFabridCvarGate(
            availability=EvidenceAvailability.INSUFFICIENT_EVIDENCE,
            reason="FABRID-CVaR gate requires a CVaR worst-client recall at every budget",
        )
    differences = tuple(
        CvarBudgetDifference(
            budget_id=worst_metric.budget_id,
            worst_client_recall=worst_metric.mean,
            macro_recall_percentage_points=_percentage_point_difference(
                guardrail_contrast.contrast.bootstrap.mean_difference
            ),
        )
        for worst_metric, guardrail_contrast in zip(worst_by_budget, guardrail, strict=True)
    )
    passing = sum(
        1
        for difference in differences
        if difference.worst_client_recall >= cvar.minimum_worst_client_recall
        and difference.macro_recall_percentage_points >= -cvar.maximum_macro_recall_loss
    )
    return AvailableFabridCvarGate(
        availability=EvidenceAvailability.AVAILABLE,
        status=_status(passing >= cvar.minimum_passing_budgets),
        budgets_passing=passing,
        total_budgets=cvar.total_budgets,
        differences=differences,
    )


def _budget_compliance(
    evaluations: tuple[SeedBudgetEvaluation, ...],
    policy: AllocationPolicy,
    budget_id: BudgetId,
    gates: PracticalGateConfig,
    expected_seed_count: SeedCount,
) -> BudgetComplianceResult:
    usages: list[BudgetUsageRatio] = []
    for evaluation in evaluations:
        if evaluation.experiment.budget_id is not budget_id:
            continue
        policy_result = evaluation.policy(policy)
        if not isinstance(policy_result, CompletedPolicyEvaluation):
            continue
        if policy_result.budget_usage is None:
            continue
        usages.append(policy_result.budget_usage)
    if len(usages) != expected_seed_count:
        return UnavailableBudgetCompliance(
            policy=policy,
            budget_id=budget_id,
            availability=EvidenceAvailability.INSUFFICIENT_EVIDENCE,
            reason=(
                f"budget compliance requires {expected_seed_count} seed values; got {len(usages)}"
            ),
        )
    raw = np.asarray(usages, dtype=np.float64)
    compliance = gates.budget_compliance
    median_usage = float(np.median(raw))
    fraction_within_limit = float(np.mean(raw <= compliance.seed_usage_limit))
    maximum_usage = float(np.max(raw))
    return AvailableBudgetCompliance(
        policy=policy,
        budget_id=budget_id,
        availability=EvidenceAvailability.AVAILABLE,
        status=_status(
            median_usage <= compliance.maximum_median_usage
            and fraction_within_limit >= compliance.minimum_seed_fraction_below_limit
        ),
        median_usage=median_usage,
        fraction_within_seed_limit=fraction_within_limit,
        maximum_usage=maximum_usage,
        seed_count=len(usages),
    )


def analyze_budget_compliance(
    evaluations: tuple[SeedBudgetEvaluation, ...],
    budgets: tuple[BudgetLevel, ...],
    gates: PracticalGateConfig,
    expected_seed_count: SeedCount,
) -> tuple[BudgetComplianceResult, ...]:
    return tuple(
        _budget_compliance(
            evaluations=evaluations,
            policy=policy,
            budget_id=level.budget_id,
            gates=gates,
            expected_seed_count=expected_seed_count,
        )
        for policy in _BUDGET_COMPLIANCE_POLICIES
        for level in budgets
        if level.budget_id in gates.budget_compliance.evaluated_budgets
    )


def analyze_practical_gates(
    evaluations: tuple[SeedBudgetEvaluation, ...],
    budgets: tuple[BudgetLevel, ...],
    statistics: StatisticsConfig,
    gates: PracticalGateConfig,
    expected_seed_count: SeedCount,
) -> PracticalGateAnalysis:
    guardrail = analyze_cvar_macro_guardrail(evaluations, budgets, statistics)
    return PracticalGateAnalysis(
        fabrid_macro=evaluate_fabrid_macro_gate(evaluations, gates),
        fabrid_cvar=evaluate_fabrid_cvar_gate(evaluations, guardrail, gates),
        budget_compliance=analyze_budget_compliance(
            evaluations=evaluations,
            budgets=budgets,
            gates=gates,
            expected_seed_count=expected_seed_count,
        ),
    )
