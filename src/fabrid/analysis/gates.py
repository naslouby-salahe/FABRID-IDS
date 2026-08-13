from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fabrid.analysis.budget_contrasts import (
    AvailableBudgetContrast,
    BudgetContrast,
    analyze_budget_contrast,
    budget_results,
)
from fabrid.analysis.primary import PrimaryInference
from fabrid.analysis.types import PercentagePointDifference
from fabrid.domain.enums import (
    AllocationPolicy,
    BudgetId,
    EvidenceAvailability,
    GateStatus,
    MetricId,
)
from fabrid.domain.identifiers import FailureReason
from fabrid.domain.values import (
    AnalysisSeed,
    BudgetUsageRatio,
    Probability,
    RowCount,
)
from fabrid.evaluation.results import CompletedPolicyEvaluation, SeedBudgetEvaluation
from fabrid.protocol.models import BudgetLevel, FabridProtocol

_PERCENTAGE_POINTS_PER_UNIT = 100.0
_BUDGET_COMPLIANCE_POLICIES = (
    AllocationPolicy.EQ_FPR,
    AllocationPolicy.GREEDY,
    AllocationPolicy.FABRID_MACRO,
    AllocationPolicy.FABRID_MINIMAX,
)


@dataclass(frozen=True, slots=True)
class BudgetDifference:
    budget_id: BudgetId
    difference: PercentagePointDifference


@dataclass(frozen=True, slots=True)
class AvailableFabridMacroGate:
    availability: EvidenceAvailability
    status: GateStatus
    budgets_passing: RowCount
    total_budgets: RowCount
    differences: tuple[BudgetDifference, ...]

    def __post_init__(self) -> None:
        if self.availability is not EvidenceAvailability.AVAILABLE:
            raise ValueError("available FABRID-Macro gate must be marked AVAILABLE")


@dataclass(frozen=True, slots=True)
class UnavailableFabridMacroGate:
    availability: EvidenceAvailability
    reason: FailureReason

    def __post_init__(self) -> None:
        if self.availability is not EvidenceAvailability.INSUFFICIENT_EVIDENCE:
            raise ValueError("unavailable FABRID-Macro gate must be insufficient evidence")


FabridMacroGateResult = AvailableFabridMacroGate | UnavailableFabridMacroGate


@dataclass(frozen=True, slots=True)
class MinimaxBudgetDifference:
    budget_id: BudgetId
    worst_client_recall: PercentagePointDifference
    macro_recall: PercentagePointDifference


@dataclass(frozen=True, slots=True)
class AvailableFabridMinimaxGate:
    availability: EvidenceAvailability
    status: GateStatus
    budgets_passing: RowCount
    total_budgets: RowCount
    differences: tuple[MinimaxBudgetDifference, ...]

    def __post_init__(self) -> None:
        if self.availability is not EvidenceAvailability.AVAILABLE:
            raise ValueError("available FABRID-Minimax gate must be marked AVAILABLE")


@dataclass(frozen=True, slots=True)
class UnavailableFabridMinimaxGate:
    availability: EvidenceAvailability
    reason: FailureReason

    def __post_init__(self) -> None:
        if self.availability is not EvidenceAvailability.INSUFFICIENT_EVIDENCE:
            raise ValueError("unavailable FABRID-Minimax gate must be insufficient evidence")


FabridMinimaxGateResult = AvailableFabridMinimaxGate | UnavailableFabridMinimaxGate


@dataclass(frozen=True, slots=True)
class AvailableBudgetCompliance:
    policy: AllocationPolicy
    budget_id: BudgetId
    availability: EvidenceAvailability
    status: GateStatus
    median_usage: BudgetUsageRatio
    fraction_within_seed_limit: Probability
    maximum_usage: BudgetUsageRatio
    seed_count: RowCount

    def __post_init__(self) -> None:
        if self.availability is not EvidenceAvailability.AVAILABLE:
            raise ValueError("available budget compliance must be marked AVAILABLE")


@dataclass(frozen=True, slots=True)
class UnavailableBudgetCompliance:
    policy: AllocationPolicy
    budget_id: BudgetId
    availability: EvidenceAvailability
    reason: FailureReason

    def __post_init__(self) -> None:
        if self.availability is not EvidenceAvailability.INSUFFICIENT_EVIDENCE:
            raise ValueError("unavailable budget compliance must be insufficient evidence")


BudgetComplianceResult = AvailableBudgetCompliance | UnavailableBudgetCompliance


@dataclass(frozen=True, slots=True)
class PracticalGateAnalysis:
    fabrid_macro: FabridMacroGateResult
    fabrid_minimax: FabridMinimaxGateResult
    budget_compliance: tuple[BudgetComplianceResult, ...]


def _status(passed: bool) -> GateStatus:
    return GateStatus.PASS if passed else GateStatus.FAIL


def _percentage_point_difference(
    budget: AvailableBudgetContrast,
) -> PercentagePointDifference:
    return PercentagePointDifference(
        budget.contrast.bootstrap.mean_difference.value * _PERCENTAGE_POINTS_PER_UNIT
    )


def evaluate_fabrid_macro_gate(
    primary: PrimaryInference,
    protocol: FabridProtocol,
) -> FabridMacroGateResult:
    family = primary.macro_recall
    available = tuple(
        budget for budget in family.budgets if isinstance(budget, AvailableBudgetContrast)
    )
    gate = protocol.practical_gates.fabrid_macro
    if len(available) != gate.total_budgets.value:
        return UnavailableFabridMacroGate(
            availability=EvidenceAvailability.INSUFFICIENT_EVIDENCE,
            reason=FailureReason("FABRID-Macro gate requires all five budget effects"),
        )

    differences = tuple(
        BudgetDifference(
            budget_id=budget.budget_id,
            difference=_percentage_point_difference(budget),
        )
        for budget in available
    )
    passing = RowCount(
        sum(
            1
            for item in differences
            if item.difference.value >= gate.minimum_macro_recall_gain.value
        )
    )
    return AvailableFabridMacroGate(
        availability=EvidenceAvailability.AVAILABLE,
        status=_status(passing.value >= gate.minimum_passing_budgets.value),
        budgets_passing=passing,
        total_budgets=gate.total_budgets,
        differences=differences,
    )


def _minimax_macro_guardrail(
    evaluations: tuple[SeedBudgetEvaluation, ...],
    protocol: FabridProtocol,
) -> tuple[BudgetContrast, ...]:
    seed_offset = protocol.statistics.holm_family_size.value
    return tuple(
        analyze_budget_contrast(
            evaluations=evaluations,
            level=level,
            treatment=AllocationPolicy.FABRID_MINIMAX,
            baseline=AllocationPolicy.EQ_FPR,
            metric=MetricId.MACRO_RECALL,
            protocol=protocol,
            analysis_seed=AnalysisSeed(seed_offset + index),
        )
        for index, level in enumerate(protocol.budgets)
    )


def evaluate_fabrid_minimax_gate(
    evaluations: tuple[SeedBudgetEvaluation, ...],
    primary: PrimaryInference,
    protocol: FabridProtocol,
) -> FabridMinimaxGateResult:
    worst = tuple(
        budget
        for budget in primary.worst_client_recall.budgets
        if isinstance(budget, AvailableBudgetContrast)
    )
    macro_guardrail = _minimax_macro_guardrail(evaluations, protocol)
    macro = tuple(
        budget
        for budget in macro_guardrail
        if isinstance(budget, AvailableBudgetContrast)
    )
    gate = protocol.practical_gates.fabrid_minimax
    if len(worst) != gate.total_budgets.value or len(macro) != gate.total_budgets.value:
        return UnavailableFabridMinimaxGate(
            availability=EvidenceAvailability.INSUFFICIENT_EVIDENCE,
            reason=FailureReason(
                "FABRID-Minimax gate requires all five worst-client and MacroRecall effects"
            ),
        )

    differences = tuple(
        MinimaxBudgetDifference(
            budget_id=worst_budget.budget_id,
            worst_client_recall=_percentage_point_difference(worst_budget),
            macro_recall=_percentage_point_difference(macro_budget),
        )
        for worst_budget, macro_budget in zip(worst, macro, strict=True)
    )
    passing = RowCount(
        sum(
            1
            for item in differences
            if item.worst_client_recall.value
            >= gate.minimum_worst_client_recall_gain.value
            and item.macro_recall.value >= -gate.maximum_macro_recall_loss.value
        )
    )
    return AvailableFabridMinimaxGate(
        availability=EvidenceAvailability.AVAILABLE,
        status=_status(passing.value >= gate.minimum_passing_budgets.value),
        budgets_passing=passing,
        total_budgets=gate.total_budgets,
        differences=differences,
    )


def _budget_compliance(
    evaluations: tuple[SeedBudgetEvaluation, ...],
    level: BudgetLevel,
    policy: AllocationPolicy,
    protocol: FabridProtocol,
) -> BudgetComplianceResult:
    results = budget_results(evaluations, level)
    usages: list[BudgetUsageRatio] = []
    for result in results:
        policy_result = result.policy(policy)
        if not isinstance(policy_result, CompletedPolicyEvaluation):
            continue
        if policy_result.budget_usage is None:
            continue
        usages.append(policy_result.budget_usage)

    expected_seed_count = len(protocol.detector.seeds)
    if len(usages) != expected_seed_count:
        return UnavailableBudgetCompliance(
            policy=policy,
            budget_id=level.budget_id,
            availability=EvidenceAvailability.INSUFFICIENT_EVIDENCE,
            reason=FailureReason(
                f"budget compliance requires {expected_seed_count} seed values; got {len(usages)}"
            ),
        )

    raw = np.asarray(tuple(usage.value for usage in usages), dtype=np.float64)
    gate = protocol.practical_gates.budget_compliance
    median_usage = BudgetUsageRatio(float(np.median(raw)))
    fraction_within_limit = Probability(
        float(np.mean(raw <= gate.seed_usage_limit.value))
    )
    maximum_usage = BudgetUsageRatio(float(np.max(raw)))
    return AvailableBudgetCompliance(
        policy=policy,
        budget_id=level.budget_id,
        availability=EvidenceAvailability.AVAILABLE,
        status=_status(
            median_usage.value <= gate.maximum_median_usage.value
            and fraction_within_limit.value
            >= gate.minimum_seed_fraction_below_limit.value
        ),
        median_usage=median_usage,
        fraction_within_seed_limit=fraction_within_limit,
        maximum_usage=maximum_usage,
        seed_count=RowCount(len(usages)),
    )


def analyze_budget_compliance(
    evaluations: tuple[SeedBudgetEvaluation, ...],
    protocol: FabridProtocol,
) -> tuple[BudgetComplianceResult, ...]:
    return tuple(
        _budget_compliance(evaluations, level, policy, protocol)
        for policy in _BUDGET_COMPLIANCE_POLICIES
        for level in protocol.budgets
    )


def analyze_practical_gates(
    evaluations: tuple[SeedBudgetEvaluation, ...],
    primary: PrimaryInference,
    protocol: FabridProtocol,
) -> PracticalGateAnalysis:
    return PracticalGateAnalysis(
        fabrid_macro=evaluate_fabrid_macro_gate(primary, protocol),
        fabrid_minimax=evaluate_fabrid_minimax_gate(
            evaluations,
            primary,
            protocol,
        ),
        budget_compliance=analyze_budget_compliance(evaluations, protocol),
    )
