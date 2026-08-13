from __future__ import annotations

from dataclasses import replace

from fabrid.analysis.gates import (
    AvailableBudgetCompliance,
    AvailableFabridMacroGate,
    AvailableFabridMinimaxGate,
    analyze_practical_gates,
)
from fabrid.analysis.primary import analyze_primary_inference
from fabrid.domain.coordinates import ExperimentCoordinate
from fabrid.domain.enums import (
    AllocationPolicy,
    DatasetId,
    EvidenceAvailability,
    ExperimentId,
    ExperimentVariantId,
    GateStatus,
    WeightMode,
)
from fabrid.domain.identifiers import CampaignId
from fabrid.domain.values import (
    BudgetUsageRatio,
    DetectorSeed,
    FalsePositiveRate,
    MacroRecall,
    MetricDifference,
    Probability,
    RowCount,
    WorstClientRecall,
)
from fabrid.evaluation.results import CompletedPolicyEvaluation, SeedBudgetEvaluation
from fabrid.protocol.models import BudgetLevel, FabridProtocol
from fabrid.protocol.specification import PROTOCOL


def _test_protocol() -> FabridProtocol:
    return replace(
        PROTOCOL,
        statistics=replace(PROTOCOL.statistics, bootstrap_resamples=RowCount(200)),
    )


def _coordinate(seed: DetectorSeed, level: BudgetLevel) -> ExperimentCoordinate:
    return ExperimentCoordinate(
        campaign_id=CampaignId("gate-test"),
        experiment_id=ExperimentId.MATCHED_BUDGET,
        variant_id=ExperimentVariantId.PRIMARY,
        dataset_id=DatasetId.NBAIOT,
        detector_seed=seed,
        budget_id=level.budget_id,
        budget=level.value,
        weight_mode=WeightMode.EQUAL_CLIENT,
    )


def _policy(
    policy: AllocationPolicy,
    macro: MacroRecall,
    worst: WorstClientRecall,
    usage: BudgetUsageRatio,
) -> CompletedPolicyEvaluation:
    return CompletedPolicyEvaluation(
        policy=policy,
        macro_recall=macro,
        worst_client_recall=worst,
        federation_fpr=FalsePositiveRate(0.01),
        budget_usage=usage,
        fallback_rate=Probability(0.0),
    )


def _evaluations(
    protocol: FabridProtocol,
    macro_gain: MetricDifference,
    minimax_worst_gain: MetricDifference,
    minimax_macro_gain: MetricDifference,
    usage: BudgetUsageRatio,
) -> tuple[SeedBudgetEvaluation, ...]:
    evaluations: list[SeedBudgetEvaluation] = []
    baseline_macro = MacroRecall(0.60)
    baseline_worst = WorstClientRecall(0.40)

    for seed in protocol.detector.seeds:
        for level in protocol.budgets:
            evaluations.append(
                SeedBudgetEvaluation(
                    experiment=_coordinate(seed, level),
                    policies=(
                        _policy(AllocationPolicy.EQ_FPR, baseline_macro, baseline_worst, usage),
                        _policy(AllocationPolicy.GREEDY, baseline_macro, baseline_worst, usage),
                        _policy(
                            AllocationPolicy.FABRID_MACRO,
                            MacroRecall(baseline_macro.value + macro_gain.value),
                            baseline_worst,
                            usage,
                        ),
                        _policy(
                            AllocationPolicy.FABRID_MINIMAX,
                            MacroRecall(baseline_macro.value + minimax_macro_gain.value),
                            WorstClientRecall(
                                baseline_worst.value + minimax_worst_gain.value
                            ),
                            usage,
                        ),
                    ),
                )
            )

    return tuple(evaluations)


def test_practical_gates_pass_for_pre_registered_effect_sizes() -> None:
    protocol = _test_protocol()
    evaluations = _evaluations(
        protocol=protocol,
        macro_gain=MetricDifference(0.03),
        minimax_worst_gain=MetricDifference(0.08),
        minimax_macro_gain=MetricDifference(0.0),
        usage=BudgetUsageRatio(1.0),
    )
    primary = analyze_primary_inference(evaluations, protocol)

    result = analyze_practical_gates(evaluations, primary, protocol)

    assert isinstance(result.fabrid_macro, AvailableFabridMacroGate)
    assert result.fabrid_macro.availability is EvidenceAvailability.AVAILABLE
    assert result.fabrid_macro.status is GateStatus.PASS
    assert result.fabrid_macro.budgets_passing == RowCount(5)

    assert isinstance(result.fabrid_minimax, AvailableFabridMinimaxGate)
    assert result.fabrid_minimax.availability is EvidenceAvailability.AVAILABLE
    assert result.fabrid_minimax.status is GateStatus.PASS
    assert result.fabrid_minimax.budgets_passing == RowCount(5)

    assert len(result.budget_compliance) == 20
    assert all(
        isinstance(item, AvailableBudgetCompliance)
        and item.status is GateStatus.PASS
        for item in result.budget_compliance
    )
