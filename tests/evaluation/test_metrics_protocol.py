from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from fabrid.allocation.optimization import not_applicable_solver_evidence
from fabrid.allocation.problem import (
    Allocation,
    AllocationDecision,
    AllocationWeights,
    ClientBudgetWeight,
    FederationWeights,
    equal_client_weights,
)
from fabrid.artifacts.paths import AllocationCoordinate, ExperimentCoordinate, ScoreCoordinate
from fabrid.config import (
    AllocationPolicy,
    AttackSplit,
    BenignSplit,
    BudgetId,
    ClientId,
    DatasetId,
    ExperimentId,
    ExperimentVariantId,
    Label,
    TargetFalsePositiveRate,
    WeightMode,
)
from fabrid.datasets.registry import ClientPopulation
from fabrid.detector.scoring import ScorePartitionArtifact
from fabrid.evaluation.metrics import (
    ClientFalseAlerts,
    ClientFalsePositiveRate,
    ClientMacroRecall,
    ClientOperatingPoint,
    CompletedPolicyEvaluation,
    ConfusionCounts,
    EvaluationProvenance,
    MetricArea,
    SubtypeRecall,
    assert_auroc_invariant,
    assert_policy_auroc_invariant,
    balanced_accuracy,
    budget_usage_ratio,
    budget_violation_ratio,
    client_macro_recall,
    compute_auprc,
    compute_auroc,
    evaluate_allocation,
    false_alert_gini,
    federation_fpr,
    federation_macro_recall,
    macro_f1,
    pooled_recall,
    worst_client_recall,
)

from ..allocation.synthetic_federation import synthetic_partition, synthetic_records


def _confusion(
    true_positive: int,
    false_negative: int,
    false_positive: int,
    true_negative: int,
) -> ConfusionCounts:
    return ConfusionCounts(
        true_positive=true_positive,
        false_negative=false_negative,
        false_positive=false_positive,
        true_negative=true_negative,
    )


def test_macro_f1_equals_per_class_f1_mean() -> None:
    confusions = (
        _confusion(80, 20, 10, 90),
        _confusion(60, 40, 30, 70),
    )
    value = macro_f1(confusions)
    first = 2 * 80 / (2 * 80 + 20 + 10)
    second = 2 * 60 / (2 * 60 + 40 + 30)
    assert value == pytest.approx((first + second) / 2)


def test_balanced_accuracy_equals_recall_specificity_mean() -> None:
    confusion = _confusion(80, 20, 10, 90)
    value = balanced_accuracy((confusion,))
    assert value == (80 / 100 + 90 / 100) / 2


def test_pooled_recall_is_total_detections_over_total_attacks() -> None:
    confusions = (
        _confusion(80, 20, 0, 0),
        _confusion(60, 40, 0, 0),
    )
    assert pooled_recall(confusions) == (80 + 60) / (100 + 100)


def test_false_alert_gini_zero_for_uniform_and_one_for_concentrated() -> None:
    uniform = tuple(ClientFalseAlerts(client_id=str(index), count=10) for index in range(4))
    assert false_alert_gini(uniform) == 0.0
    concentrated = (
        ClientFalseAlerts(client_id="a", count=40),
        ClientFalseAlerts(client_id="b", count=0),
        ClientFalseAlerts(client_id="c", count=0),
        ClientFalseAlerts(client_id="d", count=0),
    )
    assert false_alert_gini(concentrated) == 0.75


def test_budget_violation_ratio_is_excess_over_full_budget() -> None:
    assert budget_violation_ratio(0.5) == 0.0
    assert budget_violation_ratio(1.0) == 0.0
    assert budget_violation_ratio(1.2) == pytest.approx(0.2)
    assert budget_violation_ratio(None) is None


def test_compute_auroc_perfect_and_random() -> None:
    assert compute_auroc(np.asarray([0.0, 0.1]), np.asarray([0.9, 1.0])) == 1.0
    assert compute_auroc(np.asarray([0.0, 1.0]), np.asarray([0.0, 1.0])) == 0.5


def test_compute_auprc_perfect() -> None:
    assert compute_auprc(np.asarray([0.0, 0.1]), np.asarray([0.9, 1.0])) == 1.0


def test_t08_auroc_identity_across_frozen_score_policies() -> None:
    areas = tuple(
        MetricArea(policy=policy, area=0.87)
        for policy in (
            AllocationPolicy.EQ_FPR,
            AllocationPolicy.GREEDY,
            AllocationPolicy.FABRID_MACRO,
        )
    )
    assert_auroc_invariant(areas)


def test_assert_auroc_invariant_fails_when_areas_differ() -> None:
    areas = (
        MetricArea(policy=AllocationPolicy.EQ_FPR, area=0.87),
        MetricArea(policy=AllocationPolicy.GREEDY, area=0.87 + 1.0e-11),
    )
    with pytest.raises(ValueError, match="AUROC invariant violated"):
        assert_auroc_invariant(areas)


def test_client_macro_recall_is_unweighted_subtype_mean() -> None:
    recall = client_macro_recall(
        (
            SubtypeRecall(subtype="mirai_ack", rate=0.8),
            SubtypeRecall(subtype="bashlite_scan", rate=0.4),
        )
    )
    assert recall == pytest.approx(0.6)


def test_federation_macro_and_worst_client_recall() -> None:
    clients = (
        ClientMacroRecall(client_id="device_1", recall=0.9),
        ClientMacroRecall(client_id="device_2", recall=0.3),
    )
    assert federation_macro_recall(clients) == 0.6
    assert worst_client_recall(clients) == 0.3


def test_federation_fpr_is_weighted_client_sum() -> None:
    rates = (
        ClientFalsePositiveRate(client_id="device_1", rate=0.02),
        ClientFalsePositiveRate(client_id="device_2", rate=0.00),
    )
    weights = FederationWeights(
        AllocationWeights(
            (
                ClientBudgetWeight(client_id="device_1", weight=0.25),
                ClientBudgetWeight(client_id="device_2", weight=0.75),
            )
        )
    )
    assert federation_fpr(rates, weights) == 0.005


def test_budget_usage_ratio_is_rate_over_budget() -> None:
    assert budget_usage_ratio(0.01, 0.02) == 0.5
    assert budget_usage_ratio(0.0, 0.0) is None


def test_client_operating_point_validates_policy_and_rates() -> None:
    point = ClientOperatingPoint(
        client_id="device_1",
        policy=AllocationPolicy.FABRID_MACRO,
        target_rate=0.01,
        threshold=0.5,
        false_positive_rate=0.01,
        macro_attack_recall=0.9,
    )
    assert point.client_id == "device_1"
    assert point.false_positive_rate == 0.01


_KNOWN_CLIENT_A = "device_1"
_KNOWN_CLIENT_B = "device_2"
_KNOWN_TARGET_RATE = 0.5
_KNOWN_BUDGET = 0.375


@dataclass(frozen=True, slots=True)
class KnownScoreArtifacts:
    calibration_frontier: tuple[ScorePartitionArtifact, ...]
    final_calibration: tuple[ScorePartitionArtifact, ...]
    benign_test: tuple[ScorePartitionArtifact, ...]
    attack_test: tuple[ScorePartitionArtifact, ...]


def _known_coordinate(policy: AllocationPolicy) -> AllocationCoordinate:
    return AllocationCoordinate(
        experiment=ExperimentCoordinate(
            experiment_id=ExperimentId.MATCHED_BUDGET,
            variant_id=ExperimentVariantId.PRIMARY,
            dataset_id=DatasetId.NBAIOT,
            detector_seed=0,
            budget_id=BudgetId.FALSE_POSITIVE_0P020,
            budget=_KNOWN_BUDGET,
            weight_mode=WeightMode.EQUAL_CLIENT,
        ),
        policy=policy,
    )


def _known_provenance() -> EvaluationProvenance:
    return EvaluationProvenance(
        model_sha256="0" * 64,
        score_sha256="1" * 64,
        split_sha256="2" * 64,
        feature_sha256="3" * 64,
        protocol_sha256="4" * 64,
        git_commit="a" * 40,
    )


def _known_allocation(policy: AllocationPolicy, target_rate: TargetFalsePositiveRate) -> Allocation:
    return Allocation(
        policy=policy,
        decisions=(
            AllocationDecision(client_id=_KNOWN_CLIENT_A, target_rate=target_rate),
            AllocationDecision(client_id=_KNOWN_CLIENT_B, target_rate=target_rate),
        ),
    )


def _attack_test_partition(
    client_id: ClientId,
    bashlite_scores: np.ndarray,
    mirai_scores: np.ndarray,
) -> ScorePartitionArtifact:
    return ScorePartitionArtifact(
        coordinate=ScoreCoordinate(
            dataset_id=DatasetId.NBAIOT, detector_seed=0, client_id=client_id
        ),
        split=AttackSplit.TEST,
        records=synthetic_records(
            client_id,
            bashlite_scores,
            AttackSplit.TEST,
            label=Label.ATTACK,
            subtype="bashlite_scan",
        )
        + synthetic_records(
            client_id,
            mirai_scores,
            AttackSplit.TEST,
            label=Label.ATTACK,
            subtype="mirai_scan",
        ),
    )


def _known_score_artifacts() -> KnownScoreArtifacts:
    return KnownScoreArtifacts(
        calibration_frontier=(
            synthetic_partition(
                _KNOWN_CLIENT_A,
                np.asarray((0.3, 0.5, 0.7, 0.9, 1.0)),
                BenignSplit.FRONTIER,
                label=Label.BENIGN,
            ),
            synthetic_partition(
                _KNOWN_CLIENT_B,
                np.asarray((0.3, 0.5, 0.7, 0.9, 1.0)),
                BenignSplit.FRONTIER,
                label=Label.BENIGN,
            ),
        ),
        final_calibration=(
            synthetic_partition(
                _KNOWN_CLIENT_A,
                np.asarray((0.0, 0.1, 0.2, 0.3, 0.4)),
                BenignSplit.FINAL_CAL,
                label=Label.BENIGN,
            ),
            synthetic_partition(
                _KNOWN_CLIENT_B,
                np.asarray((0.0, 0.1, 0.2, 0.3, 0.4)),
                BenignSplit.FINAL_CAL,
                label=Label.BENIGN,
            ),
        ),
        benign_test=(
            synthetic_partition(
                _KNOWN_CLIENT_A,
                np.asarray((0.0, 0.05, 0.4, 0.6)),
                BenignSplit.TEST,
                label=Label.BENIGN,
            ),
            synthetic_partition(
                _KNOWN_CLIENT_B,
                np.asarray((0.0, 0.1, 0.2, 0.5)),
                BenignSplit.TEST,
                label=Label.BENIGN,
            ),
        ),
        attack_test=(
            _attack_test_partition(
                _KNOWN_CLIENT_A,
                np.asarray((0.7, 0.8)),
                np.asarray((0.2, 0.9)),
            ),
            _attack_test_partition(
                _KNOWN_CLIENT_B,
                np.asarray((0.8, 0.9)),
                np.asarray((0.3, 0.85)),
            ),
        ),
    )


def _evaluate_known_allocation(
    policy: AllocationPolicy,
    target_rate: TargetFalsePositiveRate,
) -> CompletedPolicyEvaluation:
    scores = _known_score_artifacts()
    summary, client_records = evaluate_allocation(
        _known_coordinate(policy),
        _known_allocation(policy, target_rate),
        scores.final_calibration,
        scores.benign_test,
        scores.attack_test,
        equal_client_weights(ClientPopulation(clients=(_KNOWN_CLIENT_A, _KNOWN_CLIENT_B))),
        not_applicable_solver_evidence(),
        0.0,
        _known_provenance(),
    )
    assert len(client_records) == 4
    return summary


def test_evaluate_allocation_fills_secondary_metrics_and_keeps_primaries() -> None:
    summary = _evaluate_known_allocation(AllocationPolicy.EQ_FPR, _KNOWN_TARGET_RATE)
    client_a = _confusion(3, 1, 2, 2)
    client_b = _confusion(4, 0, 1, 3)
    confusions = (client_a, client_b)
    scores = _known_score_artifacts()
    benign_scores = np.concatenate(
        tuple(artifact.score_values() for artifact in scores.benign_test)
    )
    attack_scores = np.concatenate(
        tuple(artifact.score_values() for artifact in scores.attack_test)
    )
    assert summary.macro_recall == pytest.approx(0.875)
    assert summary.worst_client_recall == pytest.approx(0.75)
    assert summary.federation_fpr == pytest.approx(0.375)
    assert summary.budget_usage == pytest.approx(0.375 / _KNOWN_BUDGET)
    assert summary.pooled_recall == pytest.approx(pooled_recall(confusions))
    assert summary.macro_f1 == pytest.approx(macro_f1(confusions))
    assert summary.balanced_accuracy == pytest.approx(balanced_accuracy(confusions))
    assert summary.auroc == pytest.approx(compute_auroc(benign_scores, attack_scores))
    assert summary.auprc == pytest.approx(compute_auprc(benign_scores, attack_scores))
    assert summary.false_alert_gini == pytest.approx(
        false_alert_gini(
            (
                ClientFalseAlerts(client_id=_KNOWN_CLIENT_A, count=2),
                ClientFalseAlerts(client_id=_KNOWN_CLIENT_B, count=1),
            )
        )
    )
    assert summary.budget_violation == 0.0


def test_evaluate_allocation_auroc_is_identical_across_policies() -> None:
    equal_fpr = _evaluate_known_allocation(AllocationPolicy.EQ_FPR, 0.5)
    greedy = _evaluate_known_allocation(AllocationPolicy.GREEDY, 0.2)
    assert equal_fpr.auroc == greedy.auroc
    assert equal_fpr.auprc == greedy.auprc
    assert_policy_auroc_invariant((equal_fpr, greedy))


def test_assert_policy_auroc_invariant_fails_when_aurocs_differ() -> None:
    equal_fpr = _evaluate_known_allocation(AllocationPolicy.EQ_FPR, 0.5)
    greedy = equal_fpr.model_copy(update={"policy": AllocationPolicy.GREEDY, "auroc": 0.5})
    with pytest.raises(ValueError, match="AUROC invariant violated"):
        assert_policy_auroc_invariant((equal_fpr, greedy))
