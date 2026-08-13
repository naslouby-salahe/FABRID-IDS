from __future__ import annotations

import pytest

from fabrid.allocation.baselines.pooled_shared import (
    ClientPooledValidation,
    FederationPooledValidation,
    pooled_candidate_thresholds,
    select_pooled_shared_threshold,
)
from fabrid.allocation.contracts import (
    AllocationWeights,
    ClientBudgetWeight,
    ClientUtilityCurve,
    ClientUtilityCurves,
    ClientUtilityPoint,
    FederationWeights,
)
from fabrid.allocation.diagnostics.test_oracle import (
    OracleAccessToken,
    OracleAuthorization,
    allocate_test_oracle,
)
from fabrid.artifacts.score import ScorePartitionArtifact, ScoreRecord
from fabrid.domain.coordinates import ScoreCoordinate
from fabrid.domain.enums import AllocationPolicy, AttackSplit, BenignSplit, DatasetId, Label
from fabrid.domain.identifiers import AttackSubtypeId, ClientId, SampleId, SourceFileId
from fabrid.domain.values import (
    AnomalyScore,
    ClientWeight,
    DetectionUtility,
    DetectorSeed,
    FalsePositiveBudget,
    SourceRowIndex,
    TargetFalsePositiveRate,
)
from fabrid.protocol.specification import PROTOCOL


def _partition(
    split: BenignSplit | AttackSplit,
    scores: tuple[float, ...],
) -> ScorePartitionArtifact:
    coordinate = ScoreCoordinate(DatasetId.NBAIOT, DetectorSeed(0), ClientId("client"))
    is_attack = isinstance(split, AttackSplit)
    return ScorePartitionArtifact(
        coordinate=coordinate,
        split=split,
        records=tuple(
            ScoreRecord(
                sample_id=SampleId(f"{split.value}-{index}"),
                source_file=SourceFileId("scores.csv"),
                source_row=SourceRowIndex(index),
                score=AnomalyScore(score),
                label=Label.ATTACK if is_attack else Label.BENIGN,
                attack_subtype=AttackSubtypeId("scan") if is_attack else None,
                timestamp=None,
            )
            for index, score in enumerate(scores)
        ),
    )


def test_pooled_shared_selects_best_budget_feasible_threshold() -> None:
    validation = FederationPooledValidation(
        (
            ClientPooledValidation(
                client_id=ClientId("client"),
                benign_frontier=_partition(BenignSplit.FRONTIER, (0.1, 0.2, 0.3, 0.9)),
                attack_validation=_partition(AttackSplit.VALIDATION, (0.5, 0.6, 0.95)),
            ),
        )
    )
    weights = FederationWeights(
        AllocationWeights(
            (ClientBudgetWeight(ClientId("client"), ClientWeight(1.0)),)
        )
    )

    result = select_pooled_shared_threshold(
        validation,
        weights,
        FalsePositiveBudget(0.25),
    )

    assert result.threshold.value == pytest.approx(0.3)
    assert result.macro_recall.value == pytest.approx(1.0)
    assert result.federation_fpr.value == pytest.approx(0.25)
    assert result.threshold in pooled_candidate_thresholds(validation)


def test_test_oracle_requires_explicit_non_deployable_enum_and_preserves_label() -> None:
    curve = ClientUtilityCurve(
        ClientId("client"),
        (
            ClientUtilityPoint(TargetFalsePositiveRate(0.0), DetectionUtility(0.0)),
            ClientUtilityPoint(TargetFalsePositiveRate(0.01), DetectionUtility(0.5)),
            ClientUtilityPoint(TargetFalsePositiveRate(0.02), DetectionUtility(1.0)),
        ),
    )
    weights = AllocationWeights(
        (ClientBudgetWeight(ClientId("client"), ClientWeight(1.0)),)
    )

    optimized = allocate_test_oracle(
        OracleAccessToken(OracleAuthorization.ACKNOWLEDGED_NON_DEPLOYABLE),
        ClientUtilityCurves((curve,)),
        weights,
        FalsePositiveBudget(0.02),
        PROTOCOL.solver,
    )

    assert optimized.allocation.policy is AllocationPolicy.TEST_ORACLE
    assert optimized.allocation.decision(ClientId("client")).target_rate == TargetFalsePositiveRate(0.02)
