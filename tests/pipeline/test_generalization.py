from __future__ import annotations

from fabrid.allocation.frontier_inputs import FrontierScoreArtifacts
from fabrid.artifacts.score import ScorePartitionArtifact, ScoreRecord
from fabrid.domain.coordinates import ScoreCoordinate
from fabrid.domain.enums import AttackSplit, BenignSplit, DatasetId, Label
from fabrid.domain.identifiers import (
    AttackSubtypeId,
    ClientId,
    SampleId,
    SourceFileId,
)
from fabrid.domain.population import ClientPopulation
from fabrid.domain.values import AnomalyScore, DetectorSeed, SourceRowIndex
from fabrid.evaluation.evaluator import ClientEvaluationArtifacts
from fabrid.pipeline.allocation import LoadedClientScores, LoadedSeedScores
from fabrid.pipeline.generalization import restrict_generalization_scores
from fabrid.protocol.specification import PROTOCOL

_CLIENT = ClientId("test-client")
_COORDINATE = ScoreCoordinate(
    dataset_id=DatasetId.NBAIOT,
    detector_seed=DetectorSeed(0),
    client_id=_CLIENT,
)
_BASHLITE_SCAN = AttackSubtypeId("bashlite_scan")
_MIRAI_ACK = AttackSubtypeId("mirai_ack")


def _record(
    sample_id: str,
    source_row: int,
    score: float,
    label: Label,
    subtype: AttackSubtypeId | None,
) -> ScoreRecord:
    return ScoreRecord(
        sample_id=SampleId(sample_id),
        source_file=SourceFileId("synthetic.csv"),
        source_row=SourceRowIndex(source_row),
        score=AnomalyScore(score),
        label=label,
        attack_subtype=subtype,
        timestamp=None,
    )


def _benign_partition(split: BenignSplit, row: int) -> ScorePartitionArtifact:
    return ScorePartitionArtifact(
        coordinate=_COORDINATE,
        split=split,
        records=(
            _record(
                sample_id=f"{split.value}-{row}",
                source_row=row,
                score=0.1,
                label=Label.BENIGN,
                subtype=None,
            ),
        ),
    )


def _attack_partition(split: AttackSplit, row_offset: int) -> ScorePartitionArtifact:
    return ScorePartitionArtifact(
        coordinate=_COORDINATE,
        split=split,
        records=(
            _record(
                sample_id=f"{split.value}-bashlite",
                source_row=row_offset,
                score=1.0,
                label=Label.ATTACK,
                subtype=_BASHLITE_SCAN,
            ),
            _record(
                sample_id=f"{split.value}-mirai",
                source_row=row_offset + 1,
                score=1.1,
                label=Label.ATTACK,
                subtype=_MIRAI_ACK,
            ),
        ),
    )


def _scores() -> LoadedSeedScores:
    benign_frontier = _benign_partition(BenignSplit.FRONTIER, 10)
    final_calibration = _benign_partition(BenignSplit.FINAL_CAL, 20)
    benign_test = _benign_partition(BenignSplit.TEST, 30)
    return LoadedSeedScores(
        (
            LoadedClientScores(
                client_id=_CLIENT,
                frontier=FrontierScoreArtifacts(
                    benign_frontier=benign_frontier,
                    attack_validation=_attack_partition(AttackSplit.VALIDATION, 40),
                ),
                evaluation=ClientEvaluationArtifacts(
                    final_calibration=final_calibration,
                    benign_test=benign_test,
                    attack_test=_attack_partition(AttackSplit.TEST, 50),
                ),
            ),
        )
    )


def test_generalization_filters_only_attack_roles() -> None:
    source = _scores()
    restricted = restrict_generalization_scores(
        scores=source,
        population=ClientPopulation((_CLIENT,)),
        validation_subtypes=(_BASHLITE_SCAN,),
        test_subtypes=(_MIRAI_ACK,),
    )

    original = source.for_client(_CLIENT)
    filtered = restricted.for_client(_CLIENT)

    assert filtered.frontier.benign_frontier is original.frontier.benign_frontier
    assert filtered.evaluation.final_calibration is original.evaluation.final_calibration
    assert filtered.evaluation.benign_test is original.evaluation.benign_test
    assert tuple(
        record.attack_subtype for record in filtered.frontier.attack_validation.records
    ) == (_BASHLITE_SCAN,)
    assert tuple(
        record.attack_subtype for record in filtered.evaluation.attack_test.records
    ) == (_MIRAI_ACK,)


def test_frozen_subtype_rotations_are_disjoint() -> None:
    for rotation in PROTOCOL.generalization.rotations:
        validation = PROTOCOL.generalization.fold(rotation.validation_fold).subtypes
        test = tuple(
            subtype
            for fold_id in rotation.test_folds
            for subtype in PROTOCOL.generalization.fold(fold_id).subtypes
        )
        assert all(subtype not in test for subtype in validation)


def test_botnet_family_directions_are_disjoint() -> None:
    for direction in PROTOCOL.generalization.family_directions:
        validation = PROTOCOL.generalization.family(direction.validation_family).subtypes
        test = PROTOCOL.generalization.family(direction.test_family).subtypes
        assert all(subtype not in test for subtype in validation)
