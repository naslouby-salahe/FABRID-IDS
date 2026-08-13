from __future__ import annotations

import pytest

from fabrid.artifacts.score import ScorePartitionArtifact, ScoreRecord
from fabrid.domain.coordinates import ScoreCoordinate
from fabrid.domain.enums import AttackSplit, BenignSplit, DatasetId, Label
from fabrid.domain.identifiers import AttackSubtypeId, ClientId, SampleId, SourceFileId
from fabrid.domain.values import AnomalyScore, DetectorSeed, SourceRowIndex


def _coordinate() -> ScoreCoordinate:
    return ScoreCoordinate(
        dataset_id=DatasetId.NBAIOT,
        detector_seed=DetectorSeed(0),
        client_id=ClientId("device"),
    )


def _benign(sample_id: SampleId, score: AnomalyScore) -> ScoreRecord:
    return ScoreRecord(
        sample_id=sample_id,
        source_file=SourceFileId("benign.csv"),
        source_row=SourceRowIndex(0),
        score=score,
        label=Label.BENIGN,
        attack_subtype=None,
        timestamp=None,
    )


def _attack(sample_id: SampleId, score: AnomalyScore) -> ScoreRecord:
    return ScoreRecord(
        sample_id=sample_id,
        source_file=SourceFileId("attack.csv"),
        source_row=SourceRowIndex(0),
        score=score,
        label=Label.ATTACK,
        attack_subtype=AttackSubtypeId("mirai_scan"),
        timestamp=None,
    )


def test_score_record_enforces_label_subtype_contract() -> None:
    with pytest.raises(ValueError):
        ScoreRecord(
            sample_id=SampleId("attack-without-subtype"),
            source_file=SourceFileId("attack.csv"),
            source_row=SourceRowIndex(0),
            score=AnomalyScore(1.0),
            label=Label.ATTACK,
            attack_subtype=None,
            timestamp=None,
        )
    with pytest.raises(ValueError):
        ScoreRecord(
            sample_id=SampleId("benign-with-subtype"),
            source_file=SourceFileId("benign.csv"),
            source_row=SourceRowIndex(0),
            score=AnomalyScore(0.1),
            label=Label.BENIGN,
            attack_subtype=AttackSubtypeId("mirai_scan"),
            timestamp=None,
        )


def test_score_partition_rejects_duplicate_samples_and_wrong_label_role() -> None:
    record = _benign(SampleId("same"), AnomalyScore(0.1))
    with pytest.raises(ValueError):
        ScorePartitionArtifact(_coordinate(), BenignSplit.TEST, (record, record))
    with pytest.raises(ValueError):
        ScorePartitionArtifact(
            _coordinate(),
            BenignSplit.TEST,
            (_attack(SampleId("attack"), AnomalyScore(0.9)),),
        )
    with pytest.raises(ValueError):
        ScorePartitionArtifact(
            _coordinate(),
            AttackSplit.TEST,
            (_benign(SampleId("benign"), AnomalyScore(0.1)),),
        )


def test_score_partition_digest_is_order_independent_and_content_sensitive() -> None:
    benign = _benign(SampleId("benign"), AnomalyScore(0.1))
    second = _benign(SampleId("second"), AnomalyScore(0.2))
    first_order = ScorePartitionArtifact(_coordinate(), BenignSplit.TEST, (benign, second))
    reversed_order = ScorePartitionArtifact(_coordinate(), BenignSplit.TEST, (second, benign))
    changed = ScorePartitionArtifact(
        _coordinate(),
        BenignSplit.TEST,
        (_benign(SampleId("benign"), AnomalyScore(0.3)), second),
    )

    assert first_order.digest() == reversed_order.digest()
    assert first_order.digest() != changed.digest()
