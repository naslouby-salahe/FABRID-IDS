from __future__ import annotations

import pytest

from fabrid.data.partitioner import AttackSplit, BenignSplit, RowCount
from fabrid.evaluation.record_level import AttackSubtype, ClientId
from fabrid.schemas.score_artifact import DetectorSeed, Label, ScoreArtifact, ScoreRecord


def _benign_record(sample_id: str, score: float) -> ScoreRecord:
    return ScoreRecord(
        sample_id=sample_id,
        source_file="benign.csv",
        source_row=RowCount(0),
        split_id=BenignSplit.TEST,
        score=score,
        label=Label.BENIGN,
        attack_type=None,
        timestamp=None,
    )


def _attack_record(sample_id: str, score: float, subtype: str) -> ScoreRecord:
    return ScoreRecord(
        sample_id=sample_id,
        source_file="attack.csv",
        source_row=RowCount(0),
        split_id=AttackSplit.TEST,
        score=score,
        label=Label.ATTACK,
        attack_type=AttackSubtype(subtype),
        timestamp=None,
    )


def test_attack_record_requires_attack_type() -> None:
    with pytest.raises(ValueError):
        ScoreRecord(
            sample_id="1",
            source_file="f.csv",
            source_row=RowCount(0),
            split_id=AttackSplit.TEST,
            score=1.0,
            label=Label.ATTACK,
            attack_type=None,
            timestamp=None,
        )


def test_benign_record_rejects_attack_type() -> None:
    with pytest.raises(ValueError):
        ScoreRecord(
            sample_id="1",
            source_file="f.csv",
            source_row=RowCount(0),
            split_id=BenignSplit.TEST,
            score=1.0,
            label=Label.BENIGN,
            attack_type=AttackSubtype("mirai_scan"),
            timestamp=None,
        )


def test_negative_detector_seed_rejected() -> None:
    with pytest.raises(ValueError):
        DetectorSeed(-1)


def test_duplicate_sample_id_rejected() -> None:
    with pytest.raises(ValueError):
        ScoreArtifact(
            dataset_id="n-baiot",
            detector_seed=DetectorSeed(0),
            client_id=ClientId("1"),
            records=(_benign_record("s1", 0.1), _benign_record("s1", 0.2)),
        )


def test_sha256_is_deterministic_and_order_independent() -> None:
    records_a = (_benign_record("s1", 0.1), _attack_record("s2", 0.9, "mirai_scan"))
    records_b = tuple(reversed(records_a))
    artifact_a = ScoreArtifact("n-baiot", DetectorSeed(0), ClientId("1"), records_a)
    artifact_b = ScoreArtifact("n-baiot", DetectorSeed(0), ClientId("1"), records_b)
    assert artifact_a.sha256() == artifact_b.sha256()


def test_sha256_changes_when_a_score_changes() -> None:
    base = ScoreArtifact("n-baiot", DetectorSeed(0), ClientId("1"), (_benign_record("s1", 0.1),))
    changed = ScoreArtifact("n-baiot", DetectorSeed(0), ClientId("1"), (_benign_record("s1", 0.2),))
    assert base.sha256() != changed.sha256()


def test_scores_by_split_filters_correctly() -> None:
    artifact = ScoreArtifact(
        "n-baiot",
        DetectorSeed(0),
        ClientId("1"),
        (_benign_record("s1", 0.1), _attack_record("s2", 0.9, "mirai_scan")),
    )
    assert len(artifact.scores_by_split(BenignSplit.TEST)) == 1
    assert len(artifact.scores_by_split(AttackSplit.TEST)) == 1
