from __future__ import annotations

from fabrid.allocation.frontier_inputs import FrontierScoreArtifacts, build_client_frontier_inputs
from fabrid.artifacts.score import ScorePartitionArtifact, ScoreRecord
from fabrid.domain.coordinates import ScoreCoordinate
from fabrid.domain.enums import AttackSplit, BenignSplit, DatasetId, Label
from fabrid.domain.identifiers import AttackSubtypeId, ClientId, SampleId, SourceFileId
from fabrid.domain.values import AnomalyScore, DetectorSeed, SourceRowIndex
from fabrid.protocol.specification import PROTOCOL


def _coordinate() -> ScoreCoordinate:
    return ScoreCoordinate(DatasetId.NBAIOT, DetectorSeed(0), ClientId("device"))


def _record(
    sample: str,
    score: float,
    label: Label,
    subtype: AttackSubtypeId | None,
) -> ScoreRecord:
    return ScoreRecord(
        sample_id=SampleId(sample),
        source_file=SourceFileId("scores.csv"),
        source_row=SourceRowIndex(0),
        score=AnomalyScore(score),
        label=label,
        attack_subtype=subtype,
        timestamp=None,
    )


def test_frontier_inputs_use_only_frontier_and_attack_validation_roles() -> None:
    coordinate = _coordinate()
    benign = ScorePartitionArtifact(
        coordinate,
        BenignSplit.FRONTIER,
        tuple(
            _record(f"b-{index}", score, Label.BENIGN, None)
            for index, score in enumerate((0.1, 0.2, 0.3, 0.4, 0.5))
        ),
    )
    scan = AttackSubtypeId("scan")
    udp = AttackSubtypeId("udp")
    attack = ScorePartitionArtifact(
        coordinate,
        AttackSplit.VALIDATION,
        (
            _record("a-1", 0.25, Label.ATTACK, scan),
            _record("a-2", 0.45, Label.ATTACK, scan),
            _record("a-3", 0.35, Label.ATTACK, udp),
            _record("a-4", 0.55, Label.ATTACK, udp),
        ),
    )

    inputs = build_client_frontier_inputs(
        FrontierScoreArtifacts(benign, attack),
        PROTOCOL.alpha_grid,
    )

    assert inputs.client_id == ClientId("device")
    assert inputs.benign_frontier_scores.row_count.value == 5
    assert tuple(subtype.subtype for subtype in inputs.candidates[0].subtypes) == (scan, udp)


def test_frontier_true_positives_are_monotone_with_target_rate() -> None:
    coordinate = _coordinate()
    benign = ScorePartitionArtifact(
        coordinate,
        BenignSplit.FRONTIER,
        tuple(
            _record(f"b-{index}", score, Label.BENIGN, None)
            for index, score in enumerate((0.1, 0.2, 0.3, 0.4, 0.5))
        ),
    )
    scan = AttackSubtypeId("scan")
    attack = ScorePartitionArtifact(
        coordinate,
        AttackSplit.VALIDATION,
        tuple(
            _record(f"a-{index}", score, Label.ATTACK, scan)
            for index, score in enumerate((0.15, 0.25, 0.35, 0.45, 0.55))
        ),
    )

    inputs = build_client_frontier_inputs(
        FrontierScoreArtifacts(benign, attack),
        PROTOCOL.alpha_grid,
    )
    true_positives = tuple(
        candidate.subtypes[0].counts.true_positive.value
        for candidate in inputs.candidates
    )

    assert true_positives == tuple(sorted(true_positives))
