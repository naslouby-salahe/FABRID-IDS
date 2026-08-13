from __future__ import annotations

import pytest

from fabrid.allocation.frontier_inputs import FrontierScoreArtifacts, build_client_frontier_inputs
from fabrid.artifacts.score import ScorePartitionArtifact, ScoreRecord
from fabrid.domain.coordinates import ScoreCoordinate
from fabrid.domain.enums import AttackSplit, BenignSplit, DatasetId, Label
from fabrid.domain.identifiers import AttackSubtypeId, ClientId, SampleId, SourceFileId
from fabrid.domain.values import AnomalyScore, DetectorSeed, SourceRowIndex
from fabrid.protocol.specification import PROTOCOL


def _coordinate() -> ScoreCoordinate:
    return ScoreCoordinate(DatasetId.NBAIOT, DetectorSeed(0), ClientId("device"))


def _partition(
    split: BenignSplit | AttackSplit,
    values: tuple[float, ...],
) -> ScorePartitionArtifact:
    is_attack = isinstance(split, AttackSplit)
    subtype = AttackSubtypeId("scan") if is_attack else None
    label = Label.ATTACK if is_attack else Label.BENIGN
    return ScorePartitionArtifact(
        coordinate=_coordinate(),
        split=split,
        records=tuple(
            ScoreRecord(
                sample_id=SampleId(f"{split.value}-{index}"),
                source_file=SourceFileId("scores.csv"),
                source_row=SourceRowIndex(index),
                score=AnomalyScore(value),
                label=label,
                attack_subtype=subtype,
                timestamp=None,
            )
            for index, value in enumerate(values)
        ),
    )


def test_allocation_input_contract_rejects_test_and_final_calibration_splits() -> None:
    frontier = _partition(BenignSplit.FRONTIER, (0.1, 0.2, 0.3, 0.4, 0.5))
    validation = _partition(AttackSplit.VALIDATION, (0.5, 0.6, 0.55, 0.62, 0.58))

    FrontierScoreArtifacts(frontier, validation)

    with pytest.raises(ValueError):
        FrontierScoreArtifacts(_partition(BenignSplit.TEST, (0.1,)), validation)
    with pytest.raises(ValueError):
        FrontierScoreArtifacts(frontier, _partition(AttackSplit.TEST, (0.9,)))
    with pytest.raises(ValueError):
        FrontierScoreArtifacts(_partition(BenignSplit.FINAL_CAL, (0.2,)), validation)


def test_attack_validation_perturbation_changes_only_utility_inputs() -> None:
    frontier = _partition(BenignSplit.FRONTIER, (0.1, 0.2, 0.3, 0.4, 0.5))
    baseline = build_client_frontier_inputs(
        FrontierScoreArtifacts(
            frontier,
            _partition(AttackSplit.VALIDATION, (0.5, 0.6, 0.55, 0.62, 0.58)),
        ),
        PROTOCOL.alpha_grid,
    )
    perturbed = build_client_frontier_inputs(
        FrontierScoreArtifacts(
            frontier,
            _partition(AttackSplit.VALIDATION, (0.1, 0.15, 0.12, 0.18, 0.11)),
        ),
        PROTOCOL.alpha_grid,
    )

    assert baseline.benign_frontier_scores is perturbed.benign_frontier_scores
    assert baseline.candidates != perturbed.candidates


def test_test_score_perturbation_cannot_change_frontier_inputs() -> None:
    frontier = _partition(BenignSplit.FRONTIER, (0.1, 0.2, 0.3, 0.4, 0.5))
    validation = _partition(AttackSplit.VALIDATION, (0.5, 0.6, 0.55, 0.62, 0.58))
    before = build_client_frontier_inputs(
        FrontierScoreArtifacts(frontier, validation),
        PROTOCOL.alpha_grid,
    )

    _partition(BenignSplit.TEST, (0.99, 0.01))
    _partition(AttackSplit.TEST, (0.02, 0.98))
    after = build_client_frontier_inputs(
        FrontierScoreArtifacts(frontier, validation),
        PROTOCOL.alpha_grid,
    )

    assert before == after
