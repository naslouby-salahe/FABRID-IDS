from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fabrid.allocation.frontier import (
    CandidateConfusions,
    ClientFrontierInputs,
    SubtypeConfusion,
    SubtypeConfusionCounts,
)
from fabrid.artifacts.score import ScorePartitionArtifact
from fabrid.calibration.order_statistic import alerts_above_threshold, calibrate_threshold
from fabrid.domain.enums import AttackSplit, BenignSplit, Label
from fabrid.domain.identifiers import AttackSubtypeId, ClientId
from fabrid.domain.scores import ScoreVector
from fabrid.domain.values import RowCount
from fabrid.protocol.models import AlphaGrid


@dataclass(frozen=True, slots=True)
class AttackSubtypeScores:
    subtype: AttackSubtypeId
    scores: ScoreVector


@dataclass(frozen=True, slots=True)
class FrontierScoreArtifacts:
    benign_frontier: ScorePartitionArtifact
    attack_validation: ScorePartitionArtifact

    def __post_init__(self) -> None:
        if self.benign_frontier.split is not BenignSplit.FRONTIER:
            raise ValueError("frontier inputs require a BENIGN_FRONTIER artifact")
        if self.attack_validation.split is not AttackSplit.VALIDATION:
            raise ValueError("frontier inputs require an ATTACK_VALIDATION artifact")
        if self.benign_frontier.coordinate != self.attack_validation.coordinate:
            raise ValueError("frontier score artifacts must share one score coordinate")

    @property
    def client_id(self) -> ClientId:
        return self.benign_frontier.coordinate.client_id


def _score_vector(artifact: ScorePartitionArtifact) -> ScoreVector:
    return ScoreVector(
        np.fromiter(
            (record.score.value for record in artifact.records),
            dtype=np.float64,
            count=len(artifact.records),
        )
    )


def _attack_subtype_scores(
    artifact: ScorePartitionArtifact,
) -> tuple[AttackSubtypeScores, ...]:
    subtypes = tuple(
        sorted(
            {
                record.attack_subtype
                for record in artifact.records
                if record.attack_subtype is not None
            },
            key=lambda subtype: subtype.value,
        )
    )
    result: list[AttackSubtypeScores] = []
    for subtype in subtypes:
        scores = np.fromiter(
            (
                record.score.value
                for record in artifact.records
                if record.attack_subtype == subtype
            ),
            dtype=np.float64,
        )
        result.append(AttackSubtypeScores(subtype=subtype, scores=ScoreVector(scores)))
    return tuple(result)


def build_client_frontier_inputs(
    artifacts: FrontierScoreArtifacts,
    alpha_grid: AlphaGrid,
) -> ClientFrontierInputs:
    benign_scores = _score_vector(artifacts.benign_frontier)
    attack_scores = _attack_subtype_scores(artifacts.attack_validation)
    if not attack_scores:
        raise ValueError("attack-validation artifact contains no attack subtype scores")

    candidates: list[CandidateConfusions] = []
    for target_rate in alpha_grid.values:
        threshold = calibrate_threshold(benign_scores, target_rate)
        subtype_confusions: list[SubtypeConfusion] = []
        for subtype in attack_scores:
            alerts = alerts_above_threshold(subtype.scores, threshold)
            true_positive_count = int(np.count_nonzero(alerts.values))
            subtype_confusions.append(
                SubtypeConfusion(
                    subtype=subtype.subtype,
                    counts=SubtypeConfusionCounts(
                        true_positive=RowCount(true_positive_count),
                        false_negative=RowCount(
                            subtype.scores.row_count.value - true_positive_count
                        ),
                    ),
                )
            )
        candidates.append(
            CandidateConfusions(
                target_rate=target_rate,
                subtypes=tuple(subtype_confusions),
            )
        )

    return ClientFrontierInputs(
        client_id=artifacts.client_id,
        benign_frontier_scores=benign_scores,
        candidates=tuple(candidates),
    )
