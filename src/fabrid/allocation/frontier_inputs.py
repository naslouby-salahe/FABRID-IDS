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
from fabrid.domain.enums import AttackSplit, BenignSplit
from fabrid.domain.identifiers import AttackSubtypeId, ClientId
from fabrid.domain.scores import ScoreVector
from fabrid.domain.values import RowCount
from fabrid.protocol.models import AlphaGrid


@dataclass(frozen=True, slots=True)
class AttackSubtypeScores:
    subtype: AttackSubtypeId
    scores: ScoreVector


@dataclass(frozen=True, slots=True)
class FrontierScorePopulation:
    client_id: ClientId
    benign_frontier: ScoreVector
    attack_validation: tuple[AttackSubtypeScores, ...]

    def __post_init__(self) -> None:
        if self.benign_frontier.row_count.value == 0:
            raise ValueError("frontier score population requires benign frontier scores")
        if not self.attack_validation:
            raise ValueError("frontier score population requires attack-validation scores")
        subtype_ids = tuple(entry.subtype for entry in self.attack_validation)
        if len(set(subtype_ids)) != len(subtype_ids):
            raise ValueError("frontier score population contains duplicate attack subtypes")
        if any(entry.scores.row_count.value == 0 for entry in self.attack_validation):
            raise ValueError("frontier score population contains an empty attack subtype")


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
    return tuple(
        AttackSubtypeScores(
            subtype=subtype,
            scores=ScoreVector(
                np.fromiter(
                    (
                        record.score.value
                        for record in artifact.records
                        if record.attack_subtype == subtype
                    ),
                    dtype=np.float64,
                )
            ),
        )
        for subtype in subtypes
    )


def frontier_score_population(
    artifacts: FrontierScoreArtifacts,
) -> FrontierScorePopulation:
    return FrontierScorePopulation(
        client_id=artifacts.client_id,
        benign_frontier=_score_vector(artifacts.benign_frontier),
        attack_validation=_attack_subtype_scores(artifacts.attack_validation),
    )


def build_client_frontier_inputs(
    source: FrontierScorePopulation | FrontierScoreArtifacts,
    alpha_grid: AlphaGrid,
) -> ClientFrontierInputs:
    population = (
        frontier_score_population(source)
        if isinstance(source, FrontierScoreArtifacts)
        else source
    )
    candidates: list[CandidateConfusions] = []
    for target_rate in alpha_grid.values:
        threshold = calibrate_threshold(population.benign_frontier, target_rate)
        subtype_confusions: list[SubtypeConfusion] = []
        for subtype in population.attack_validation:
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
        client_id=population.client_id,
        benign_frontier_scores=population.benign_frontier,
        candidates=tuple(candidates),
    )
