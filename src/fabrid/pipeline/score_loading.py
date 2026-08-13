from __future__ import annotations

from fabrid.artifacts.score_store import read_score_partition
from fabrid.detector.scoring import ClientScoreArtifacts
from fabrid.domain.coordinates import ScoreCoordinate
from fabrid.domain.enums import AttackSplit, BenignSplit
from fabrid.domain.identifiers import CampaignId
from fabrid.pipeline.context import PipelinePaths


def load_client_scores(
    campaign_id: CampaignId,
    coordinate: ScoreCoordinate,
    paths: PipelinePaths,
) -> ClientScoreArtifacts:
    layout = paths.artifacts
    return ClientScoreArtifacts(
        benign_train=read_score_partition(
            layout.score_path(campaign_id, coordinate, BenignSplit.TRAIN),
            coordinate,
            BenignSplit.TRAIN,
        ),
        benign_frontier=read_score_partition(
            layout.score_path(campaign_id, coordinate, BenignSplit.FRONTIER),
            coordinate,
            BenignSplit.FRONTIER,
        ),
        benign_final_cal=read_score_partition(
            layout.score_path(campaign_id, coordinate, BenignSplit.FINAL_CAL),
            coordinate,
            BenignSplit.FINAL_CAL,
        ),
        benign_test=read_score_partition(
            layout.score_path(campaign_id, coordinate, BenignSplit.TEST),
            coordinate,
            BenignSplit.TEST,
        ),
        attack_validation=read_score_partition(
            layout.score_path(campaign_id, coordinate, AttackSplit.VALIDATION),
            coordinate,
            AttackSplit.VALIDATION,
        ),
        attack_test=read_score_partition(
            layout.score_path(campaign_id, coordinate, AttackSplit.TEST),
            coordinate,
            AttackSplit.TEST,
        ),
    )
