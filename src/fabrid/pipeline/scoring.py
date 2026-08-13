from __future__ import annotations

from dataclasses import dataclass

from fabrid.artifacts.score_store import StoredScoreArtifact, write_score_partition
from fabrid.detector.scoring import ClientScoreArtifacts, generate_client_score_artifacts
from fabrid.domain.coordinates import ScoreCoordinate
from fabrid.domain.identifiers import CampaignId, ClientId
from fabrid.pipeline.context import PipelinePaths
from fabrid.pipeline.training import TrainedDetectorSeed


@dataclass(frozen=True, slots=True)
class StoredClientScores:
    client_id: ClientId
    benign_train: StoredScoreArtifact
    benign_frontier: StoredScoreArtifact
    benign_final_cal: StoredScoreArtifact
    benign_test: StoredScoreArtifact
    attack_validation: StoredScoreArtifact
    attack_test: StoredScoreArtifact

    @property
    def all(self) -> tuple[StoredScoreArtifact, ...]:
        return (
            self.benign_train,
            self.benign_frontier,
            self.benign_final_cal,
            self.benign_test,
            self.attack_validation,
            self.attack_test,
        )


@dataclass(frozen=True, slots=True)
class StoredFederationScores:
    clients: tuple[StoredClientScores, ...]

    def __post_init__(self) -> None:
        client_ids = tuple(client.client_id for client in self.clients)
        if len(set(client_ids)) != len(client_ids):
            raise ValueError("stored federation scores contain duplicate clients")

    def for_client(self, client_id: ClientId) -> StoredClientScores:
        for client in self.clients:
            if client.client_id == client_id:
                return client
        raise KeyError(client_id.value)


def _persist_client_scores(
    campaign_id: CampaignId,
    scores: ClientScoreArtifacts,
    paths: PipelinePaths,
) -> StoredClientScores:
    coordinate = scores.benign_train.coordinate
    layout = paths.artifacts
    return StoredClientScores(
        client_id=coordinate.client_id,
        benign_train=write_score_partition(
            scores.benign_train,
            layout.score_path(campaign_id, coordinate, scores.benign_train.split),
        ),
        benign_frontier=write_score_partition(
            scores.benign_frontier,
            layout.score_path(campaign_id, coordinate, scores.benign_frontier.split),
        ),
        benign_final_cal=write_score_partition(
            scores.benign_final_cal,
            layout.score_path(campaign_id, coordinate, scores.benign_final_cal.split),
        ),
        benign_test=write_score_partition(
            scores.benign_test,
            layout.score_path(campaign_id, coordinate, scores.benign_test.split),
        ),
        attack_validation=write_score_partition(
            scores.attack_validation,
            layout.score_path(campaign_id, coordinate, scores.attack_validation.split),
        ),
        attack_test=write_score_partition(
            scores.attack_test,
            layout.score_path(campaign_id, coordinate, scores.attack_test.split),
        ),
    )


def materialize_detector_scores(
    campaign_id: CampaignId,
    trained: TrainedDetectorSeed,
    paths: PipelinePaths,
) -> StoredFederationScores:
    stored_clients: list[StoredClientScores] = []
    for prepared_client in trained.prepared_federation.clients:
        coordinate = ScoreCoordinate(
            dataset_id=trained.coordinate.dataset_id,
            detector_seed=trained.coordinate.detector_seed,
            client_id=prepared_client.dataset.client_id,
        )
        if coordinate.client_id != prepared_client.dataset.client_id:
            raise AssertionError("score coordinate client mismatch")
        scores = generate_client_score_artifacts(
            dataset_id=trained.coordinate.dataset_id,
            detector_seed=trained.coordinate.detector_seed,
            dataset=prepared_client.dataset,
            split_plan=prepared_client.split_plan,
            scaler=prepared_client.scaler,
            model=trained.model,
        )
        stored_clients.append(_persist_client_scores(campaign_id, scores, paths))
    return StoredFederationScores(tuple(stored_clients))
