from __future__ import annotations

import subprocess

from fabrid.artifacts.dataset_store import StoredDatasetManifests
from fabrid.artifacts.json_store import StoredJsonArtifact
from fabrid.domain.identifiers import GitCommit
from fabrid.domain.provenance import (
    DetectorProvenance,
    ExperimentProvenance,
    ScoreProvenance,
)
from fabrid.evaluation.evaluator import (
    ClientExperimentProvenance,
    EvaluationProvenance,
)
from fabrid.pipeline.scoring import StoredFederationScores
from fabrid.pipeline.training import TrainedDetectorSeed


def resolve_git_commit() -> GitCommit:
    completed = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    )
    return GitCommit(completed.stdout.strip())


def build_evaluation_provenance(
    trained: TrainedDetectorSeed,
    scores: StoredFederationScores,
    dataset_manifests: StoredDatasetManifests,
    protocol_snapshot: StoredJsonArtifact,
    git_commit: GitCommit,
) -> EvaluationProvenance:
    clients: list[ClientExperimentProvenance] = []
    for prepared_client in trained.prepared_federation.clients:
        client_id = prepared_client.dataset.client_id
        client_scores = scores.for_client(client_id)
        clients.append(
            ClientExperimentProvenance(
                client_id=client_id,
                provenance=ExperimentProvenance(
                    detector=DetectorProvenance(
                        model=trained.artifacts.model,
                        preprocessing=trained.artifacts.scaler_digest(client_id),
                        feature_manifest=dataset_manifests.features.digest,
                    ),
                    scores=ScoreProvenance(
                        benign_frontier=client_scores.benign_frontier.logical_digest,
                        benign_final_cal=client_scores.benign_final_cal.logical_digest,
                        benign_test=client_scores.benign_test.logical_digest,
                        attack_validation=client_scores.attack_validation.logical_digest,
                        attack_test=client_scores.attack_test.logical_digest,
                        split_manifest=dataset_manifests.splits.digest,
                        protocol=protocol_snapshot.digest,
                    ),
                    git_commit=git_commit,
                ),
            )
        )
    return EvaluationProvenance(tuple(clients))
