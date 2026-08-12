"""Score generation: apply the frozen detector once to produce an immutable `ScoreArtifact`.

Ties together raw ingestion, partitioning, preprocessing, and the trained
model. Called exactly once per dataset x seed x client coordinate; every
policy downstream only reads the resulting artifact.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from fabrid.data.nbaiot_reader import RawDeviceData
from fabrid.data.partitioner import AttackSplitBoundary, BenignSplitBoundaries, RowCount, RowIndex
from fabrid.data.preprocessing import FeatureScaler
from fabrid.detector.model import Autoencoder, reconstruction_error_scores
from fabrid.evaluation.record_level import AttackSubtype, ClientId
from fabrid.schemas.score_artifact import DetectorSeed, Label, ScoreArtifact, ScoreRecord


def _benign_records(
    client_id: ClientId,
    scores: np.ndarray,
    boundaries: BenignSplitBoundaries,
    source_file: str,
) -> list[ScoreRecord]:
    return [
        ScoreRecord(
            sample_id=f"{client_id}:benign:{row_index}",
            source_file=source_file,
            source_row=RowCount(row_index),
            split_id=boundaries.split_of(RowIndex(row_index)),
            score=float(scores[row_index]),
            label=Label.BENIGN,
            attack_type=None,
            timestamp=None,
        )
        for row_index in range(boundaries.total_rows)
    ]


def _attack_records(
    client_id: ClientId,
    subtype: AttackSubtype,
    scores: np.ndarray,
    boundary: AttackSplitBoundary,
    source_file: str,
) -> list[ScoreRecord]:
    return [
        ScoreRecord(
            sample_id=f"{client_id}:{subtype}:{row_index}",
            source_file=source_file,
            source_row=RowCount(row_index),
            split_id=boundary.split_of(RowIndex(row_index)),
            score=float(scores[row_index]),
            label=Label.ATTACK,
            attack_type=subtype,
            timestamp=None,
        )
        for row_index in range(boundary.total_rows)
    ]


def generate_score_artifact(
    client_id: ClientId,
    dataset_id: str,
    detector_seed: DetectorSeed,
    raw_data: RawDeviceData,
    benign_boundaries: BenignSplitBoundaries,
    attack_boundaries: Mapping[AttackSubtype, AttackSplitBoundary],
    scaler: FeatureScaler,
    model: Autoencoder,
) -> ScoreArtifact:
    if raw_data.benign_features.shape[0] != benign_boundaries.total_rows:
        raise ValueError(
            f"client {client_id}: benign row count {raw_data.benign_features.shape[0]} does not "
            f"match boundaries.total_rows {benign_boundaries.total_rows}"
        )

    benign_scores = reconstruction_error_scores(model, scaler.transform(raw_data.benign_features))
    records = _benign_records(client_id, benign_scores, benign_boundaries, "benign_traffic.csv")

    for subtype, boundary in attack_boundaries.items():
        if subtype not in raw_data.attack_features_by_subtype:
            continue
        features = raw_data.attack_features_by_subtype[subtype]
        if features.shape[0] != boundary.total_rows:
            raise ValueError(
                f"client {client_id}, subtype {subtype}: row count {features.shape[0]} does not "
                f"match boundary.total_rows {boundary.total_rows}"
            )
        attack_scores = reconstruction_error_scores(model, scaler.transform(features))
        records.extend(_attack_records(client_id, subtype, attack_scores, boundary, str(subtype)))

    return ScoreArtifact(
        dataset_id=dataset_id,
        detector_seed=detector_seed,
        client_id=client_id,
        records=tuple(records),
    )
