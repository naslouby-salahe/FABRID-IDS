"""End-to-end smoke check: real N-BaIoT data through the full FABRID-IDS pipeline.

Subsamples heavily so it runs in well under a minute. This validates that
ingestion -> partitioning -> preprocessing -> FedAvg training -> scoring ->
score artifact all compose correctly against real data; it is not a
confirmatory experiment run (see docs/tmp/fabrid-implementation/state.md for
the real 10-seed protocol run this smoke check precedes).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from fabrid.config.protocol import load_protocol
from fabrid.data.nbaiot_reader import read_device_directory
from fabrid.data.partitioner import (
    RowCount,
    compute_attack_split_boundary,
    compute_benign_split_boundaries,
)
from fabrid.data.preprocessing import fit_feature_scaler
from fabrid.detector.training import FederatedTrainingConfig, train_federated_autoencoder
from fabrid.evaluation.record_level import ClientId
from fabrid.schemas.score_artifact import DetectorSeed
from fabrid.scoring.score_contract import compute_auroc
from fabrid.scoring.score_generation import generate_score_artifact

_RAW_DIR = Path(__file__).parents[1] / "data" / "raw" / "N-BaIoT"
_SUBSAMPLE_ROWS = 400
_CLIENT_NAMES = [
    "Danmini_Doorbell",
    "Ennio_Doorbell",
    "Ecobee_Thermostat",
]


def main() -> None:
    t0 = time.time()
    protocol = load_protocol()

    raw_by_client = {}
    for name in _CLIENT_NAMES:
        raw = read_device_directory(ClientId(name), _RAW_DIR / name)
        raw_by_client[ClientId(name)] = raw
        subtypes = list(raw.attack_features_by_subtype)
        print(f"read {name}: benign={raw.benign_features.shape}, attacks={subtypes}")

    train_features = {}
    benign_boundaries_by_client = {}
    attack_boundaries_by_client = {}
    for client_id, raw in raw_by_client.items():
        subsampled_benign = raw.benign_features[:_SUBSAMPLE_ROWS]
        boundaries = compute_benign_split_boundaries(
            RowCount(subsampled_benign.shape[0]), protocol.benign_split_fractions
        )
        benign_boundaries_by_client[client_id] = (subsampled_benign, boundaries)
        train_features[client_id] = subsampled_benign[: boundaries.train_end]

        attack_boundaries: dict = {}
        subsampled_attacks = {}
        for subtype, features in raw.attack_features_by_subtype.items():
            subsampled = features[:_SUBSAMPLE_ROWS]
            subsampled_attacks[subtype] = subsampled
            attack_boundaries[subtype] = compute_attack_split_boundary(
                RowCount(subsampled.shape[0]), protocol.attack_split_fraction
            )
        attack_boundaries_by_client[client_id] = (subsampled_attacks, attack_boundaries)

    scalers = {client_id: fit_feature_scaler(feats) for client_id, feats in train_features.items()}

    config = FederatedTrainingConfig(
        hidden_dims=(32, 8), learning_rate=0.005, local_epochs=2, rounds=3, batch_size=32, seed=0
    )
    scaled_train = {
        client_id: scalers[client_id].transform(feats)
        for client_id, feats in train_features.items()
    }
    model = train_federated_autoencoder(scaled_train, config)
    print(f"trained FedAvg model across {len(scaled_train)} clients")

    artifacts = {}
    for client_id, raw in raw_by_client.items():
        benign_features, benign_boundaries = benign_boundaries_by_client[client_id]
        attack_features, attack_boundaries = attack_boundaries_by_client[client_id]
        raw_subsampled = type(raw)(
            client_id=client_id,
            benign_features=benign_features,
            attack_features_by_subtype=attack_features,
        )
        artifact = generate_score_artifact(
            client_id,
            "n-baiot",
            DetectorSeed(0),
            raw_subsampled,
            benign_boundaries,
            attack_boundaries,
            scalers[client_id],
            model,
        )
        artifacts[client_id] = artifact
        print(f"{client_id}: {len(artifact.records)} score records, hash={artifact.sha256()[:12]}")

    for client_id, artifact in artifacts.items():
        benign_test_scores = [
            r.score
            for r in artifact.records
            if r.label.value == "benign" and r.split_id.value == "benign_test"
        ]
        attack_test_scores = [
            r.score
            for r in artifact.records
            if r.label.value == "attack" and "test" in r.split_id.value
        ]
        if benign_test_scores and attack_test_scores:
            scores = np.array(benign_test_scores + attack_test_scores)
            is_attack = np.array(
                [False] * len(benign_test_scores) + [True] * len(attack_test_scores)
            )
            auroc = compute_auroc(scores, is_attack)
            print(
                f"{client_id}: smoke AUROC={auroc:.3f} "
                f"(n_benign={len(benign_test_scores)}, n_attack={len(attack_test_scores)})"
            )

    print(f"smoke pipeline completed in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
