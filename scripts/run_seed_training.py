"""Full-scale (non-subsampled) FedAvg training and score generation for one detector seed.

Usage: python scripts/run_seed_training.py <seed>

Persists one pickled `ScoreArtifact` per client to `results/scores/seed_<seed>/<client>.pkl`
plus a `manifest.json` recording each artifact's sha256. This is an internal storage format
for resuming/inspecting runs, not the final roadmap-mandated result-table format.
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

from fabrid.config.detector import load_detector_hyperparameters
from fabrid.config.protocol import load_protocol
from fabrid.data.nbaiot_reader import read_device_directory
from fabrid.data.partitioner import (
    AttackSplitBoundary,
    BenignSplitBoundaries,
    RowCount,
    compute_attack_split_boundary,
    compute_benign_split_boundaries,
)
from fabrid.data.preprocessing import FeatureScaler, fit_feature_scaler
from fabrid.detector.training import FederatedTrainingConfig, train_federated_autoencoder
from fabrid.evaluation.record_level import AttackSubtype, ClientId
from fabrid.schemas.score_artifact import DetectorSeed, ScoreArtifact
from fabrid.scoring.score_generation import generate_score_artifact

_RAW_DIR = Path(__file__).parents[1] / "data" / "raw" / "N-BaIoT"
_RESULTS_DIR = Path(__file__).parents[1] / "results" / "scores"

_CLIENT_NAMES = [
    "Danmini_Doorbell",
    "Ennio_Doorbell",
    "Ecobee_Thermostat",
    "Philips_B120N10_Baby_Monitor",
    "Provision_PT_737E_Security_Camera",
    "Provision_PT_838_Security_Camera",
    "SimpleHome_XCS7_1002_WHT_Security_Camera",
    "SimpleHome_XCS7_1003_WHT_Security_Camera",
    "Samsung_SNH_1011_N_Webcam",
]


def run_seed(seed: int) -> dict[str, str]:
    t0 = time.time()
    protocol = load_protocol()
    hyperparameters = load_detector_hyperparameters()

    boundaries_by_client: dict[ClientId, BenignSplitBoundaries] = {}
    attack_boundaries_by_client: dict[ClientId, dict[AttackSubtype, AttackSplitBoundary]] = {}
    train_features_by_client: dict[ClientId, np.ndarray] = {}
    scaler_by_client: dict[ClientId, FeatureScaler] = {}
    raw_by_client = {}

    for name in _CLIENT_NAMES:
        client_id = ClientId(name)
        raw = read_device_directory(client_id, _RAW_DIR / name)
        raw_by_client[client_id] = raw
        print(f"[{time.time() - t0:6.1f}s] read {name}: benign={raw.benign_features.shape[0]}")

        boundaries = compute_benign_split_boundaries(
            RowCount(raw.benign_features.shape[0]), protocol.benign_split_fractions
        )
        boundaries_by_client[client_id] = boundaries
        train_features = raw.benign_features[: boundaries.train_end]
        train_features_by_client[client_id] = train_features
        scaler_by_client[client_id] = fit_feature_scaler(train_features)

        attack_boundaries = {
            subtype: compute_attack_split_boundary(
                RowCount(features.shape[0]), protocol.attack_split_fraction
            )
            for subtype, features in raw.attack_features_by_subtype.items()
        }
        attack_boundaries_by_client[client_id] = attack_boundaries

    scaled_train = {
        client_id: scaler_by_client[client_id].transform(features)
        for client_id, features in train_features_by_client.items()
    }
    training_config = FederatedTrainingConfig(
        hidden_dims=hyperparameters.hidden_dims,
        learning_rate=hyperparameters.learning_rate,
        local_epochs=hyperparameters.local_epochs,
        rounds=hyperparameters.rounds,
        batch_size=hyperparameters.batch_size,
        seed=seed,
    )
    print(f"[{time.time() - t0:6.1f}s] starting FedAvg training, seed={seed}")
    model = train_federated_autoencoder(scaled_train, training_config)
    print(f"[{time.time() - t0:6.1f}s] training complete")

    output_dir = _RESULTS_DIR / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, str] = {}

    for client_id, raw in raw_by_client.items():
        artifact: ScoreArtifact = generate_score_artifact(
            client_id,
            "n-baiot",
            DetectorSeed(seed),
            raw,
            boundaries_by_client[client_id],
            attack_boundaries_by_client[client_id],
            scaler_by_client[client_id],
            model,
        )
        artifact_hash = artifact.sha256()
        manifest[str(client_id)] = artifact_hash
        with (output_dir / f"{client_id}.pkl").open("wb") as handle:
            pickle.dump(artifact, handle)
        print(
            f"[{time.time() - t0:6.1f}s] {client_id}: {len(artifact.records)} records, "
            f"hash={artifact_hash[:12]}"
        )

    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"[{time.time() - t0:6.1f}s] seed {seed} complete")
    return manifest


if __name__ == "__main__":
    seed_arg = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    run_seed(seed_arg)
