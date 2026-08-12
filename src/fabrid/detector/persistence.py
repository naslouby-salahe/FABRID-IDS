"""Detector model/scaler weight persistence (TRAIN-002): the frozen global model and every
client's TRAIN-only feature scaler for one seed, hashed for provenance, separate from the
`ScoreArtifact`s that record only the model's *output* on that seed.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from fabrid.data.preprocessing import FeatureScaler
from fabrid.detector.model import Autoencoder, AutoencoderArchitecture
from fabrid.evaluation.record_level import ClientId

_MODEL_FILENAME = "model_state_dict.pt"
_ARCHITECTURE_FILENAME = "architecture.json"
_SCALERS_FILENAME = "scalers.npz"


@dataclass(frozen=True, slots=True)
class DetectorStateHashes:
    model_sha256: str
    scalers_sha256: str


def _sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_detector_state(
    output_dir: Path,
    model: Autoencoder,
    architecture: AutoencoderArchitecture,
    scaler_by_client: Mapping[ClientId, FeatureScaler],
) -> DetectorStateHashes:
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / _MODEL_FILENAME
    torch.save(model.state_dict(), model_path)

    architecture_path = output_dir / _ARCHITECTURE_FILENAME
    architecture_path.write_text(
        f'{{"n_features": {architecture.n_features}, '
        f'"hidden_dims": {list(architecture.hidden_dims)}}}',
        encoding="utf-8",
    )

    scalers_path = output_dir / _SCALERS_FILENAME
    scaler_arrays: dict[str, np.ndarray] = {}
    for client_id, scaler in scaler_by_client.items():
        scaler_arrays[f"{client_id}__mean"] = scaler.mean
        scaler_arrays[f"{client_id}__std"] = scaler.std
    # numpy-stubs' savez overloads don't resolve cleanly against **dict unpacking with dynamic
    # (non-literal) keyword names; this is a documented stub gap, not an application error.
    np.savez(scalers_path, **scaler_arrays)  # pyright: ignore[reportArgumentType]

    return DetectorStateHashes(
        model_sha256=_sha256_of_file(model_path),
        scalers_sha256=_sha256_of_file(scalers_path),
    )


def load_detector_state(
    output_dir: Path, architecture: AutoencoderArchitecture
) -> tuple[Autoencoder, dict[ClientId, FeatureScaler]]:
    model = Autoencoder(architecture)
    state_dict = torch.load(output_dir / _MODEL_FILENAME, weights_only=True)
    model.load_state_dict(state_dict)

    scaler_by_client: dict[ClientId, FeatureScaler] = {}
    with np.load(output_dir / _SCALERS_FILENAME) as archive:
        client_ids = {key.rsplit("__", 1)[0] for key in archive}
        for client_id_str in client_ids:
            scaler_by_client[ClientId(client_id_str)] = FeatureScaler(
                mean=archive[f"{client_id_str}__mean"],
                std=archive[f"{client_id_str}__std"],
            )

    return model, scaler_by_client
