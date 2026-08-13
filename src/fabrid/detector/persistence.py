from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict

from fabrid.detector.model import Autoencoder, AutoencoderArchitecture
from fabrid.detector.preprocessing import ClientScaler, FeatureScaler, FederatedScalers
from fabrid.domain.identifiers import ArtifactDigest, ClientId
from fabrid.domain.values import FeatureCount, LayerWidth

_MODEL_FILENAME = "model_state_dict.pt"
_ARCHITECTURE_FILENAME = "architecture.json"
_SCALER_DIRECTORY = "scalers"


class _ArchitecturePayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    feature_count: int
    hidden_layers: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ClientScalerArtifact:
    client_id: ClientId
    digest: ArtifactDigest


@dataclass(frozen=True, slots=True)
class DetectorArtifactSet:
    model: ArtifactDigest
    architecture: ArtifactDigest
    scalers: tuple[ClientScalerArtifact, ...]

    def scaler_digest(self, client_id: ClientId) -> ArtifactDigest:
        for scaler in self.scalers:
            if scaler.client_id == client_id:
                return scaler.digest
        raise KeyError(client_id.value)


@dataclass(frozen=True, slots=True)
class PersistedDetector:
    model: Autoencoder
    scalers: FederatedScalers


def _digest_file(path: Path) -> ArtifactDigest:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return ArtifactDigest(digest.hexdigest())


def _architecture_payload(architecture: AutoencoderArchitecture) -> _ArchitecturePayload:
    return _ArchitecturePayload(
        feature_count=architecture.feature_count.value,
        hidden_layers=tuple(layer.value for layer in architecture.hidden_layers),
    )


def save_detector_state(
    output_dir: Path,
    model: Autoencoder,
    scalers: FederatedScalers,
) -> DetectorArtifactSet:
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / _MODEL_FILENAME
    torch.save(model.state_dict(), model_path)

    architecture_path = output_dir / _ARCHITECTURE_FILENAME
    architecture_path.write_text(
        _architecture_payload(model.architecture).model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

    scaler_directory = output_dir / _SCALER_DIRECTORY
    scaler_directory.mkdir(parents=True, exist_ok=True)
    scaler_artifacts: list[ClientScalerArtifact] = []
    for client in scalers.clients:
        scaler_path = scaler_directory / f"{client.client_id.value}.npz"
        np.savez(
            scaler_path,
            mean=client.scaler.mean,
            standard_deviation=client.scaler.standard_deviation,
        )
        scaler_artifacts.append(
            ClientScalerArtifact(
                client_id=client.client_id,
                digest=_digest_file(scaler_path),
            )
        )

    return DetectorArtifactSet(
        model=_digest_file(model_path),
        architecture=_digest_file(architecture_path),
        scalers=tuple(scaler_artifacts),
    )


def load_detector_state(output_dir: Path) -> PersistedDetector:
    architecture_payload = _ArchitecturePayload.model_validate_json(
        (output_dir / _ARCHITECTURE_FILENAME).read_text(encoding="utf-8")
    )
    architecture = AutoencoderArchitecture(
        feature_count=FeatureCount(architecture_payload.feature_count),
        hidden_layers=tuple(
            LayerWidth(width) for width in architecture_payload.hidden_layers
        ),
    )
    model = Autoencoder(architecture)
    state_dict = torch.load(output_dir / _MODEL_FILENAME, weights_only=True)
    model.load_state_dict(state_dict)

    scaler_directory = output_dir / _SCALER_DIRECTORY
    client_scalers: list[ClientScaler] = []
    for scaler_path in sorted(scaler_directory.glob("*.npz")):
        with np.load(scaler_path) as archive:
            client_scalers.append(
                ClientScaler(
                    client_id=ClientId(scaler_path.stem),
                    scaler=FeatureScaler(
                        mean=archive["mean"],
                        standard_deviation=archive["standard_deviation"],
                    ),
                )
            )

    return PersistedDetector(
        model=model,
        scalers=FederatedScalers(tuple(client_scalers)),
    )
