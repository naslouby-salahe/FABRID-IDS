from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict
from safetensors.torch import load_file, save_file

from fabrid.artifacts.digests import digest_file
from fabrid.detector.model import Autoencoder, AutoencoderArchitecture
from fabrid.detector.preprocessing import ClientScaler, FeatureScaler, FederatedScalers
from fabrid.domain.identifiers import ArtifactDigest, ClientId
from fabrid.domain.values import FeatureCount, LayerWidth

_MODEL_FILENAME = "model.safetensors"
_ARCHITECTURE_FILENAME = "architecture.json"
_SCALER_DIRECTORY = "scalers"
_SCALER_SUFFIX = ".safetensors"
_SCALER_MEAN_KEY = "mean"
_SCALER_STANDARD_DEVIATION_KEY = "standard_deviation"


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


def _architecture_payload(architecture: AutoencoderArchitecture) -> _ArchitecturePayload:
    return _ArchitecturePayload(
        feature_count=architecture.feature_count.value,
        hidden_layers=tuple(layer.value for layer in architecture.hidden_layers),
    )


def _state_dict_for_storage(model: Autoencoder) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in model.state_dict().items()
    }


def _scaler_tensors(scaler: FeatureScaler) -> dict[str, torch.Tensor]:
    return {
        _SCALER_MEAN_KEY: torch.from_numpy(np.ascontiguousarray(scaler.mean)),
        _SCALER_STANDARD_DEVIATION_KEY: torch.from_numpy(
            np.ascontiguousarray(scaler.standard_deviation)
        ),
    }


def save_detector_state(
    output_dir: Path,
    model: Autoencoder,
    scalers: FederatedScalers,
) -> DetectorArtifactSet:
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / _MODEL_FILENAME
    save_file(_state_dict_for_storage(model), model_path)

    architecture_path = output_dir / _ARCHITECTURE_FILENAME
    architecture_path.write_text(
        _architecture_payload(model.architecture).model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

    scaler_directory = output_dir / _SCALER_DIRECTORY
    scaler_directory.mkdir(parents=True, exist_ok=True)
    scaler_artifacts: list[ClientScalerArtifact] = []
    for client in scalers.clients:
        scaler_path = scaler_directory / f"{client.client_id.value}{_SCALER_SUFFIX}"
        save_file(_scaler_tensors(client.scaler), scaler_path)
        scaler_artifacts.append(
            ClientScalerArtifact(
                client_id=client.client_id,
                digest=digest_file(scaler_path),
            )
        )

    return DetectorArtifactSet(
        model=digest_file(model_path),
        architecture=digest_file(architecture_path),
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
    model.load_state_dict(load_file(output_dir / _MODEL_FILENAME))

    scaler_directory = output_dir / _SCALER_DIRECTORY
    client_scalers: list[ClientScaler] = []
    for scaler_path in sorted(scaler_directory.glob(f"*{_SCALER_SUFFIX}")):
        tensors = load_file(scaler_path)
        client_scalers.append(
            ClientScaler(
                client_id=ClientId(scaler_path.name.removesuffix(_SCALER_SUFFIX)),
                scaler=FeatureScaler(
                    mean=tensors[_SCALER_MEAN_KEY].numpy().copy(),
                    standard_deviation=tensors[_SCALER_STANDARD_DEVIATION_KEY]
                    .numpy()
                    .copy(),
                ),
            )
        )

    return PersistedDetector(
        model=model,
        scalers=FederatedScalers(tuple(client_scalers)),
    )
