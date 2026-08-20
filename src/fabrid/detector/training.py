from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, TypeAdapter
from torch import nn

from fabrid.artifacts.checkpoints import (
    TensorName,
    load_module_state,
    load_scaler,
    save_module_state,
    save_scaler,
)
from fabrid.artifacts.json import digest_file, digest_text, read_typed_json, write_typed_json
from fabrid.artifacts.paths import ArtifactPaths, DetectorCoordinate
from fabrid.config import (
    TIGHT_TOLERANCE,
    ArtifactDigest,
    BatchSize,
    ClientId,
    DatasetId,
    DetectorSeed,
    FeatureCount,
    FederatedRoundCount,
    LayerWidth,
    LearningRate,
    LocalEpochCount,
    RowCount,
)
from fabrid.datasets.registry import FeatureMatrix
from fabrid.detector.autoencoder import Autoencoder, AutoencoderArchitecture

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ClientTrainingData:
    client_id: ClientId
    features: FeatureMatrix

    def __post_init__(self) -> None:
        if self.features.row_count == 0:
            raise ValueError("client training data must contain at least one row")

    @property
    def weight(self) -> RowCount:
        return self.features.row_count


@dataclass(frozen=True, slots=True)
class FederatedTrainingData:
    clients: tuple[ClientTrainingData, ...]

    def __post_init__(self) -> None:
        if not self.clients:
            raise ValueError("federated training requires at least one client")
        client_ids = tuple(client.client_id for client in self.clients)
        if len(set(client_ids)) != len(client_ids):
            raise ValueError("federated training data contains duplicate clients")
        feature_count = self.clients[0].features.feature_count
        if any(client.features.feature_count != feature_count for client in self.clients):
            raise ValueError("all clients must share the same feature width")


@dataclass(frozen=True, slots=True)
class FederatedTrainingSettings:
    learning_rate: LearningRate
    local_epochs: LocalEpochCount
    rounds: FederatedRoundCount
    batch_size: BatchSize


@dataclass(frozen=True, slots=True)
class _TensorParameter:
    name: TensorName
    tensor: torch.Tensor


@dataclass(frozen=True, slots=True)
class _TensorState:
    parameters: tuple[_TensorParameter, ...]

    def __post_init__(self) -> None:
        if not self.parameters:
            raise ValueError("tensor state must contain at least one parameter")
        names = tuple(parameter.name for parameter in self.parameters)
        if len(set(names)) != len(names):
            raise ValueError("tensor state contains duplicate parameter names")

    @classmethod
    def from_module(cls, module: nn.Module) -> _TensorState:
        return cls(
            tuple(
                _TensorParameter(name=name, tensor=tensor.detach().clone())
                for name, tensor in module.state_dict().items()
            )
        )

    def tensor(self, name: TensorName) -> torch.Tensor:
        for parameter in self.parameters:
            if parameter.name == name:
                return parameter.tensor
        raise KeyError(name)

    def load_into(self, module: nn.Module) -> None:
        module.load_state_dict({parameter.name: parameter.tensor for parameter in self.parameters})


@dataclass(frozen=True, slots=True)
class _WeightedTensorState:
    state: _TensorState
    weight: RowCount

    def __post_init__(self) -> None:
        if self.weight == 0:
            raise ValueError("weighted tensor state requires a positive row count")


def _average_weighted_tensor_states(states: tuple[_WeightedTensorState, ...]) -> _TensorState:
    if not states:
        raise ValueError("weighted tensor averaging requires at least one state")
    reference_names = tuple(parameter.name for parameter in states[0].state.parameters)
    for weighted in states[1:]:
        names = tuple(parameter.name for parameter in weighted.state.parameters)
        if names != reference_names:
            raise ValueError("all tensor states must contain the same ordered parameters")
    total_weight: RowCount = sum(weighted.weight for weighted in states)
    parameters: list[_TensorParameter] = []
    for name in reference_names:
        stacked = torch.stack(
            tuple(weighted.state.tensor(name).float() * weighted.weight for weighted in states)
        )
        parameters.append(_TensorParameter(name=name, tensor=stacked.sum(dim=0) / total_weight))
    return _TensorState(tuple(parameters))


def _train_local_autoencoder(
    model: Autoencoder,
    features: FeatureMatrix,
    settings: FederatedTrainingSettings,
    device: torch.device,
) -> None:
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=0.0,
    )
    loss_function = nn.MSELoss()
    inputs = torch.as_tensor(features.values, dtype=torch.float32, device=device)
    row_count: RowCount = inputs.shape[0]
    for _ in range(settings.local_epochs):
        permutation = torch.randperm(row_count, device=device)
        for start in range(0, row_count, settings.batch_size):
            batch = inputs[permutation[start : start + settings.batch_size]]
            optimizer.zero_grad()
            loss = loss_function(model(batch), batch)
            loss.backward()
            optimizer.step()
    model.to("cpu")


def train_federated_autoencoder(
    training_data: FederatedTrainingData,
    architecture: AutoencoderArchitecture,
    settings: FederatedTrainingSettings,
    seed: DetectorSeed,
    device: torch.device,
) -> Autoencoder:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    global_model = Autoencoder(architecture)
    for round_index in range(1, settings.rounds + 1):
        logger.info(
            "[PROGRESS] federated training seed=%d round %d/%d",
            seed,
            round_index,
            settings.rounds,
        )
        global_state = _TensorState.from_module(global_model)
        local_states: list[_WeightedTensorState] = []
        for client in training_data.clients:
            local_model = Autoencoder(architecture)
            global_state.load_into(local_model)
            _train_local_autoencoder(local_model, client.features, settings, device)
            local_states.append(
                _WeightedTensorState(
                    state=_TensorState.from_module(local_model),
                    weight=client.weight,
                )
            )
        _average_weighted_tensor_states(tuple(local_states)).load_into(global_model)
    return global_model


@dataclass(frozen=True, slots=True)
class FeatureScaler:
    mean: np.ndarray
    standard_deviation: np.ndarray

    def __post_init__(self) -> None:
        if self.mean.shape != self.standard_deviation.shape:
            raise ValueError("scaler mean and standard deviation must share shape")
        if self.mean.ndim != 1:
            raise ValueError("scaler statistics must be one-dimensional")
        if np.any(self.standard_deviation <= 0):
            raise ValueError("scaler standard deviation must be strictly positive")

    def transform(self, features: FeatureMatrix) -> FeatureMatrix:
        if features.feature_count != self.mean.shape[0]:
            raise ValueError("feature matrix width does not match scaler statistics")
        return FeatureMatrix((features.values - self.mean) / self.standard_deviation)


@dataclass(frozen=True, slots=True)
class ClientScaler:
    client_id: ClientId
    scaler: FeatureScaler


@dataclass(frozen=True, slots=True)
class FederatedScalers:
    clients: tuple[ClientScaler, ...]

    def __post_init__(self) -> None:
        if not self.clients:
            raise ValueError("federated scalers require at least one client")
        client_ids = tuple(client.client_id for client in self.clients)
        if len(set(client_ids)) != len(client_ids):
            raise ValueError("federated scalers contain duplicate clients")

    def for_client(self, client_id: ClientId) -> FeatureScaler:
        for client in self.clients:
            if client.client_id == client_id:
                return client.scaler
        raise KeyError(client_id)


def fit_feature_scaler(train_features: FeatureMatrix) -> FeatureScaler:
    if train_features.row_count == 0:
        raise ValueError("feature scaler requires at least one training row")
    mean = train_features.values.mean(axis=0)
    standard_deviation = train_features.values.std(axis=0)
    safe_standard_deviation = np.where(
        standard_deviation < TIGHT_TOLERANCE,
        1.0,
        standard_deviation,
    )
    return FeatureScaler(mean=mean, standard_deviation=safe_standard_deviation)


class ClientScalerDigest(BaseModel):
    model_config = ConfigDict(frozen=True)
    client_id: ClientId
    digest: ArtifactDigest


class DetectorInputSpace(StrEnum):
    PER_CLIENT_ZSCORE = "per_client_zscore"


class CheckpointMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, revalidate_instances="never")
    dataset_id: DatasetId
    detector_seed: DetectorSeed
    feature_count: FeatureCount
    hidden_layers: tuple[LayerWidth, ...]
    input_space: DetectorInputSpace
    training: FederatedTrainingSettings
    preprocessing_digest: ArtifactDigest
    model_digest: ArtifactDigest
    scaler_digests: tuple[ClientScalerDigest, ...]

    @property
    def combined_digest(self) -> ArtifactDigest:
        return digest_text(
            (
                self.dataset_id.value,
                str(self.detector_seed),
                str(self.feature_count),
                ",".join(str(width) for width in self.hidden_layers),
                self.input_space.value,
                str(self.training.learning_rate),
                str(self.training.local_epochs),
                str(self.training.rounds),
                str(self.training.batch_size),
                self.preprocessing_digest,
                self.model_digest,
                "|".join(f"{digest.client_id}:{digest.digest}" for digest in self.scaler_digests),
            )
        )


_CHECKPOINT_METADATA_ADAPTER: TypeAdapter[CheckpointMetadata] = TypeAdapter(CheckpointMetadata)


def save_detector_checkpoint(
    paths: ArtifactPaths,
    coordinate: DetectorCoordinate,
    model: Autoencoder,
    scalers: FederatedScalers,
    training: FederatedTrainingSettings,
    preprocessing_digest: ArtifactDigest,
) -> CheckpointMetadata:
    save_module_state(model, paths.checkpoint_model_path(coordinate))
    for client in scalers.clients:
        save_scaler(
            client.scaler.mean,
            client.scaler.standard_deviation,
            paths.checkpoint_client_scaler_path(coordinate, client.client_id),
        )
    metadata = CheckpointMetadata(
        dataset_id=coordinate.dataset_id,
        detector_seed=coordinate.detector_seed,
        feature_count=model.architecture.feature_count,
        hidden_layers=model.architecture.hidden_layers,
        input_space=DetectorInputSpace.PER_CLIENT_ZSCORE,
        training=training,
        preprocessing_digest=preprocessing_digest,
        model_digest=digest_file(paths.checkpoint_model_path(coordinate)),
        scaler_digests=tuple(
            ClientScalerDigest(
                client_id=client.client_id,
                digest=digest_file(
                    paths.checkpoint_client_scaler_path(coordinate, client.client_id)
                ),
            )
            for client in scalers.clients
        ),
    )
    write_typed_json(
        metadata,
        _CHECKPOINT_METADATA_ADAPTER,
        paths.checkpoint_metadata_path(coordinate),
    )
    return metadata


def load_detector_checkpoint(
    paths: ArtifactPaths,
    coordinate: DetectorCoordinate,
) -> tuple[Autoencoder, FederatedScalers, CheckpointMetadata]:
    metadata = read_typed_json(
        paths.checkpoint_metadata_path(coordinate),
        _CHECKPOINT_METADATA_ADAPTER,
    )
    architecture = AutoencoderArchitecture(
        feature_count=metadata.feature_count,
        hidden_layers=metadata.hidden_layers,
    )
    model = Autoencoder(architecture)
    load_module_state(model, paths.checkpoint_model_path(coordinate))
    client_scalers: list[ClientScaler] = []
    for scaler_digest in metadata.scaler_digests:
        mean, standard_deviation = load_scaler(
            paths.checkpoint_client_scaler_path(coordinate, scaler_digest.client_id)
        )
        client_scalers.append(
            ClientScaler(
                client_id=scaler_digest.client_id,
                scaler=FeatureScaler(mean=mean, standard_deviation=standard_deviation),
            )
        )
    return model, FederatedScalers(tuple(client_scalers)), metadata
