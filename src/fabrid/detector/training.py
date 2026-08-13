from __future__ import annotations

from dataclasses import dataclass

import torch

from fabrid.datasets.common import FeatureMatrix
from fabrid.detector.local_training import train_local_autoencoder
from fabrid.detector.model import Autoencoder, AutoencoderArchitecture
from fabrid.detector.state import (
    TensorState,
    WeightedTensorState,
    average_weighted_tensor_states,
)
from fabrid.domain.identifiers import ClientId
from fabrid.domain.values import (
    BatchSize,
    DetectorSeed,
    FederatedRoundCount,
    LayerWidth,
    LearningRate,
    LocalEpochCount,
)


@dataclass(frozen=True, slots=True)
class ClientTrainingData:
    client_id: ClientId
    features: FeatureMatrix

    def __post_init__(self) -> None:
        if self.features.row_count.value == 0:
            raise ValueError("client training data must contain at least one row")


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
class FederatedTrainingConfig:
    hidden_layers: tuple[LayerWidth, ...]
    learning_rate: LearningRate
    local_epochs: LocalEpochCount
    rounds: FederatedRoundCount
    batch_size: BatchSize
    seed: DetectorSeed

    def __post_init__(self) -> None:
        if not self.hidden_layers:
            raise ValueError("federated training requires at least one hidden layer")


def train_federated_autoencoder(
    training_data: FederatedTrainingData,
    config: FederatedTrainingConfig,
) -> Autoencoder:
    torch.manual_seed(config.seed.value)  # pyright: ignore[reportUnknownMemberType]
    torch.cuda.manual_seed_all(config.seed.value)
    architecture = AutoencoderArchitecture(
        feature_count=training_data.clients[0].features.feature_count,
        hidden_layers=config.hidden_layers,
    )
    global_model = Autoencoder(architecture)
    for _ in range(config.rounds.value):
        global_state = TensorState.from_module(global_model)
        local_states: list[WeightedTensorState] = []
        for client in training_data.clients:
            local_model = Autoencoder(architecture)
            global_state.load_into(local_model)
            train_local_autoencoder(
                local_model,
                client.features,
                config.learning_rate,
                config.local_epochs,
                config.batch_size,
            )
            local_states.append(
                WeightedTensorState(
                    state=TensorState.from_module(local_model),
                    weight=client.features.row_count,
                )
            )
        average_weighted_tensor_states(tuple(local_states)).load_into(global_model)
    return global_model
