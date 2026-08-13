from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from fabrid.datasets.common import FeatureMatrix
from fabrid.detector.model import Autoencoder, AutoencoderArchitecture, resolve_device
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


def _train_local_epochs(
    model: Autoencoder,
    features: FeatureMatrix,
    config: FederatedTrainingConfig,
) -> None:
    device = resolve_device()
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate.value,
    )
    loss_function = nn.MSELoss()
    inputs = torch.as_tensor(features.values, dtype=torch.float32, device=device)
    row_count = inputs.shape[0]

    for _ in range(config.local_epochs.value):
        permutation = torch.randperm(row_count, device=device)
        for start in range(0, row_count, config.batch_size.value):
            batch_indices = permutation[start : start + config.batch_size.value]
            batch = inputs[batch_indices]
            optimizer.zero_grad()
            reconstructed = model(batch)
            loss = loss_function(reconstructed, batch)
            loss.backward()
            optimizer.step()  # pyright: ignore[reportUnknownMemberType]

    model.to("cpu")


def _weighted_average_state_dicts(
    state_dicts: list[dict[str, torch.Tensor]],
    weights: list[float],
) -> dict[str, torch.Tensor]:
    total_weight = sum(weights)
    averaged: dict[str, torch.Tensor] = {}
    for key in state_dicts[0]:
        stacked = torch.stack(
            [
                state[key].float() * weight
                for state, weight in zip(state_dicts, weights, strict=True)
            ]
        )
        averaged[key] = stacked.sum(dim=0) / total_weight
    return averaged


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
        local_states: list[dict[str, torch.Tensor]] = []
        local_weights: list[float] = []
        for client in training_data.clients:
            local_model = Autoencoder(architecture)
            local_model.load_state_dict(global_model.state_dict())
            _train_local_epochs(local_model, client.features, config)
            local_states.append(local_model.state_dict())
            local_weights.append(float(client.features.row_count.value))

        averaged_state = _weighted_average_state_dicts(local_states, local_weights)
        global_model.load_state_dict(averaged_state)

    return global_model
