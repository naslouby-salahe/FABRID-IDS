"""Federated averaging (FedAvg) training of the fixed autoencoder detector.

Each round: every client trains a local copy for a fixed number of local
epochs on its own `BENIGN_TRAIN` rows, then the server averages client
parameters weighted by local row count. Training uses only benign rows and
never sees attack labels, keeping the detector itself unsupervised over
attacks (the FABRID decision layer is what becomes validation-informed).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from fabrid.detector.model import Autoencoder, AutoencoderArchitecture
from fabrid.evaluation.record_level import ClientId


@dataclass(frozen=True, slots=True)
class FederatedTrainingConfig:
    hidden_dims: tuple[int, ...]
    learning_rate: float
    local_epochs: int
    rounds: int
    batch_size: int
    seed: int

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.local_epochs < 1:
            raise ValueError(f"local_epochs must be at least 1, got {self.local_epochs}")
        if self.rounds < 1:
            raise ValueError(f"rounds must be at least 1, got {self.rounds}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {self.batch_size}")
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}")


def _train_local_epochs(
    model: Autoencoder, features: np.ndarray, config: FederatedTrainingConfig
) -> None:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.MSELoss()
    inputs = torch.as_tensor(features, dtype=torch.float32)
    n_rows = inputs.shape[0]

    for _ in range(config.local_epochs):
        permutation = torch.randperm(n_rows)
        for start in range(0, n_rows, config.batch_size):
            batch_indices = permutation[start : start + config.batch_size]
            batch = inputs[batch_indices]
            optimizer.zero_grad()
            reconstructed = model(batch)
            loss = loss_fn(reconstructed, batch)
            loss.backward()
            optimizer.step()  # pyright: ignore[reportUnknownMemberType]  # torch stub gap


def _weighted_average_state_dicts(
    state_dicts: list[dict[str, torch.Tensor]], weights: list[float]
) -> dict[str, torch.Tensor]:
    total_weight = sum(weights)
    averaged: dict[str, torch.Tensor] = {}
    for key in state_dicts[0]:
        stacked = torch.stack(
            [state[key].float() * w for state, w in zip(state_dicts, weights, strict=True)]
        )
        averaged[key] = stacked.sum(dim=0) / total_weight
    return averaged


def train_federated_autoencoder(
    client_train_features: Mapping[ClientId, np.ndarray], config: FederatedTrainingConfig
) -> Autoencoder:
    if not client_train_features:
        raise ValueError("train_federated_autoencoder requires at least one client")

    n_features = next(iter(client_train_features.values())).shape[1]
    for client_id, features in client_train_features.items():
        if features.shape[1] != n_features:
            raise ValueError(
                f"client {client_id} has {features.shape[1]} features, expected {n_features}"
            )

    torch.manual_seed(config.seed)  # pyright: ignore[reportUnknownMemberType]  # torch stub gap
    architecture = AutoencoderArchitecture(n_features=n_features, hidden_dims=config.hidden_dims)
    global_model = Autoencoder(architecture)

    for _ in range(config.rounds):
        local_states: list[dict[str, torch.Tensor]] = []
        local_weights: list[float] = []
        for features in client_train_features.values():
            local_model = Autoencoder(architecture)
            local_model.load_state_dict(global_model.state_dict())
            _train_local_epochs(local_model, features, config)
            local_states.append(local_model.state_dict())
            local_weights.append(float(features.shape[0]))

        global_model.load_state_dict(_weighted_average_state_dicts(local_states, local_weights))

    return global_model
