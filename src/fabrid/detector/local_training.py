from __future__ import annotations

import torch
from torch import nn

from fabrid.datasets.common import FeatureMatrix
from fabrid.detector.model import Autoencoder, resolve_device
from fabrid.domain.values import BatchSize, LearningRate, LocalEpochCount


def train_local_autoencoder(
    model: Autoencoder,
    features: FeatureMatrix,
    learning_rate: LearningRate,
    local_epochs: LocalEpochCount,
    batch_size: BatchSize,
) -> None:
    device = resolve_device()
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate.value)
    loss_function = nn.MSELoss()
    inputs = torch.as_tensor(features.values, dtype=torch.float32, device=device)
    row_count = inputs.shape[0]
    for _ in range(local_epochs.value):
        permutation = torch.randperm(row_count, device=device)
        for start in range(0, row_count, batch_size.value):
            batch = inputs[permutation[start : start + batch_size.value]]
            optimizer.zero_grad()
            loss = loss_function(model(batch), batch)
            loss.backward()
            optimizer.step()  # pyright: ignore[reportUnknownMemberType]
    model.to("cpu")
