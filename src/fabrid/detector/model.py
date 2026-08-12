"""Fixed detector architecture: a benign-trained autoencoder.

Anomaly score is per-row mean squared reconstruction error; larger values
mean greater anomaly evidence, matching the score contract in
`fabrid.scoring.score_contract`. The architecture itself is not the
contribution and is intentionally simple.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class AutoencoderArchitecture:
    n_features: int
    hidden_dims: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.n_features < 1:
            raise ValueError(f"n_features must be positive, got {self.n_features}")
        if not self.hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer size")
        if any(dim < 1 for dim in self.hidden_dims):
            raise ValueError(f"all hidden_dims must be positive, got {self.hidden_dims}")


class Autoencoder(nn.Module):
    def __init__(self, architecture: AutoencoderArchitecture) -> None:
        super().__init__()
        self.architecture = architecture

        encoder_dims = (architecture.n_features, *architecture.hidden_dims)
        encoder_layers: list[nn.Module] = []
        for in_dim, out_dim in zip(encoder_dims, encoder_dims[1:], strict=False):
            encoder_layers += [nn.Linear(in_dim, out_dim), nn.ReLU()]
        self.encoder = nn.Sequential(*encoder_layers[:-1])  # drop final ReLU on the bottleneck

        decoder_dims = (*architecture.hidden_dims[::-1], architecture.n_features)
        decoder_layers: list[nn.Module] = []
        for in_dim, out_dim in zip(decoder_dims, decoder_dims[1:], strict=False):
            decoder_layers += [nn.Linear(in_dim, out_dim), nn.ReLU()]
        self.decoder = nn.Sequential(*decoder_layers[:-1])  # linear output layer, no final ReLU

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def reconstruction_error_scores(model: Autoencoder, features: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        inputs = torch.as_tensor(features, dtype=torch.float32)
        reconstructed = model(inputs)
        per_row_mse = torch.mean((inputs - reconstructed) ** 2, dim=1)
    return per_row_mse.numpy().astype(np.float64)
