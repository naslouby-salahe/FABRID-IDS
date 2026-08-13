from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from fabrid.datasets.common import FeatureMatrix
from fabrid.domain.scores import ScoreVector
from fabrid.domain.values import FeatureCount, LayerWidth


@dataclass(frozen=True, slots=True)
class AutoencoderArchitecture:
    feature_count: FeatureCount
    hidden_layers: tuple[LayerWidth, ...]

    def __post_init__(self) -> None:
        if not self.hidden_layers:
            raise ValueError("autoencoder requires at least one hidden layer")


class Autoencoder(nn.Module):
    def __init__(self, architecture: AutoencoderArchitecture) -> None:
        super().__init__()
        self.architecture = architecture

        encoder_dimensions = (
            architecture.feature_count.value,
            *(layer.value for layer in architecture.hidden_layers),
        )
        encoder_layers: list[nn.Module] = []
        for input_width, output_width in zip(
            encoder_dimensions,
            encoder_dimensions[1:],
            strict=False,
        ):
            encoder_layers.extend(
                (nn.Linear(input_width, output_width), nn.ReLU())
            )
        self.encoder = nn.Sequential(*encoder_layers[:-1])

        decoder_dimensions = (
            *(layer.value for layer in reversed(architecture.hidden_layers)),
            architecture.feature_count.value,
        )
        decoder_layers: list[nn.Module] = []
        for input_width, output_width in zip(
            decoder_dimensions,
            decoder_dimensions[1:],
            strict=False,
        ):
            decoder_layers.extend(
                (nn.Linear(input_width, output_width), nn.ReLU())
            )
        self.decoder = nn.Sequential(*decoder_layers[:-1])

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(features))


def resolve_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def reconstruction_error_scores(
    model: Autoencoder,
    features: FeatureMatrix,
) -> ScoreVector:
    device = resolve_device()
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        inputs = torch.as_tensor(
            features.values,
            dtype=torch.float32,
            device=device,
        )
        reconstructed = model(inputs)
        per_row_mse = torch.mean((inputs - reconstructed) ** 2, dim=1)
    return ScoreVector(per_row_mse.cpu().numpy().astype(np.float64))
