from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from fabrid.config import FeatureCount, LayerWidth


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
        encoder_dimensions = (architecture.feature_count, *architecture.hidden_layers)
        encoder_layers: list[nn.Module] = []
        for index in range(len(encoder_dimensions) - 1):
            input_width = encoder_dimensions[index]
            output_width = encoder_dimensions[index + 1]
            encoder_layers.extend((nn.Linear(input_width, output_width), nn.ReLU()))
        self.encoder = nn.Sequential(*encoder_layers[:-1])
        decoder_dimensions = (*reversed(architecture.hidden_layers), architecture.feature_count)
        decoder_layers: list[nn.Module] = []
        for index in range(len(decoder_dimensions) - 1):
            input_width = decoder_dimensions[index]
            output_width = decoder_dimensions[index + 1]
            decoder_layers.extend((nn.Linear(input_width, output_width), nn.ReLU()))
        self.decoder = nn.Sequential(*decoder_layers[:-1])

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(features))


def resolve_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
