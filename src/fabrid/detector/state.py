from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from fabrid.domain.identifiers import TensorParameterName
from fabrid.domain.values import RowCount


@dataclass(frozen=True, slots=True)
class TensorParameter:
    name: TensorParameterName
    tensor: torch.Tensor


@dataclass(frozen=True, slots=True)
class TensorState:
    parameters: tuple[TensorParameter, ...]

    def __post_init__(self) -> None:
        if not self.parameters:
            raise ValueError("tensor state must contain at least one parameter")
        names = tuple(parameter.name for parameter in self.parameters)
        if len(set(names)) != len(names):
            raise ValueError("tensor state contains duplicate parameter names")

    @classmethod
    def from_module(cls, module: nn.Module) -> TensorState:
        return cls(
            tuple(
                TensorParameter(
                    name=TensorParameterName(name),
                    tensor=tensor.detach().clone(),
                )
                for name, tensor in module.state_dict().items()
            )
        )

    @classmethod
    def read_safetensors(cls, path: Path) -> TensorState:
        tensors = load_file(path)
        return cls(
            tuple(
                TensorParameter(
                    name=TensorParameterName(name),
                    tensor=tensor,
                )
                for name, tensor in sorted(tensors.items())
            )
        )

    def write_safetensors(self, path: Path) -> None:
        save_file(
            OrderedDict(
                (
                    parameter.name.value,
                    parameter.tensor.detach().cpu().contiguous(),
                )
                for parameter in self.parameters
            ),
            path,
        )

    def tensor(self, name: TensorParameterName) -> torch.Tensor:
        for parameter in self.parameters:
            if parameter.name == name:
                return parameter.tensor
        raise KeyError(name.value)

    def load_into(self, module: nn.Module) -> None:
        module.load_state_dict(
            OrderedDict(
                (parameter.name.value, parameter.tensor)
                for parameter in self.parameters
            )
        )


@dataclass(frozen=True, slots=True)
class WeightedTensorState:
    state: TensorState
    weight: RowCount

    def __post_init__(self) -> None:
        if self.weight.value == 0:
            raise ValueError("weighted tensor state requires a positive row count")


def average_weighted_tensor_states(
    states: tuple[WeightedTensorState, ...],
) -> TensorState:
    if not states:
        raise ValueError("weighted tensor averaging requires at least one state")

    reference_names = tuple(parameter.name for parameter in states[0].state.parameters)
    for weighted in states[1:]:
        names = tuple(parameter.name for parameter in weighted.state.parameters)
        if names != reference_names:
            raise ValueError("all tensor states must contain the same ordered parameters")

    total_weight = sum(weighted.weight.value for weighted in states)
    parameters: list[TensorParameter] = []
    for name in reference_names:
        stacked = torch.stack(
            tuple(
                weighted.state.tensor(name).float() * weighted.weight.value
                for weighted in states
            )
        )
        parameters.append(
            TensorParameter(
                name=name,
                tensor=stacked.sum(dim=0) / total_weight,
            )
        )
    return TensorState(tuple(parameters))
