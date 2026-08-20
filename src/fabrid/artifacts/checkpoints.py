from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Annotated

import numpy as np
import torch
from pydantic import StringConstraints
from safetensors import torch as safetensors_torch
from torch import nn

_SCALER_MEAN_KEY = "mean"
TensorName = Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]
_SCALER_STANDARD_DEVIATION_KEY = "standard_deviation"


def save_module_state(module: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = module.state_dict()
    safetensors_torch.save_file(
        OrderedDict(
            (name, tensor.detach().cpu().contiguous()) for name, tensor in sorted(state.items())
        ),
        path,
    )


def load_module_state(module: nn.Module, path: Path) -> None:
    state = dict(safetensors_torch.load_file(path))
    module.load_state_dict(state)


def save_scaler(mean: np.ndarray, standard_deviation: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safetensors_torch.save_file(
        OrderedDict(
            (
                (
                    _SCALER_MEAN_KEY,
                    torch.from_numpy(np.ascontiguousarray(mean)),
                ),
                (
                    _SCALER_STANDARD_DEVIATION_KEY,
                    torch.from_numpy(np.ascontiguousarray(standard_deviation)),
                ),
            )
        ),
        path,
    )


def load_scaler(path: Path) -> tuple[np.ndarray, np.ndarray]:
    tensors = dict(safetensors_torch.load_file(path))
    return (
        tensors[_SCALER_MEAN_KEY].numpy().copy(),
        tensors[_SCALER_STANDARD_DEVIATION_KEY].numpy().copy(),
    )
