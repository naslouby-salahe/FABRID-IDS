from __future__ import annotations

import numpy as np
import pytest
import torch

from fabrid.detector.model import Autoencoder, AutoencoderArchitecture, reconstruction_error_scores


def test_forward_pass_shape() -> None:
    architecture = AutoencoderArchitecture(n_features=10, hidden_dims=(6, 3))
    model = Autoencoder(architecture)
    inputs = torch.randn(4, 10)
    output = model(inputs)
    assert output.shape == (4, 10)


def test_reconstruction_error_is_nonnegative_and_higher_for_novel_inputs() -> None:
    torch.manual_seed(0)
    architecture = AutoencoderArchitecture(n_features=5, hidden_dims=(3,))
    model = Autoencoder(architecture)

    rng = np.random.default_rng(0)
    normal_like = rng.normal(0, 0.01, size=(20, 5)).astype(np.float64)
    scores = reconstruction_error_scores(model, normal_like)
    assert np.all(scores >= 0)
    assert scores.shape == (20,)


def test_invalid_architecture_rejected() -> None:
    with pytest.raises(ValueError):
        AutoencoderArchitecture(n_features=0, hidden_dims=(3,))
    with pytest.raises(ValueError):
        AutoencoderArchitecture(n_features=5, hidden_dims=())
    with pytest.raises(ValueError):
        AutoencoderArchitecture(n_features=5, hidden_dims=(0,))
