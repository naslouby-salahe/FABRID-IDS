from __future__ import annotations

import numpy as np
import pytest
import torch

from fabrid.datasets.common import FeatureMatrix
from fabrid.detector.model import Autoencoder, AutoencoderArchitecture, reconstruction_error_scores
from fabrid.domain.values import FeatureCount, LayerWidth


def test_forward_pass_shape() -> None:
    architecture = AutoencoderArchitecture(
        feature_count=FeatureCount(10),
        hidden_layers=(LayerWidth(6), LayerWidth(3)),
    )
    output = Autoencoder(architecture)(torch.randn(4, 10))
    assert output.shape == (4, 10)


def test_reconstruction_error_is_nonnegative() -> None:
    torch.manual_seed(0)
    architecture = AutoencoderArchitecture(
        feature_count=FeatureCount(5),
        hidden_layers=(LayerWidth(3),),
    )
    rng = np.random.default_rng(0)
    features = FeatureMatrix(rng.normal(0, 0.01, size=(20, 5)).astype(np.float64))
    scores = reconstruction_error_scores(Autoencoder(architecture), features)
    assert np.all(scores.values >= 0)
    assert scores.row_count.value == 20


def test_invalid_architecture_rejected() -> None:
    with pytest.raises(ValueError):
        FeatureCount(0)
    with pytest.raises(ValueError):
        AutoencoderArchitecture(feature_count=FeatureCount(5), hidden_layers=())
    with pytest.raises(ValueError):
        LayerWidth(0)
