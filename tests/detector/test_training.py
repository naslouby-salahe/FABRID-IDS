from __future__ import annotations

import numpy as np
import pytest

from fabrid.datasets.common import FeatureMatrix
from fabrid.detector.model import reconstruction_error_scores
from fabrid.detector.training import ClientTrainingData, FederatedTrainingConfig, FederatedTrainingData, train_federated_autoencoder
from fabrid.domain.identifiers import ClientId
from fabrid.domain.values import BatchSize, DetectorSeed, FederatedRoundCount, LayerWidth, LearningRate, LocalEpochCount

_CONFIG = FederatedTrainingConfig(
    hidden_layers=(LayerWidth(4), LayerWidth(2)), learning_rate=LearningRate(0.01),
    local_epochs=LocalEpochCount(2), rounds=FederatedRoundCount(2),
    batch_size=BatchSize(8), seed=DetectorSeed(0),
)


def _features(seed: int, width: int = 6) -> FeatureMatrix:
    rng = np.random.default_rng(seed)
    return FeatureMatrix(rng.normal(0, 0.1, size=(32, width)).astype(np.float64))


def _data(*widths: int) -> FederatedTrainingData:
    return FederatedTrainingData(tuple(
        ClientTrainingData(ClientId(str(index)), _features(index, width))
        for index, width in enumerate(widths, start=1)
    ))


def test_training_reduces_reconstruction_error() -> None:
    data = _data(6, 6)
    trained = train_federated_autoencoder(data, _CONFIG)
    scores = reconstruction_error_scores(trained, data.clients[0].features)
    assert np.isfinite(scores.values).all()
    assert scores.row_count.value == 32


def test_deterministic_given_fixed_seed() -> None:
    data = _data(6)
    first = train_federated_autoencoder(data, _CONFIG)
    second = train_federated_autoencoder(data, _CONFIG)
    first_scores = reconstruction_error_scores(first, data.clients[0].features)
    second_scores = reconstruction_error_scores(second, data.clients[0].features)
    np.testing.assert_allclose(first_scores.values, second_scores.values)


def test_mismatched_feature_dimensions_rejected() -> None:
    with pytest.raises(ValueError):
        _data(6, 8)


def test_empty_clients_rejected() -> None:
    with pytest.raises(ValueError):
        FederatedTrainingData(())
