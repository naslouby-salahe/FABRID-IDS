from __future__ import annotations

import numpy as np
import pytest
import torch

from fabrid.detector.model import Autoencoder, AutoencoderArchitecture, reconstruction_error_scores
from fabrid.detector.training import FederatedTrainingConfig, train_federated_autoencoder
from fabrid.evaluation.record_level import ClientId

_TINY_CONFIG = FederatedTrainingConfig(
    hidden_dims=(4, 2), learning_rate=0.01, local_epochs=2, rounds=2, batch_size=8, seed=0
)


def _client_features(seed: int, n_rows: int = 32, n_features: int = 6) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0, 0.1, size=(n_rows, n_features)).astype(np.float64)


def test_training_reduces_reconstruction_error() -> None:
    client_features = {
        ClientId("1"): _client_features(1),
        ClientId("2"): _client_features(2),
    }

    torch.manual_seed(_TINY_CONFIG.seed)
    untrained = Autoencoder(AutoencoderArchitecture(n_features=6, hidden_dims=(4, 2)))
    untrained_error = reconstruction_error_scores(untrained, client_features[ClientId("1")]).mean()

    trained = train_federated_autoencoder(client_features, _TINY_CONFIG)
    trained_error = reconstruction_error_scores(trained, client_features[ClientId("1")]).mean()

    assert trained_error < untrained_error


def test_deterministic_given_fixed_seed() -> None:
    client_features = {ClientId("1"): _client_features(1)}
    first = train_federated_autoencoder(client_features, _TINY_CONFIG)
    second = train_federated_autoencoder(client_features, _TINY_CONFIG)

    first_scores = reconstruction_error_scores(first, client_features[ClientId("1")])
    second_scores = reconstruction_error_scores(second, client_features[ClientId("1")])
    np.testing.assert_allclose(first_scores, second_scores)


def test_mismatched_feature_dimensions_rejected() -> None:
    client_features = {
        ClientId("1"): _client_features(1, n_features=6),
        ClientId("2"): _client_features(2, n_features=8),
    }
    with pytest.raises(ValueError):
        train_federated_autoencoder(client_features, _TINY_CONFIG)


def test_empty_clients_rejected() -> None:
    with pytest.raises(ValueError):
        train_federated_autoencoder({}, _TINY_CONFIG)


def test_invalid_config_rejected() -> None:
    with pytest.raises(ValueError):
        FederatedTrainingConfig(
            hidden_dims=(4,), learning_rate=-0.1, local_epochs=1, rounds=1, batch_size=8, seed=0
        )
    with pytest.raises(ValueError):
        FederatedTrainingConfig(
            hidden_dims=(4,), learning_rate=0.01, local_epochs=0, rounds=1, batch_size=8, seed=0
        )
