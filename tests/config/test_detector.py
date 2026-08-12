from __future__ import annotations

import pytest

from fabrid.config.detector import (
    DetectorHyperparameters,
    load_detector_hyperparameters,
    load_detector_seeds,
)


def test_load_detector_hyperparameters_reads_frozen_yaml() -> None:
    hyperparameters = load_detector_hyperparameters()
    assert hyperparameters.hidden_dims == (64, 16)
    assert hyperparameters.learning_rate == pytest.approx(0.001)
    assert hyperparameters.local_epochs == 3
    assert hyperparameters.rounds == 10
    assert hyperparameters.batch_size == 64


def test_load_detector_seeds_matches_ten_seed_protocol() -> None:
    seeds = load_detector_seeds()
    assert seeds == tuple(range(10))


def test_invalid_hyperparameters_rejected() -> None:
    with pytest.raises(ValueError):
        DetectorHyperparameters(
            hidden_dims=(), learning_rate=0.01, local_epochs=1, rounds=1, batch_size=8
        )
    with pytest.raises(ValueError):
        DetectorHyperparameters(
            hidden_dims=(4,), learning_rate=0.0, local_epochs=1, rounds=1, batch_size=8
        )
