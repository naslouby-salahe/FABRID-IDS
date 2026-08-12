from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from fabrid.data.preprocessing import FeatureScaler
from fabrid.detector.model import Autoencoder, AutoencoderArchitecture
from fabrid.detector.persistence import (
    load_detector_state,
    save_detector_state,
)
from fabrid.evaluation.record_level import ClientId


def test_save_and_load_detector_state_roundtrips(tmp_path: Path) -> None:
    architecture = AutoencoderArchitecture(n_features=4, hidden_dims=(3,))
    torch.manual_seed(0)
    model = Autoencoder(architecture)
    scalers = {
        ClientId("1"): FeatureScaler(
            mean=np.array([0.0, 1.0, 2.0, 3.0]), std=np.array([1.0, 1.0, 1.0, 1.0])
        ),
        ClientId("2"): FeatureScaler(
            mean=np.array([4.0, 5.0, 6.0, 7.0]), std=np.array([2.0, 2.0, 2.0, 2.0])
        ),
    }

    hashes = save_detector_state(tmp_path, model, architecture, scalers)
    assert len(hashes.model_sha256) == 64
    assert len(hashes.scalers_sha256) == 64

    loaded_model, loaded_scalers = load_detector_state(tmp_path, architecture)

    original_state = model.state_dict()
    loaded_state = loaded_model.state_dict()
    assert original_state.keys() == loaded_state.keys()
    for key in original_state:
        assert torch.equal(original_state[key], loaded_state[key])

    assert set(loaded_scalers) == set(scalers)
    for client_id, scaler in scalers.items():
        np.testing.assert_array_equal(loaded_scalers[client_id].mean, scaler.mean)
        np.testing.assert_array_equal(loaded_scalers[client_id].std, scaler.std)


def test_save_detector_state_hashes_are_deterministic(tmp_path: Path) -> None:
    architecture = AutoencoderArchitecture(n_features=2, hidden_dims=(2,))
    torch.manual_seed(1)
    model = Autoencoder(architecture)
    scalers = {ClientId("1"): FeatureScaler(mean=np.array([0.0, 0.0]), std=np.array([1.0, 1.0]))}

    first = save_detector_state(tmp_path / "a", model, architecture, scalers)
    second = save_detector_state(tmp_path / "b", model, architecture, scalers)

    assert first.model_sha256 == second.model_sha256
    assert first.scalers_sha256 == second.scalers_sha256
