from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from fabrid.detector.model import Autoencoder, AutoencoderArchitecture
from fabrid.detector.persistence import load_detector_state, save_detector_state
from fabrid.detector.preprocessing import ClientScaler, FeatureScaler, FederatedScalers
from fabrid.domain.identifiers import ClientId
from fabrid.domain.values import FeatureCount, LayerWidth


def _scalers() -> FederatedScalers:
    return FederatedScalers(tuple(
        ClientScaler(ClientId(name), FeatureScaler(mean=np.array(mean), standard_deviation=np.array(std)))
        for name, mean, std in (
            ("1", [0., 1., 2., 3.], [1., 1., 1., 1.]),
            ("2", [4., 5., 6., 7.], [2., 2., 2., 2.]),
        )
    ))


def test_detector_persistence_roundtrips(tmp_path: Path) -> None:
    model = Autoencoder(AutoencoderArchitecture(FeatureCount(4), (LayerWidth(3),)))
    torch.manual_seed(0)
    scalers = _scalers()
    artifacts = save_detector_state(tmp_path, model, scalers)
    assert len(artifacts.model.value) == 64
    loaded = load_detector_state(tmp_path)
    for key, value in model.state_dict().items():
        assert torch.equal(value, loaded.model.state_dict()[key])
    for client in scalers.clients:
        actual = loaded.scalers.for_client(client.client_id)
        np.testing.assert_array_equal(actual.mean, client.scaler.mean)
        np.testing.assert_array_equal(actual.standard_deviation, client.scaler.standard_deviation)


def test_detector_digests_are_deterministic(tmp_path: Path) -> None:
    model = Autoencoder(AutoencoderArchitecture(FeatureCount(4), (LayerWidth(3),)))
    scalers = _scalers()
    first = save_detector_state(tmp_path / "a", model, scalers)
    second = save_detector_state(tmp_path / "b", model, scalers)
    assert first == second
