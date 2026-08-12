from __future__ import annotations

import numpy as np
import pytest

from fabrid.data.preprocessing import fit_feature_scaler


def test_scaler_zero_means_and_unit_variance_on_training_data() -> None:
    rng = np.random.default_rng(0)
    train = rng.normal(loc=5.0, scale=2.0, size=(1000, 3))
    scaler = fit_feature_scaler(train)
    scaled = scaler.transform(train)
    assert np.allclose(scaled.mean(axis=0), 0.0, atol=1e-9)
    assert np.allclose(scaled.std(axis=0), 1.0, atol=1e-9)


def test_constant_column_mean_centered_not_divided_by_zero() -> None:
    train = np.column_stack([np.full(10, 3.0), np.arange(10, dtype=np.float64)])
    scaler = fit_feature_scaler(train)
    scaled = scaler.transform(train)
    assert np.allclose(scaled[:, 0], 0.0)  # constant column collapses to exactly 0
    assert not np.any(np.isnan(scaled))
    assert not np.any(np.isinf(scaled))


def test_transform_applies_train_statistics_to_other_data() -> None:
    train = np.array([[0.0], [10.0]])
    scaler = fit_feature_scaler(train)
    test_data = np.array([[5.0]])
    scaled = scaler.transform(test_data)
    assert scaled[0, 0] == pytest.approx(0.0)  # midpoint of train range -> 0 after centering


def test_empty_training_data_rejected() -> None:
    with pytest.raises(ValueError):
        fit_feature_scaler(np.empty((0, 3)))


def test_column_count_mismatch_rejected() -> None:
    scaler = fit_feature_scaler(np.array([[1.0, 2.0], [3.0, 4.0]]))
    with pytest.raises(ValueError):
        scaler.transform(np.array([[1.0, 2.0, 3.0]]))
