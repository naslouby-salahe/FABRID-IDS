from __future__ import annotations

import numpy as np
import pytest

from fabrid.datasets.common import FeatureMatrix
from fabrid.detector.preprocessing import fit_feature_scaler


def test_scaler_normalizes_training_features() -> None:
    rng = np.random.default_rng(0)
    train = FeatureMatrix(rng.normal(loc=5.0, scale=2.0, size=(1_000, 3)))

    scaled = fit_feature_scaler(train).transform(train)

    assert np.allclose(scaled.values.mean(axis=0), 0.0, atol=1e-9)
    assert np.allclose(scaled.values.std(axis=0), 1.0, atol=1e-9)


def test_constant_column_is_centered_without_non_finite_values() -> None:
    train = FeatureMatrix(
        np.column_stack(
            (np.full(10, 3.0), np.arange(10, dtype=np.float64))
        )
    )

    scaled = fit_feature_scaler(train).transform(train)

    assert np.allclose(scaled.values[:, 0], 0.0)
    assert np.isfinite(scaled.values).all()


def test_transform_uses_training_statistics_for_other_features() -> None:
    train = FeatureMatrix(np.asarray(((0.0,), (10.0,)), dtype=np.float64))
    other = FeatureMatrix(np.asarray(((5.0,),), dtype=np.float64))

    scaled = fit_feature_scaler(train).transform(other)

    assert scaled.values[0, 0] == pytest.approx(0.0)


def test_empty_training_features_are_rejected() -> None:
    with pytest.raises(ValueError):
        fit_feature_scaler(FeatureMatrix(np.empty((0, 3), dtype=np.float64)))


def test_transform_rejects_feature_width_mismatch() -> None:
    scaler = fit_feature_scaler(
        FeatureMatrix(np.asarray(((1.0, 2.0), (3.0, 4.0)), dtype=np.float64))
    )

    with pytest.raises(ValueError):
        scaler.transform(
            FeatureMatrix(np.asarray(((1.0, 2.0, 3.0),), dtype=np.float64))
        )
