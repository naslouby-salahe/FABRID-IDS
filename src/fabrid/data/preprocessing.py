"""Frozen z-score feature scaling, fit on `BENIGN_TRAIN` only and reused everywhere else.

Constant-valued training columns (std == 0) are mean-centered without
rescaling rather than dividing by zero, since N-BaIoT's constructed features
can be exactly constant for some client/window combinations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_ZERO_STD_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class FeatureScaler:
    mean: np.ndarray
    std: np.ndarray

    def __post_init__(self) -> None:
        if self.mean.shape != self.std.shape:
            raise ValueError(f"mean shape {self.mean.shape} must match std shape {self.std.shape}")
        if self.mean.ndim != 1:
            raise ValueError(f"mean/std must be 1-dimensional, got ndim={self.mean.ndim}")
        if np.any(self.std <= 0):
            raise ValueError("std must be strictly positive in every dimension")

    def transform(self, features: np.ndarray) -> np.ndarray:
        if features.shape[1] != self.mean.shape[0]:
            raise ValueError(
                f"features has {features.shape[1]} columns, scaler expects {self.mean.shape[0]}"
            )
        return (features - self.mean) / self.std


def fit_feature_scaler(train_features: np.ndarray) -> FeatureScaler:
    if train_features.shape[0] == 0:
        raise ValueError("fit_feature_scaler requires at least one training row")
    mean = train_features.mean(axis=0)
    std = train_features.std(axis=0)
    std_safe = np.where(std < _ZERO_STD_EPSILON, 1.0, std)
    return FeatureScaler(mean=mean, std=std_safe)
