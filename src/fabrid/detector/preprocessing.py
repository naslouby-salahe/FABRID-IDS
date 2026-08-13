from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fabrid.datasets.common import FeatureMatrix
from fabrid.domain.identifiers import ClientId

_ZERO_STANDARD_DEVIATION_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class FeatureScaler:
    mean: np.ndarray
    standard_deviation: np.ndarray

    def __post_init__(self) -> None:
        if self.mean.shape != self.standard_deviation.shape:
            raise ValueError("scaler mean and standard deviation must share shape")
        if self.mean.ndim != 1:
            raise ValueError("scaler statistics must be one-dimensional")
        if np.any(self.standard_deviation <= 0):
            raise ValueError("scaler standard deviation must be strictly positive")

    def transform(self, features: FeatureMatrix) -> FeatureMatrix:
        if features.feature_count.value != self.mean.shape[0]:
            raise ValueError("feature matrix width does not match scaler statistics")
        return FeatureMatrix(
            (features.values - self.mean) / self.standard_deviation
        )


@dataclass(frozen=True, slots=True)
class ClientScaler:
    client_id: ClientId
    scaler: FeatureScaler


@dataclass(frozen=True, slots=True)
class FederatedScalers:
    clients: tuple[ClientScaler, ...]

    def __post_init__(self) -> None:
        if not self.clients:
            raise ValueError("federated scalers require at least one client")
        client_ids = tuple(client.client_id for client in self.clients)
        if len(set(client_ids)) != len(client_ids):
            raise ValueError("federated scalers contain duplicate clients")

    def for_client(self, client_id: ClientId) -> FeatureScaler:
        for client in self.clients:
            if client.client_id == client_id:
                return client.scaler
        raise KeyError(client_id.value)


def fit_feature_scaler(train_features: FeatureMatrix) -> FeatureScaler:
    if train_features.row_count.value == 0:
        raise ValueError("feature scaler requires at least one training row")
    mean = train_features.values.mean(axis=0)
    standard_deviation = train_features.values.std(axis=0)
    safe_standard_deviation = np.where(
        standard_deviation < _ZERO_STANDARD_DEVIATION_EPSILON,
        1.0,
        standard_deviation,
    )
    return FeatureScaler(
        mean=mean,
        standard_deviation=safe_standard_deviation,
    )
