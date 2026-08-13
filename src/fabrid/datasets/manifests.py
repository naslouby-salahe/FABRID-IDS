from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from fabrid.domain.identifiers import ArtifactDigest, ColumnName


@dataclass(frozen=True, slots=True)
class FeatureManifest:
    features: tuple[ColumnName, ...]

    def __post_init__(self) -> None:
        if not self.features:
            raise ValueError("feature manifest must contain at least one feature")
        if len(set(self.features)) != len(self.features):
            raise ValueError("feature manifest must not contain duplicate features")

    def digest(self) -> ArtifactDigest:
        digest = hashlib.sha256()
        for feature in self.features:
            encoded = feature.value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
            digest.update(encoded)
        return ArtifactDigest(digest.hexdigest())


def build_feature_manifest_from_csv_header(csv_path: Path) -> FeatureManifest:
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    return FeatureManifest(
        features=tuple(ColumnName(str(column)) for column in header)
    )
