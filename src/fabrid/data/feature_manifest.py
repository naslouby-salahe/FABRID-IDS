"""Frozen feature manifest: the exact, hashed, ordered list of model input columns.

Freezing this once (and hashing it) prevents silent feature-set drift between
detector training and score generation.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True, slots=True)
class FeatureManifest:
    feature_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.feature_names:
            raise ValueError("feature manifest must contain at least one feature")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature manifest must not contain duplicate feature names")

    def sha256(self) -> str:
        digest = hashlib.sha256()
        for name in self.feature_names:
            digest.update(name.encode("utf-8"))
        return digest.hexdigest()


def build_feature_manifest_from_csv_header(csv_path: Path) -> FeatureManifest:
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    return FeatureManifest(feature_names=tuple(header))
