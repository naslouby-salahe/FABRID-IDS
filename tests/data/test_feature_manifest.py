from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fabrid.data.feature_manifest import FeatureManifest, build_feature_manifest_from_csv_header


def test_manifest_rejects_empty() -> None:
    with pytest.raises(ValueError):
        FeatureManifest(feature_names=())


def test_manifest_rejects_duplicates() -> None:
    with pytest.raises(ValueError):
        FeatureManifest(feature_names=("a", "a"))


def test_sha256_deterministic_and_order_sensitive() -> None:
    a = FeatureManifest(("x", "y", "z"))
    b = FeatureManifest(("x", "y", "z"))
    c = FeatureManifest(("z", "y", "x"))
    assert a.sha256() == b.sha256()
    assert a.sha256() != c.sha256()


def test_build_from_csv_header(tmp_path: Path) -> None:
    path = tmp_path / "features.csv"
    pd.DataFrame({"f1": [1.0], "f2": [2.0], "f3": [3.0]}).to_csv(path, index=False)
    manifest = build_feature_manifest_from_csv_header(path)
    assert manifest.feature_names == ("f1", "f2", "f3")
