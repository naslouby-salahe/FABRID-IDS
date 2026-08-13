from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fabrid.datasets.manifests import FeatureManifest, build_feature_manifest_from_csv_header
from fabrid.domain.identifiers import ColumnName


def test_feature_manifest_rejects_empty_and_duplicates() -> None:
    with pytest.raises(ValueError):
        FeatureManifest(features=())
    with pytest.raises(ValueError):
        FeatureManifest(features=(ColumnName("a"), ColumnName("a")))


def test_feature_manifest_digest_is_deterministic_and_order_sensitive() -> None:
    first = FeatureManifest((ColumnName("x"), ColumnName("y"), ColumnName("z")))
    same = FeatureManifest((ColumnName("x"), ColumnName("y"), ColumnName("z")))
    reordered = FeatureManifest((ColumnName("z"), ColumnName("y"), ColumnName("x")))

    assert first.digest() == same.digest()
    assert first.digest() != reordered.digest()


def test_feature_manifest_is_built_from_csv_header(tmp_path: Path) -> None:
    path = tmp_path / "features.csv"
    pd.DataFrame({"f1": (1.0,), "f2": (2.0,), "f3": (3.0,)}).to_csv(path, index=False)

    manifest = build_feature_manifest_from_csv_header(path)

    assert manifest.features == (ColumnName("f1"), ColumnName("f2"), ColumnName("f3"))
