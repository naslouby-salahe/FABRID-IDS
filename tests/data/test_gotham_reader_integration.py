"""Integration check against a real extracted Gotham 2025 processed/ CSV.

Skipped automatically when the shared raw-data symlink is not present.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fabrid.data.gotham_reader import read_processed_csv

_PROCESSED_DIR = (
    Path(__file__).parents[2] / "data" / "raw" / "Gotham2025" / "extracted" / "processed"
)
_SAMPLE_FILE = _PROCESSED_DIR / "iotsim-air-quality-1.csv"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _SAMPLE_FILE.exists(), reason="raw Gotham 2025 data not present"),
]

_SAMPLE_ROWS = 5000


def test_real_processed_csv_splits_benign_and_attack() -> None:
    result = read_processed_csv(
        _SAMPLE_FILE,
        excluded_features=frozenset({"eth.dst", "frame.protocols"}),
        numeric_parse_success_threshold=0.9,
        nrows=_SAMPLE_ROWS,
    )
    assert len(result.kept_feature_columns) > 0
    assert result.rows_dropped_unparseable / _SAMPLE_ROWS < 0.5
    assert (
        len(result.benign_features_by_client) + len(result.attack_features_by_client_and_subtype)
        > 0
    )
