"""Integration check against the real symlinked N-BaIoT raw data.

Skipped automatically when the shared raw-data symlink is not present (e.g. a
clean checkout on a machine without access to the shared dataset storage).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fabrid.data.feature_manifest import build_feature_manifest_from_csv_header
from fabrid.data.nbaiot_reader import read_device_directory
from fabrid.evaluation.record_level import ClientId

_RAW_NBAIOT_DIR = Path(__file__).parents[2] / "data" / "raw" / "N-BaIoT"

_EXPECTED_BENIGN_ROWS = {
    "Danmini_Doorbell": 49_548,
    "Ennio_Doorbell": 39_100,
    "Ecobee_Thermostat": 13_113,
    "Philips_B120N10_Baby_Monitor": 175_240,
    "Provision_PT_737E_Security_Camera": 62_154,
    "Provision_PT_838_Security_Camera": 98_514,
    "SimpleHome_XCS7_1002_WHT_Security_Camera": 46_585,
    "SimpleHome_XCS7_1003_WHT_Security_Camera": 19_528,
    "Samsung_SNH_1011_N_Webcam": 52_150,
}
_EXPECTED_FEATURE_COUNT = 115

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _RAW_NBAIOT_DIR.exists(), reason="raw N-BaIoT data not present at data/raw/N-BaIoT"
    ),
]


@pytest.mark.parametrize(("device_name", "expected_rows"), _EXPECTED_BENIGN_ROWS.items())
def test_real_benign_row_counts_match_published_table(device_name: str, expected_rows: int) -> None:
    data = read_device_directory(ClientId(device_name), _RAW_NBAIOT_DIR / device_name)
    assert data.benign_features.shape == (expected_rows, _EXPECTED_FEATURE_COUNT)


@pytest.mark.parametrize("device_name", _EXPECTED_BENIGN_ROWS.keys())
def test_real_feature_manifest_identical_across_devices(device_name: str) -> None:
    reference = build_feature_manifest_from_csv_header(
        _RAW_NBAIOT_DIR / "Danmini_Doorbell" / "benign_traffic.csv"
    )
    manifest = build_feature_manifest_from_csv_header(
        _RAW_NBAIOT_DIR / device_name / "benign_traffic.csv"
    )
    assert manifest.feature_names == reference.feature_names
    assert manifest.sha256() == reference.sha256()
    assert len(manifest.feature_names) == _EXPECTED_FEATURE_COUNT
