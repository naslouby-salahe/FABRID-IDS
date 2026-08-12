"""Integration check against the real downloaded CIC IoT-DIAD 2024 packet-level CSVs.

Skipped automatically when the shared raw-data symlink is not present.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fabrid.config.external_datasets import load_external_dataset_config
from fabrid.data.cic_iot_diad_reader import read_packet_csv

_RAW_DIR = (
    Path(__file__).parents[2]
    / "data"
    / "raw"
    / "CIC_IoT_DIAD_2024"
    / "Device Identification_Anomaly Detection - Packet Based Features"
    / "BruteForce"
    / "DictionaryBruteForce"
    / "DictionaryBruteForce.csv"
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _RAW_DIR.exists(), reason="raw CIC IoT-DIAD 2024 data not present"),
]

_SAMPLE_ROWS = 5000


def test_real_packet_csv_ingests_multiple_devices_with_low_row_loss() -> None:
    config = load_external_dataset_config("cic_iot_diad_2024")
    result = read_packet_csv(
        _RAW_DIR,
        config.excluded_features,
        config.numeric_parse_success_threshold,
        device_column="src_mac",
        nrows=_SAMPLE_ROWS,
    )

    assert len(result.device_features) > 1
    assert len(result.kept_feature_columns) > 0
    assert result.rows_dropped_unparseable / _SAMPLE_ROWS < 0.01
    for matrix in result.device_features.values():
        assert matrix.shape[1] == len(result.kept_feature_columns)
        assert matrix.shape[0] > 0
