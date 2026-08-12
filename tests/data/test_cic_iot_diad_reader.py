from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fabrid.data.cic_iot_diad_reader import (
    read_packet_csv,
    select_numeric_feature_columns,
)
from fabrid.evaluation.record_level import ClientId


def _write_csv(tmp_path: Path) -> Path:
    frame = pd.DataFrame(
        {
            "src_mac": ["dev1", "dev1", "dev2", "dev2", "dev2"],
            "device_mac": ["dev1", "dev1", "dev2", "dev2", "dev2"],
            "protocol_name": ["TLS", "TLS", "none", "none", "TLS"],
            "ttl": [64, 64, 63, "corrupt", 63],
            "payload_entropy": [3.5, 3.6, 1.2, 1.3, 1.1],
            "anomaly_label": [0, 0, 1, 1, 1],
        }
    )
    path = tmp_path / "sample.csv"
    frame.to_csv(path, index=False)
    return path


def test_select_numeric_feature_columns_excludes_configured_and_device_columns() -> None:
    frame = pd.DataFrame(
        {
            "src_mac": ["dev1", "dev2"],
            "device_mac": ["dev1", "dev2"],
            "ttl": [64, 63],
        }
    )
    reports = select_numeric_feature_columns(
        frame,
        excluded_features=frozenset({"device_mac"}),
        numeric_parse_success_threshold=0.999,
        device_column="src_mac",
    )
    kept = {r.column for r in reports if r.kept}
    assert kept == {"ttl"}
    excluded = {r.column: r.kept for r in reports if not r.kept}
    assert excluded == {"src_mac": False, "device_mac": False}


def test_select_numeric_feature_columns_drops_low_parse_rate_columns() -> None:
    frame = pd.DataFrame({"mostly_bad": ["1", "x", "y", "z", "2"]})
    reports = select_numeric_feature_columns(
        frame,
        excluded_features=frozenset(),
        numeric_parse_success_threshold=0.999,
        device_column="nonexistent",
    )
    assert reports[0].kept is False
    assert reports[0].parse_success_fraction == pytest.approx(0.4)


def test_read_packet_csv_groups_by_device_and_drops_unparseable_rows(tmp_path: Path) -> None:
    path = _write_csv(tmp_path)
    result = read_packet_csv(
        path,
        excluded_features=frozenset({"device_mac", "protocol_name", "anomaly_label"}),
        numeric_parse_success_threshold=0.75,
        device_column="src_mac",
    )

    assert set(result.kept_feature_columns) == {"ttl", "payload_entropy"}
    assert set(result.device_features.keys()) == {ClientId("dev1"), ClientId("dev2")}
    # dev2 had one row with a corrupt ttl value; that row is dropped, not coerced.
    assert result.device_features[ClientId("dev1")].shape == (2, 2)
    assert result.device_features[ClientId("dev2")].shape == (2, 2)
    assert result.rows_dropped_unparseable == 1


def test_read_packet_csv_missing_device_column_raises(tmp_path: Path) -> None:
    path = _write_csv(tmp_path)
    with pytest.raises(ValueError, match="device column"):
        read_packet_csv(
            path,
            excluded_features=frozenset(),
            numeric_parse_success_threshold=0.999,
            device_column="nonexistent_column",
        )


def test_read_packet_csv_no_surviving_columns_raises(tmp_path: Path) -> None:
    path = _write_csv(tmp_path)
    with pytest.raises(ValueError, match="no feature columns survived"):
        read_packet_csv(
            path,
            excluded_features=frozenset(
                {"device_mac", "protocol_name", "ttl", "payload_entropy", "anomaly_label"}
            ),
            numeric_parse_success_threshold=0.999,
            device_column="src_mac",
        )
