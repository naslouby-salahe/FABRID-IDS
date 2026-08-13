from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fabrid.datasets.cic_iot_diad.reader import (
    FeatureColumns,
    read_packet_csv,
    select_numeric_feature_columns,
)
from fabrid.domain.enums import FeatureColumnStatus
from fabrid.domain.identifiers import ClientId, ColumnName
from fabrid.domain.values import Probability, RowCount


def _write_csv(tmp_path: Path) -> Path:
    frame = pd.DataFrame(
        {
            "src_mac": ("dev1", "dev1", "dev2", "dev2", "dev2"),
            "device_mac": ("dev1", "dev1", "dev2", "dev2", "dev2"),
            "protocol_name": ("TLS", "TLS", "none", "none", "TLS"),
            "ttl": (64, 64, 63, "corrupt", 63),
            "payload_entropy": (3.5, 3.6, 1.2, 1.3, 1.1),
            "anomaly_label": (0, 0, 1, 1, 1),
        }
    )
    path = tmp_path / "sample.csv"
    frame.to_csv(path, index=False)
    return path


def test_numeric_selection_excludes_identity_and_configured_columns() -> None:
    frame = pd.DataFrame(
        {
            "src_mac": ("dev1", "dev2"),
            "device_mac": ("dev1", "dev2"),
            "ttl": (64, 63),
        }
    )

    reports = select_numeric_feature_columns(
        frame=frame,
        excluded_features=FeatureColumns((ColumnName("device_mac"),)),
        threshold=Probability(0.999),
        device_column=ColumnName("src_mac"),
    )

    assert tuple(
        report.column
        for report in reports
        if report.status is FeatureColumnStatus.KEPT
    ) == (ColumnName("ttl"),)
    assert tuple(
        report.column
        for report in reports
        if report.status is FeatureColumnStatus.EXCLUDED
    ) == (ColumnName("src_mac"), ColumnName("device_mac"))


def test_numeric_selection_rejects_low_parse_success() -> None:
    frame = pd.DataFrame({"mostly_bad": ("1", "x", "y", "z", "2")})

    report = select_numeric_feature_columns(
        frame=frame,
        excluded_features=FeatureColumns(()),
        threshold=Probability(0.999),
        device_column=ColumnName("nonexistent"),
    )[0]

    assert report.status is FeatureColumnStatus.PARSE_REJECTED
    assert report.parse_success == Probability(0.4)


def test_packet_ingest_groups_clients_and_drops_unparseable_rows(tmp_path: Path) -> None:
    result = read_packet_csv(
        path=_write_csv(tmp_path),
        excluded_features=FeatureColumns(
            (
                ColumnName("device_mac"),
                ColumnName("protocol_name"),
                ColumnName("anomaly_label"),
            )
        ),
        numeric_parse_threshold=Probability(0.75),
        device_column=ColumnName("src_mac"),
    )

    assert result.kept_columns.values == (
        ColumnName("ttl"),
        ColumnName("payload_entropy"),
    )
    assert tuple(device.client_id for device in result.devices) == (
        ClientId("dev1"),
        ClientId("dev2"),
    )
    assert tuple(device.features.row_count for device in result.devices) == (
        RowCount(2),
        RowCount(2),
    )
    assert result.dropped_rows == RowCount(1)


def test_packet_ingest_rejects_missing_device_column(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="device column"):
        read_packet_csv(
            path=_write_csv(tmp_path),
            excluded_features=FeatureColumns(()),
            numeric_parse_threshold=Probability(0.999),
            device_column=ColumnName("nonexistent_column"),
        )
