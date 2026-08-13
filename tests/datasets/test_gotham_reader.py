from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fabrid.datasets.cic_iot_diad.reader import FeatureColumns
from fabrid.datasets.gotham.reader import read_processed_csv
from fabrid.domain.identifiers import AttackSubtypeId, ClientId, ColumnName
from fabrid.domain.values import Probability, RowCount


def _write_csv(tmp_path: Path) -> Path:
    frame = pd.DataFrame(
        {
            "eth.src": ("aa:bb", "aa:bb", "aa:bb", "cc:dd", "cc:dd"),
            "eth.dst": ("ff:ff",) * 5,
            "frame.time": ("t1", "t2", "t3", "t4", "t5"),
            "frame.protocols": ("eth:ip:tcp",) * 5,
            "ip.ttl": (64, 64, 63, "corrupt", 63),
            "tcp.window_size_value": (100.0, 101.0, 50.0, 51.0, 52.0),
            "label": (
                "Benign",
                "Benign",
                "TCP Scan",
                "Benign",
                "Telnet Brute Force",
            ),
        }
    )
    path = tmp_path / "sample.csv"
    frame.to_csv(path, index=False)
    return path


def test_processed_gotham_csv_splits_benign_and_attack_per_client(tmp_path: Path) -> None:
    result = read_processed_csv(
        path=_write_csv(tmp_path),
        excluded_features=FeatureColumns(
            (ColumnName("eth.dst"), ColumnName("frame.protocols"))
        ),
        numeric_parse_threshold=Probability(0.75),
    )

    assert result.kept_columns.values == (
        ColumnName("ip.ttl"),
        ColumnName("tcp.window_size_value"),
    )
    assert result.dropped_rows == RowCount(1)

    aa = next(device for device in result.devices if device.client_id == ClientId("aa:bb"))
    cc = next(device for device in result.devices if device.client_id == ClientId("cc:dd"))
    assert aa.benign.row_count == RowCount(2)
    assert aa.attack(AttackSubtypeId("TCP Scan")).features.row_count == RowCount(1)
    assert cc.attack(AttackSubtypeId("Telnet Brute Force")).features.row_count == RowCount(1)


def test_processed_gotham_csv_requires_client_column(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not found"):
        read_processed_csv(
            path=_write_csv(tmp_path),
            excluded_features=FeatureColumns(()),
            numeric_parse_threshold=Probability(0.999),
            device_column=ColumnName("nonexistent"),
        )
