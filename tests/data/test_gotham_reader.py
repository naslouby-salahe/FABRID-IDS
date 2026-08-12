from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fabrid.data.gotham_reader import read_processed_csv
from fabrid.evaluation.record_level import AttackSubtype, ClientId


def _write_csv(tmp_path: Path) -> Path:
    frame = pd.DataFrame(
        {
            "eth.src": ["aa:bb", "aa:bb", "aa:bb", "cc:dd", "cc:dd"],
            "eth.dst": ["ff:ff"] * 5,
            "frame.time": ["t1", "t2", "t3", "t4", "t5"],
            "frame.protocols": ["eth:ip:tcp"] * 5,
            "ip.ttl": [64, 64, 63, "corrupt", 63],
            "tcp.window_size_value": [100.0, 101.0, 50.0, 51.0, 52.0],
            "label": ["Benign", "Benign", "TCP Scan", "Benign", "Telnet Brute Force"],
        }
    )
    path = tmp_path / "sample.csv"
    frame.to_csv(path, index=False)
    return path


def test_read_processed_csv_splits_benign_and_attack_by_client(tmp_path: Path) -> None:
    path = _write_csv(tmp_path)
    result = read_processed_csv(
        path,
        excluded_features=frozenset({"eth.dst", "frame.protocols"}),
        numeric_parse_success_threshold=0.75,
        nrows=None,
    )
    assert set(result.kept_feature_columns) == {"ip.ttl", "tcp.window_size_value"}
    assert result.rows_dropped_unparseable == 1

    assert ClientId("aa:bb") in result.benign_features_by_client
    assert result.benign_features_by_client[ClientId("aa:bb")].shape == (2, 2)

    assert ClientId("cc:dd") in result.attack_features_by_client_and_subtype
    cc_attacks = result.attack_features_by_client_and_subtype[ClientId("cc:dd")]
    assert set(cc_attacks) == {AttackSubtype("Telnet Brute Force")}

    aa_attacks = result.attack_features_by_client_and_subtype.get(ClientId("aa:bb"), {})
    assert set(aa_attacks) == {AttackSubtype("TCP Scan")}


def test_read_processed_csv_missing_required_column_raises(tmp_path: Path) -> None:
    path = _write_csv(tmp_path)
    with pytest.raises(ValueError, match="not found"):
        read_processed_csv(
            path,
            excluded_features=frozenset(),
            numeric_parse_success_threshold=0.999,
            device_column="nonexistent",
        )
