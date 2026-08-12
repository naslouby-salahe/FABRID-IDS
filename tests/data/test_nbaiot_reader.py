from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fabrid.data.nbaiot_reader import read_device_directory
from fabrid.evaluation.record_level import AttackSubtype, ClientId


def _write_csv(path: Path, n_rows: int, n_features: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [f"feature_{i}" for i in range(n_features)]
    frame = pd.DataFrame(
        [[float(row * n_features + col) for col in range(n_features)] for row in range(n_rows)],
        columns=columns,
    )
    frame.to_csv(path, index=False)


def test_reads_benign_and_both_attack_families(tmp_path: Path) -> None:
    device_dir = tmp_path / "Danmini_Doorbell"
    _write_csv(device_dir / "benign_traffic.csv", n_rows=10)
    _write_csv(device_dir / "gafgyt_attacks" / "scan.csv", n_rows=5)
    _write_csv(device_dir / "gafgyt_attacks" / "combo.csv", n_rows=4)
    _write_csv(device_dir / "mirai_attacks" / "ack.csv", n_rows=6)

    data = read_device_directory(ClientId("1"), device_dir)

    assert data.benign_features.shape == (10, 3)
    assert data.attack_features_by_subtype[AttackSubtype("bashlite_scan")].shape == (5, 3)
    assert data.attack_features_by_subtype[AttackSubtype("bashlite_combo")].shape == (4, 3)
    assert data.attack_features_by_subtype[AttackSubtype("mirai_ack")].shape == (6, 3)
    assert AttackSubtype("mirai_udp") not in data.attack_features_by_subtype


def test_device_with_no_mirai_directory(tmp_path: Path) -> None:
    device_dir = tmp_path / "Ennio_Doorbell"
    _write_csv(device_dir / "benign_traffic.csv", n_rows=8)
    _write_csv(device_dir / "gafgyt_attacks" / "junk.csv", n_rows=3)

    data = read_device_directory(ClientId("2"), device_dir)

    assert data.benign_features.shape == (8, 3)
    assert set(data.attack_features_by_subtype.keys()) == {AttackSubtype("bashlite_junk")}


def test_missing_benign_file_raises(tmp_path: Path) -> None:
    device_dir = tmp_path / "Empty_Device"
    device_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        read_device_directory(ClientId("3"), device_dir)


def test_source_row_order_preserved(tmp_path: Path) -> None:
    device_dir = tmp_path / "Device"
    _write_csv(device_dir / "benign_traffic.csv", n_rows=5)
    data = read_device_directory(ClientId("1"), device_dir)
    # row i's first feature is i * n_features by construction -> strictly increasing.
    first_column = data.benign_features[:, 0]
    assert list(first_column) == sorted(first_column)
