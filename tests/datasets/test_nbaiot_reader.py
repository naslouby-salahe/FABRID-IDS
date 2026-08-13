from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fabrid.datasets.nbaiot.reader import read_device_directory
from fabrid.domain.identifiers import AttackSubtypeId, ClientId
from fabrid.domain.values import FeatureCount, RowCount


def _write_csv(path: Path, rows: RowCount, features: FeatureCount = FeatureCount(3)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = tuple(f"feature_{index}" for index in range(features.value))
    frame = pd.DataFrame(
        tuple(
            tuple(float(row * features.value + column) for column in range(features.value))
            for row in range(rows.value)
        ),
        columns=columns,
    )
    frame.to_csv(path, index=False)


def test_reads_benign_and_available_attack_families(tmp_path: Path) -> None:
    device_dir = tmp_path / "Danmini_Doorbell"
    _write_csv(device_dir / "benign_traffic.csv", RowCount(10))
    _write_csv(device_dir / "gafgyt_attacks" / "scan.csv", RowCount(5))
    _write_csv(device_dir / "gafgyt_attacks" / "combo.csv", RowCount(4))
    _write_csv(device_dir / "mirai_attacks" / "ack.csv", RowCount(6))

    dataset = read_device_directory(ClientId("Danmini_Doorbell"), device_dir)

    assert dataset.benign.row_count == RowCount(10)
    assert dataset.benign.feature_count == FeatureCount(3)
    assert dataset.attack(AttackSubtypeId("bashlite_scan")).features.row_count == RowCount(5)
    assert dataset.attack(AttackSubtypeId("bashlite_combo")).features.row_count == RowCount(4)
    assert dataset.attack(AttackSubtypeId("mirai_ack")).features.row_count == RowCount(6)
    with pytest.raises(KeyError):
        dataset.attack(AttackSubtypeId("mirai_udp"))


def test_device_without_mirai_keeps_only_present_attack_blocks(tmp_path: Path) -> None:
    device_dir = tmp_path / "Ennio_Doorbell"
    _write_csv(device_dir / "benign_traffic.csv", RowCount(8))
    _write_csv(device_dir / "gafgyt_attacks" / "junk.csv", RowCount(3))

    dataset = read_device_directory(ClientId("Ennio_Doorbell"), device_dir)

    assert tuple(block.subtype for block in dataset.attacks) == (
        AttackSubtypeId("bashlite_junk"),
    )


def test_missing_benign_file_is_rejected(tmp_path: Path) -> None:
    device_dir = tmp_path / "Empty_Device"
    device_dir.mkdir()

    with pytest.raises(FileNotFoundError):
        read_device_directory(ClientId("Empty_Device"), device_dir)


def test_benign_source_order_is_preserved(tmp_path: Path) -> None:
    device_dir = tmp_path / "Device"
    _write_csv(device_dir / "benign_traffic.csv", RowCount(5))

    dataset = read_device_directory(ClientId("Device"), device_dir)
    first_column = dataset.benign.values[:, 0]

    assert tuple(first_column) == tuple(sorted(first_column))
