from __future__ import annotations

from pathlib import Path

import pytest

from fabrid.datasets.ciciomt.reader import (
    parse_attack_filename,
    read_attacks_directory,
    read_profiling_directory,
    session_id_from_profiling_filename,
)
from fabrid.domain.enums import SourceSplit
from fabrid.domain.identifiers import AttackSubtypeId, ClientId, SourceFileId
from fabrid.domain.values import FeatureCount, RowCount


def test_session_id_is_derived_from_profiling_filename() -> None:
    assert session_id_from_profiling_filename(
        SourceFileId("Blink_Camera_LAN_MIC.pcap.csv")
    ) == ClientId("Blink_Camera_LAN_MIC")
    assert session_id_from_profiling_filename(SourceFileId("Active.pcap.csv")) == ClientId(
        "Active"
    )


def test_session_id_rejects_wrong_suffix() -> None:
    with pytest.raises(ValueError, match="expected a"):
        session_id_from_profiling_filename(SourceFileId("something.csv"))


def test_attack_filename_parses_subtype_and_split_enum() -> None:
    subtype, split = parse_attack_filename(
        SourceFileId("MQTT-DoS-Publish_Flood_test.pcap.csv")
    )
    assert subtype == AttackSubtypeId("MQTT-DoS-Publish_Flood")
    assert split is SourceSplit.TEST

    subtype, split = parse_attack_filename(SourceFileId("Benign_train.pcap.csv"))
    assert subtype == AttackSubtypeId("Benign")
    assert split is SourceSplit.TRAIN


def test_attack_filename_rejects_unmatched_pattern() -> None:
    with pytest.raises(ValueError, match="does not match"):
        parse_attack_filename(SourceFileId("no_split_suffix.pcap.csv"))


def _write_csv(path: Path, rows: RowCount, columns: FeatureCount) -> None:
    header = ",".join(f"f{index}" for index in range(columns.value))
    lines = [header]
    for row in range(rows.value):
        lines.append(",".join(str(float(row + index)) for index in range(columns.value)))
    path.write_text("\n".join(lines) + "\n")


def test_profiling_directory_returns_typed_sessions(tmp_path: Path) -> None:
    _write_csv(tmp_path / "DeviceA_LAN_MIC.pcap.csv", RowCount(3), FeatureCount(4))
    _write_csv(tmp_path / "Active.pcap.csv", RowCount(2), FeatureCount(4))

    sessions = read_profiling_directory(tmp_path)

    assert tuple(session.client_id for session in sessions.sessions) == (
        ClientId("Active"),
        ClientId("DeviceA_LAN_MIC"),
    )
    assert tuple(session.features.row_count for session in sessions.sessions) == (
        RowCount(2),
        RowCount(3),
    )


def test_attack_directory_recurses_and_preserves_source_split(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    test_dir = tmp_path / "test"
    train_dir.mkdir()
    test_dir.mkdir()
    _write_csv(train_dir / "Benign_train.pcap.csv", RowCount(5), FeatureCount(3))
    _write_csv(test_dir / "MQTT-DoS_test.pcap.csv", RowCount(4), FeatureCount(3))

    files = read_attacks_directory(tmp_path)

    benign = next(file for file in files if file.subtype == AttackSubtypeId("Benign"))
    attack = next(file for file in files if file.subtype == AttackSubtypeId("MQTT-DoS"))
    assert benign.source_split is SourceSplit.TRAIN
    assert benign.features.row_count == RowCount(5)
    assert attack.source_split is SourceSplit.TEST
    assert attack.features.row_count == RowCount(4)
