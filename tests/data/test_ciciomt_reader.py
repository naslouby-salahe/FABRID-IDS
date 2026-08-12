from __future__ import annotations

from pathlib import Path

import pytest

from fabrid.data.ciciomt_reader import (
    parse_attack_filename,
    read_attacks_directory,
    read_profiling_directory,
    session_id_from_profiling_filename,
)
from fabrid.evaluation.record_level import AttackSubtype, ClientId


def test_session_id_from_profiling_filename_strips_suffix() -> None:
    assert session_id_from_profiling_filename("Blink_Camera_LAN_MIC.pcap.csv") == ClientId(
        "Blink_Camera_LAN_MIC"
    )
    assert session_id_from_profiling_filename("Active.pcap.csv") == ClientId("Active")


def test_session_id_from_profiling_filename_rejects_wrong_suffix() -> None:
    with pytest.raises(ValueError, match="expected a"):
        session_id_from_profiling_filename("something.csv")


def test_parse_attack_filename_extracts_subtype_and_split() -> None:
    subtype, is_train = parse_attack_filename("MQTT-DoS-Publish_Flood_test.pcap.csv")
    assert subtype == AttackSubtype("MQTT-DoS-Publish_Flood")
    assert is_train is False

    subtype, is_train = parse_attack_filename("Benign_train.pcap.csv")
    assert subtype == AttackSubtype("Benign")
    assert is_train is True


def test_parse_attack_filename_rejects_unmatched_pattern() -> None:
    with pytest.raises(ValueError, match="does not match"):
        parse_attack_filename("no_split_suffix.pcap.csv")


def _write_csv(path: Path, rows: int, columns: int) -> None:
    header = ",".join(f"f{i}" for i in range(columns))
    lines = [header]
    for r in range(rows):
        lines.append(",".join(str(float(r + i)) for i in range(columns)))
    path.write_text("\n".join(lines) + "\n")


def test_read_profiling_directory_keys_by_filename_stem(tmp_path: Path) -> None:
    _write_csv(tmp_path / "DeviceA_LAN_MIC.pcap.csv", rows=3, columns=4)
    _write_csv(tmp_path / "Active.pcap.csv", rows=2, columns=4)

    sessions = read_profiling_directory(tmp_path)

    assert set(sessions) == {ClientId("DeviceA_LAN_MIC"), ClientId("Active")}
    assert sessions[ClientId("DeviceA_LAN_MIC")].shape == (3, 4)
    assert sessions[ClientId("Active")].shape == (2, 4)


def test_read_attacks_directory_recurses_and_parses_each_file(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    test_dir = tmp_path / "test"
    train_dir.mkdir()
    test_dir.mkdir()
    _write_csv(train_dir / "Benign_train.pcap.csv", rows=5, columns=3)
    _write_csv(test_dir / "MQTT-DoS_test.pcap.csv", rows=4, columns=3)

    files = read_attacks_directory(tmp_path)

    assert len(files) == 2
    by_subtype = {f.attack_subtype: f for f in files}
    assert by_subtype[AttackSubtype("Benign")].is_train_split is True
    assert by_subtype[AttackSubtype("Benign")].features.shape == (5, 3)
    assert by_subtype[AttackSubtype("MQTT-DoS")].is_train_split is False
    assert by_subtype[AttackSubtype("MQTT-DoS")].features.shape == (4, 3)
