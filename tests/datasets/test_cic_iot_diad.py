from __future__ import annotations

import csv
from pathlib import Path

import pytest

from fabrid.config import (
    CicIotDiadDatasetConfig,
    DatasetId,
    EventCriterionId,
    ExternalEligibilityConfig,
    GateStatus,
    ReplicationEvidenceLevel,
)
from fabrid.datasets.cic_iot_diad import (
    assess_external_evidence,
    device_eligibility,
    device_row_census,
    global_feature_manifest,
    iter_collection_csvs,
    prepare_external_federation,
    read_packet_csv,
)
from fabrid.datasets.registry import ClientPopulation, plan_device_splits
from fabrid.errors import DatasetError
from fabrid.experiments.event_level import assess_event_data_gate
from fabrid.validation.datasets import build_split_manifest
from tests.support import event_evidence, production_application


def _layout() -> CicIotDiadDatasetConfig:
    return production_application().datasets.cic_iot_diad


def _eligibility() -> ExternalEligibilityConfig:
    return production_application().external_replication.eligibility


def _write_packet_csv(
    path: Path,
    feature_columns: tuple[str, ...],
    rows: list[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    layout = _layout()
    columns = [layout.device_column, layout.target_column, *feature_columns]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    layout.device_column: row.get("device", ""),
                    layout.target_column: row.get("target", ""),
                    **{column: row.get(column, "1.0") for column in feature_columns},
                }
            )


def _attack_capture_dir(collection_root: Path) -> Path:
    return collection_root / "flood" / "syn"


def test_global_feature_manifest_pools_parse_success_across_captures(tmp_path: Path) -> None:
    layout = _layout()
    collection_root = tmp_path / layout.packet_collection_directory
    benign_root = collection_root / layout.benign_category
    attack_root = _attack_capture_dir(collection_root)
    benign_root.mkdir(parents=True)
    attack_root.mkdir(parents=True)
    _write_packet_csv(
        benign_root / "aa.csv",
        feature_columns=("f1", "f2", "garbage"),
        rows=[
            {"device": "aa", "f1": 1.0, "f2": 1.0, "garbage": "x"},
            {"device": "aa", "f1": 2.0, "f2": 2.0, "garbage": "y"},
        ],
    )
    _write_packet_csv(
        attack_root / "aa.csv",
        feature_columns=("f1", "f2", "garbage"),
        rows=[
            {"target": "aa", "f1": 1.0, "f2": "", "garbage": "x"},
            {"target": "aa", "f1": 2.0, "f2": "", "garbage": "y"},
        ],
    )
    manifest = global_feature_manifest(
        iter_collection_csvs(tmp_path, layout), layout, numeric_parse_threshold=0.5
    )
    assert manifest == ("f1", "f2")
    assert "garbage" not in manifest
    assert layout.device_column not in manifest
    assert layout.target_column not in manifest


def test_read_packet_csv_groups_devices_and_drops_incomplete_rows(tmp_path: Path) -> None:
    path = tmp_path / "capture.csv"
    _write_packet_csv(
        path,
        feature_columns=("f1", "f2"),
        rows=[
            {"device": "aa", "f1": 1.0, "f2": 1.0},
            {"device": "aa", "f1": 2.0, "f2": 2.0},
            {"device": "bb", "f1": "not-a-number", "f2": 3.0},
            {"device": "bb", "f1": 4.0, "f2": 4.0},
        ],
    )
    result = read_packet_csv(path, _layout(), manifest=("f1", "f2"))
    assert result.dropped_rows == 1
    by_client = {device.client_id: device for device in result.devices}
    assert set(by_client) == {"aa", "bb"}
    assert by_client["aa"].features.row_count == 2
    assert by_client["bb"].features.row_count == 1
    assert result.kept_columns == ("f1", "f2")


def test_read_packet_csv_rejects_missing_device_column(tmp_path: Path) -> None:
    layout = _layout()
    path = tmp_path / "capture.csv"
    _write_packet_csv(path, feature_columns=("f1",), rows=[{"device": "aa", "f1": 1.0}])
    raw = path.read_text().replace(layout.device_column, "other_column")
    path.write_text(raw)
    with pytest.raises(ValueError):
        read_packet_csv(path, layout, manifest=("f1",))


def test_device_eligibility() -> None:
    eligibility = _eligibility()
    assert device_eligibility(
        benign_rows=eligibility.minimum_benign_rows,
        attack_rows=eligibility.minimum_attack_rows,
        minimum_benign=eligibility.minimum_benign_rows,
        minimum_attack=eligibility.minimum_attack_rows,
    )
    assert not device_eligibility(
        benign_rows=eligibility.minimum_benign_rows - 1,
        attack_rows=eligibility.minimum_attack_rows,
        minimum_benign=eligibility.minimum_benign_rows,
        minimum_attack=eligibility.minimum_attack_rows,
    )
    assert not device_eligibility(
        benign_rows=eligibility.minimum_benign_rows,
        attack_rows=eligibility.minimum_attack_rows - 1,
        minimum_benign=eligibility.minimum_benign_rows,
        minimum_attack=eligibility.minimum_attack_rows,
    )


def test_device_row_census_and_evidence_level(tmp_path: Path) -> None:
    layout = _layout()
    eligibility = _eligibility()
    collection_root = tmp_path / layout.packet_collection_directory
    benign_root = collection_root / layout.benign_category
    attack_root = _attack_capture_dir(collection_root)
    benign_root.mkdir(parents=True)
    attack_root.mkdir(parents=True)
    for device, count in (("aa", 12_000), ("bb", 11_000)):
        _write_packet_csv(
            benign_root / f"{device}.csv",
            feature_columns=("f1",),
            rows=[{"device": device, "f1": 1.0} for _ in range(count)],
        )
    _write_packet_csv(
        attack_root / "aa.csv",
        feature_columns=("f1",),
        rows=[{"target": "aa", "f1": 1.0} for _ in range(1_500)],
    )
    census = device_row_census(tmp_path, layout)
    assert {device.client_id for device in census.devices} == {"aa", "bb"}
    assessment = assess_external_evidence(
        census,
        minimum_benign_rows=eligibility.minimum_benign_rows,
        minimum_attack_rows=eligibility.minimum_attack_rows,
        minimum_confirmatory_clients=1,
    )
    assert assessment.evidence_level is ReplicationEvidenceLevel.CONFIRMATORY
    assert assessment.qualifying_clients == 1


def test_parens_device_identity_survives_census_federation_and_split_manifest(
    tmp_path: Path,
) -> None:
    layout = _layout()
    external = production_application().external_replication
    collection_root = tmp_path / layout.packet_collection_directory
    benign_root = collection_root / layout.benign_category
    attack_root = _attack_capture_dir(collection_root)
    benign_root.mkdir(parents=True)
    attack_root.mkdir(parents=True)
    _write_packet_csv(
        benign_root / "parens.csv",
        feature_columns=("f1",),
        rows=[{"device": "Harman Kardon (Ampak Technology)", "f1": 1.0} for _ in range(100)],
    )
    _write_packet_csv(
        attack_root / "parens.csv",
        feature_columns=("f1",),
        rows=[{"target": "Harman Kardon (Ampak Technology)", "f1": 1.0} for _ in range(50)],
    )
    census = device_row_census(tmp_path, layout)
    assert tuple(device.client_id for device in census.devices) == (
        "harman kardon _ampak technology_",
    )
    federation = prepare_external_federation(
        tmp_path,
        layout,
        numeric_parse_threshold=0.5,
        eligible_devices=tuple(device.client_id for device in census.devices),
    )
    assert tuple(device.client_id for device in federation.devices) == (
        "harman kardon _ampak technology_",
    )
    population = ClientPopulation(tuple(device.client_id for device in federation.devices))
    plans = tuple(
        plan_device_splits(device, external.benign_splits, external.attack_split)
        for device in federation.devices
    )
    manifest = build_split_manifest(external.dataset_id, population, plans)
    assert tuple(client.client_id for client in manifest.clients) == population.clients


def test_prepare_external_federation_rejects_capture_missing_manifest_feature(
    tmp_path: Path,
) -> None:
    layout = _layout()
    collection_root = tmp_path / layout.packet_collection_directory
    benign_root = collection_root / layout.benign_category
    attack_root = _attack_capture_dir(collection_root)
    benign_root.mkdir(parents=True)
    attack_root.mkdir(parents=True)
    _write_packet_csv(
        benign_root / "aa.csv",
        feature_columns=("f1", "f2"),
        rows=[{"device": "aa", "f1": 1.0, "f2": 1.0} for _ in range(100)],
    )
    _write_packet_csv(
        attack_root / "aa.csv",
        feature_columns=("f1",),
        rows=[{"target": "aa", "f1": 1.0} for _ in range(10)],
    )
    with pytest.raises(DatasetError, match="lacks frozen manifest feature"):
        prepare_external_federation(
            tmp_path,
            layout,
            numeric_parse_threshold=0.5,
            eligible_devices=("aa",),
        )


def test_prepare_external_federation_restricts_to_eligible_devices(tmp_path: Path) -> None:
    layout = _layout()
    collection_root = tmp_path / layout.packet_collection_directory
    benign_root = collection_root / layout.benign_category
    attack_root = _attack_capture_dir(collection_root)
    benign_root.mkdir(parents=True)
    attack_root.mkdir(parents=True)
    for device in ("aa", "bb"):
        _write_packet_csv(
            benign_root / f"{device}.csv",
            feature_columns=("f1",),
            rows=[{"device": device, "f1": 1.0} for _ in range(20)],
        )
        _write_packet_csv(
            attack_root / f"{device}.csv",
            feature_columns=("f1",),
            rows=[{"target": device, "f1": 1.0} for _ in range(10)],
        )
    federation = prepare_external_federation(
        tmp_path,
        layout,
        numeric_parse_threshold=0.5,
        eligible_devices=("aa",),
    )
    assert tuple(device.client_id for device in federation.devices) == ("aa",)


def test_prepare_external_federation_rejects_empty_eligibility(tmp_path: Path) -> None:
    layout = _layout()
    with pytest.raises(DatasetError, match="requires eligible devices"):
        prepare_external_federation(
            tmp_path,
            layout,
            numeric_parse_threshold=0.5,
            eligible_devices=(),
        )


def test_cic_iot_diad_event_gate_fails_on_identity_only(tmp_path: Path) -> None:
    layout = _layout()
    benign_root = tmp_path / layout.packet_collection_directory / layout.benign_category
    benign_root.mkdir(parents=True)
    _write_packet_csv(
        benign_root / "aa.csv",
        feature_columns=("f1",),
        rows=[{"device": "aa", "f1": 1.0} for _ in range(50)],
    )
    census = device_row_census(tmp_path, layout)
    assert census.devices
    evidence = event_evidence(failing=EventCriterionId.PACKET_TIMESTAMP)
    assessment = assess_event_data_gate(DatasetId.CIC_IOT_DIAD, evidence)
    assert assessment.status is GateStatus.FAIL
