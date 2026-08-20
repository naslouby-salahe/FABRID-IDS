from __future__ import annotations

from pathlib import Path

import pytest

from fabrid.config import EventCriterionId, EventGateConfig, GothamDatasetConfig, Label
from fabrid.datasets.gotham import (
    audit_gotham_event_provenance,
    load_gotham_numeric_features,
)
from fabrid.datasets.registry import EventProvenanceEvidence
from fabrid.errors import DatasetError
from tests.support import production_application


def _layout() -> GothamDatasetConfig:
    return production_application().datasets.gotham


def _event_gate() -> EventGateConfig:
    return production_application().event_level.event_gate


def write_gotham_capture(
    root: Path,
    name: str,
    rows: list[tuple[str, str, str]],
    *,
    feature_names: tuple[str, ...] = ("f1",),
    feature_rows: tuple[tuple[str, ...], ...] | None = None,
) -> None:
    layout = _layout()
    root.mkdir(parents=True, exist_ok=True)
    path = root / name
    extras = "," + ",".join(feature_names) if feature_names else ""
    with path.open("w") as handle:
        handle.write(
            f"{layout.device_column},{layout.label_column},{layout.timestamp_column}{extras}\n"
        )
        for index, (device, label, timestamp) in enumerate(rows):
            values = ("1.0",) * len(feature_names) if feature_rows is None else feature_rows[index]
            feature_text = "," + ",".join(values) if values else ""
            handle.write(f'{device},{label},"{timestamp}"{feature_text}\n')


def _audit(raw_root: Path) -> EventProvenanceEvidence:
    gate = _event_gate()
    return audit_gotham_event_provenance(
        raw_root,
        _layout(),
        gate.minimum_timestamp_parse_success,
        gate.maximum_out_of_order_fraction,
        gate.minimum_capture_seam_seconds,
    )


def test_audit_gotham_event_provenance_passes_with_capture_provenance(tmp_path: Path) -> None:
    layout = _layout()
    write_gotham_capture(
        tmp_path,
        "capture1.csv",
        [
            ("aa", layout.benign_label, "Jan 01, 2020 00:00:00.000000000 GMT"),
            ("aa", layout.benign_label, "Jan 01, 2020 00:00:01.000000000 GMT"),
            ("aa", "attack-x", "Jan 01, 2020 00:00:02.000000000 GMT"),
        ],
    )
    evidence = _audit(tmp_path)
    assert isinstance(evidence, EventProvenanceEvidence)
    assert evidence.passed(EventCriterionId.IMMUTABLE_CLIENT_ID)
    assert evidence.passed(EventCriterionId.PACKET_TIMESTAMP)
    assert evidence.passed(EventCriterionId.INTERVAL_PROVENANCE)
    assert evidence.passed(EventCriterionId.DETERMINISTIC_SCORE_ASSOCIATION)
    assert evidence.passed(EventCriterionId.WITHIN_CLIENT_ORDERING)
    assert evidence.passed(EventCriterionId.OBSERVATION_DURATION)
    assert evidence.passed(EventCriterionId.NON_OVERLAPPING_EVALUATION_PERIOD)


def test_audit_gotham_event_provenance_empty_tree(tmp_path: Path) -> None:
    evidence = _audit(tmp_path)
    assert not evidence.passed(EventCriterionId.PACKET_TIMESTAMP)
    assert not evidence.passed(EventCriterionId.WITHIN_CLIENT_ORDERING)
    assert not evidence.passed(EventCriterionId.DETERMINISTIC_SCORE_ASSOCIATION)


def test_audit_tolerates_capture_segment_seams_and_requires_separable_period(
    tmp_path: Path,
) -> None:
    layout = _layout()
    rows: list[tuple[str, str, str]] = [
        *(
            (
                "aa",
                "attack-x",
                f"Jan 18, 2020 21:{33 + index // 60:02d}:{index % 60:02d}.000000000 GMT",
            )
            for index in range(200)
        ),
        *(
            (
                "aa",
                layout.benign_label,
                f"Jan 14, 2020 18:{40 + index // 60:02d}:{index % 60:02d}.000000000 GMT",
            )
            for index in range(200)
        ),
    ]
    write_gotham_capture(tmp_path, "seamed.csv", rows)
    evidence = _audit(tmp_path)
    assert evidence.passed(EventCriterionId.WITHIN_CLIENT_ORDERING)
    assert evidence.passed(EventCriterionId.NON_OVERLAPPING_EVALUATION_PERIOD)
    assert evidence.passed(EventCriterionId.INTERVAL_PROVENANCE)


def test_audit_requires_benign_period_outside_the_attack_phase(tmp_path: Path) -> None:
    layout = _layout()
    rows: list[tuple[str, str, str]] = [
        ("aa", "attack-x", "Jan 18, 2020 21:33:16.000000000 GMT"),
        ("aa", layout.benign_label, "Jan 18, 2020 21:33:17.000000000 GMT"),
        ("aa", "attack-x", "Jan 18, 2020 21:33:18.000000000 GMT"),
    ]
    write_gotham_capture(tmp_path, "mixed.csv", rows)
    evidence = _audit(tmp_path)
    assert not evidence.passed(EventCriterionId.NON_OVERLAPPING_EVALUATION_PERIOD)
    assert evidence.passed(EventCriterionId.WITHIN_CLIENT_ORDERING)


def test_numeric_reader_excludes_layout_columns_and_keeps_original_rows(tmp_path: Path) -> None:
    layout = _layout()
    write_gotham_capture(
        tmp_path,
        "capture1.csv",
        [
            ("aa", layout.benign_label, "Jan 01, 2020 00:00:00.000000000 GMT"),
            ("aa", layout.benign_label, "Jan 01, 2020 00:00:01.000000000 GMT"),
            ("aa", "attack-x", "Jan 01, 2020 00:00:02.000000000 GMT"),
        ],
        feature_names=("f1", "f2"),
        feature_rows=(("1.0", "2.0"), ("3.0", "4.0"), ("5.0", "6.0")),
    )
    clients = load_gotham_numeric_features(tmp_path, layout)
    assert len(clients) == 1
    client = clients[0]
    assert client.client_id == "aa"
    assert client.features.feature_count == 2
    assert client.source_files == ("capture1.csv", "capture1.csv", "capture1.csv")
    assert client.source_rows == (0, 1, 2)
    assert client.labels == (Label.BENIGN, Label.BENIGN, Label.ATTACK)
    assert client.attack_subtypes == (None, None, "attack-x")
    assert client.features.values.tolist() == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]


def test_numeric_reader_drops_non_finite_rows_without_coercing(tmp_path: Path) -> None:
    layout = _layout()
    write_gotham_capture(
        tmp_path,
        "capture1.csv",
        [
            ("aa", layout.benign_label, "Jan 01, 2020 00:00:00.000000000 GMT"),
            ("aa", layout.benign_label, "Jan 01, 2020 00:00:01.000000000 GMT"),
            ("aa", "attack-x", "Jan 01, 2020 00:00:02.000000000 GMT"),
        ],
        feature_rows=(("1.0",), ("not-a-number",), ("7.0",)),
    )
    clients = load_gotham_numeric_features(tmp_path, layout)
    assert len(clients) == 1
    client = clients[0]
    assert client.source_rows == (0, 2)
    assert client.features.values.tolist() == [[1.0], [7.0]]


def test_numeric_reader_missing_files_raises(tmp_path: Path) -> None:
    layout = _layout()
    with pytest.raises(DatasetError, match="no processed CSV files"):
        load_gotham_numeric_features(tmp_path, layout)
