from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path

import numpy as np
import polars as pl

from fabrid.config import (
    AttackSubtypeId,
    ClientId,
    ColumnName,
    DurationSeconds,
    EnvironmentText,
    EventCriterionId,
    EventTimestamp,
    GothamDatasetConfig,
    Label,
    Probability,
    RowCount,
    SourceFileId,
    SourceRowIndex,
)
from fabrid.datasets.registry import (
    EventProvenanceEvidence,
    FeatureMatrix,
    event_criterion,
    unavailable_event_evidence,
)
from fabrid.errors import DatasetError

_ATTACK_SUBTYPE_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


@dataclass(frozen=True, slots=True)
class CaptureProvenance:
    source_file: SourceFileId
    device_count: RowCount
    timestamp_parse_rate: Probability
    associable_row_fraction: Probability
    has_attack_labels: bool
    out_of_order_fraction: Probability
    separable_benign_period: bool
    observation_duration: DurationSeconds | None


@dataclass(frozen=True, slots=True)
class GothamTimedRow:
    client_id: ClientId
    timestamp: EventTimestamp
    label: Label
    source_file: SourceFileId
    source_row: SourceRowIndex


@dataclass(frozen=True, slots=True)
class GothamNumericClient:
    client_id: ClientId
    source_files: tuple[SourceFileId, ...]
    source_rows: tuple[SourceRowIndex, ...]
    labels: tuple[Label, ...]
    attack_subtypes: tuple[AttackSubtypeId | None, ...]
    features: FeatureMatrix

    def __post_init__(self) -> None:
        row_count = self.features.row_count
        if not (
            len(self.source_files)
            == len(self.source_rows)
            == len(self.labels)
            == len(self.attack_subtypes)
            == row_count
        ):
            raise ValueError("gotham numeric client fields must share row count")
        for label, subtype in zip(self.labels, self.attack_subtypes, strict=True):
            if label is Label.ATTACK and subtype is None:
                raise ValueError("attack rows must carry an attack subtype")
            if label is Label.BENIGN and subtype is not None:
                raise ValueError("benign rows must not carry an attack subtype")


@dataclass(frozen=True, slots=True)
class _ParsedNumericRow:
    client_id: ClientId
    label: Label
    attack_subtype: AttackSubtypeId | None
    source_file: SourceFileId
    source_row: SourceRowIndex
    values: np.ndarray


def gotham_capture_paths(raw_root: Path, layout: GothamDatasetConfig) -> tuple[Path, ...]:
    return tuple(sorted(raw_root.rglob(layout.csv_glob)))


def _layout_columns(layout: GothamDatasetConfig) -> frozenset[ColumnName]:
    return frozenset((layout.device_column, layout.label_column, layout.timestamp_column))


def _read_gotham_csv(path: Path) -> pl.DataFrame:
    preview = pl.read_csv(path, n_rows=1, ignore_errors=True)
    schema_overrides = {column: pl.String for column in preview.columns}
    return pl.read_csv(path, schema_overrides=schema_overrides)


def _attack_subtype(raw_label: EnvironmentText) -> AttackSubtypeId:
    subtype = raw_label.strip().lower().replace("&", "-").replace(" ", "-")
    if _ATTACK_SUBTYPE_PATTERN.fullmatch(subtype) is None:
        raise DatasetError(f"gotham attack label {raw_label!r} is not a valid attack subtype")
    return subtype


def _feature_columns(
    columns: tuple[ColumnName, ...], layout: GothamDatasetConfig
) -> tuple[ColumnName, ...]:
    excluded = _layout_columns(layout) | frozenset(layout.excluded_features)
    return tuple(column for column in columns if column not in excluded)


def _parse_timestamps(column: pl.Series) -> pl.Series:
    normalized = column.str.replace(r"\s+GMT$", "", literal=False)
    normalized = normalized.str.replace(r"\.(\d{6})\d+$", ".$1", literal=False)
    return normalized.str.to_datetime(format="%b %d, %Y %H:%M:%S%.f", strict=False)


def _timestamp_epoch_seconds(timestamps: pl.Series) -> np.ndarray:
    epoch_nanoseconds = timestamps.cast(pl.Datetime("ns"), strict=False).to_numpy()
    return np.asarray(epoch_nanoseconds, dtype=np.float64) / 1.0e9


def _benign_period_separable(
    epochs: np.ndarray, labels: np.ndarray, benign_label: EnvironmentText
) -> bool:
    attack_times = epochs[labels != benign_label]
    benign_times = epochs[labels == benign_label]
    if attack_times.size == 0 or benign_times.size == 0:
        return True
    attack_lo = float(np.min(attack_times))
    attack_hi = float(np.max(attack_times))
    outside = benign_times[(benign_times < attack_lo) | (benign_times > attack_hi)]
    return bool(outside.size > 0)


def _device_out_of_order_fraction(
    device_times: np.ndarray, minimum_capture_seam_seconds: DurationSeconds
) -> Probability:
    device_times = device_times[~np.isnan(device_times)]
    if device_times.size < 2:
        return 0.0
    deltas = np.diff(device_times)
    jitter = deltas[(deltas < 0.0) & (deltas > -minimum_capture_seam_seconds)]
    return float(jitter.size) / float(device_times.size - 1)


def _audit_file(
    path: Path,
    layout: GothamDatasetConfig,
    minimum_capture_seam_seconds: DurationSeconds,
    row_limit: RowCount | None,
) -> CaptureProvenance:
    frame = pl.read_csv(
        path,
        columns=[layout.device_column, layout.label_column, layout.timestamp_column],
        n_rows=None if row_limit is None else row_limit,
    )
    device_column = frame[layout.device_column]
    label_column = frame[layout.label_column]
    timestamps = _parse_timestamps(frame[layout.timestamp_column])
    device_count = int(device_column.n_unique())
    parse_success = 1.0 - timestamps.null_count() / timestamps.len()
    attack_labels = tuple(
        sorted(
            {str(value) for value in label_column.to_numpy() if str(value) != layout.benign_label}
        )
    )
    device_values: np.ndarray = device_column.to_numpy()
    epochs = _timestamp_epoch_seconds(timestamps)
    label_values: np.ndarray = label_column.to_numpy()
    out_of_order = 0.0
    separable = True
    for raw_device in np.unique(device_values):
        device_mask = device_values == raw_device
        device_times = epochs[device_mask]
        device_labels = label_values[device_mask]
        out_of_order = max(
            out_of_order, _device_out_of_order_fraction(device_times, minimum_capture_seam_seconds)
        )
        if not _benign_period_separable(device_times, device_labels, layout.benign_label):
            separable = False
    finite_times = epochs[~np.isnan(epochs)]
    duration = (
        None if finite_times.size == 0 else float(np.max(finite_times) - np.min(finite_times))
    )
    device_present = np.array(
        [str(value).strip() != "" for value in device_values],
        dtype=np.bool_,
    )
    associable = int(np.count_nonzero(device_present & np.isfinite(epochs)))
    associable_fraction = 0.0 if device_values.size == 0 else associable / device_values.size
    return CaptureProvenance(
        source_file=path.name,
        device_count=device_count,
        timestamp_parse_rate=parse_success,
        associable_row_fraction=associable_fraction,
        has_attack_labels=bool(attack_labels),
        out_of_order_fraction=out_of_order,
        separable_benign_period=separable,
        observation_duration=duration,
    )


def audit_gotham_event_provenance(
    raw_root: Path,
    layout: GothamDatasetConfig,
    minimum_timestamp_parse_success: Probability,
    maximum_out_of_order_fraction: Probability,
    minimum_capture_seam_seconds: DurationSeconds,
    sample_rows: RowCount | None = None,
    maximum_files: RowCount | None = None,
) -> EventProvenanceEvidence:
    csv_paths = gotham_capture_paths(raw_root, layout)
    if maximum_files is not None:
        csv_paths = csv_paths[:maximum_files]
    if not csv_paths:
        return unavailable_event_evidence(
            "no processed CSV files available under the raw dataset root"
        )
    audits = tuple(
        _audit_file(path, layout, minimum_capture_seam_seconds, sample_rows) for path in csv_paths
    )
    device_counts = tuple(audit.device_count for audit in audits)
    parse_rates = tuple(audit.timestamp_parse_rate for audit in audits)
    associable_rates = tuple(audit.associable_row_fraction for audit in audits)
    durations = tuple(
        audit.observation_duration for audit in audits if audit.observation_duration is not None
    )
    out_of_order_rates = tuple(audit.out_of_order_fraction for audit in audits)
    attack_captures = tuple(audit.source_file for audit in audits if audit.has_attack_labels)
    return EventProvenanceEvidence(
        criteria=(
            event_criterion(
                EventCriterionId.IMMUTABLE_CLIENT_ID,
                all(count > 0 for count in device_counts),
                f"per-capture device counts {device_counts} from column {layout.device_column}; "
                "client partition is the device identity",
            ),
            event_criterion(
                EventCriterionId.PACKET_TIMESTAMP,
                all(rate >= minimum_timestamp_parse_success for rate in parse_rates),
                f"timestamp parse success rates {tuple(round(rate, 6) for rate in parse_rates)} "
                f"from column {layout.timestamp_column} "
                f"(minimum required {minimum_timestamp_parse_success})",
            ),
            event_criterion(
                EventCriterionId.INTERVAL_PROVENANCE,
                bool(attack_captures),
                f"attack-label interval provenance in {len(attack_captures)} capture(s); "
                "benign-only captures are the benign baseline",
            ),
            event_criterion(
                EventCriterionId.DETERMINISTIC_SCORE_ASSOCIATION,
                all(rate >= minimum_timestamp_parse_success for rate in associable_rates),
                "fraction of rows with a device identity and a parsed timestamp "
                f"{tuple(round(rate, 6) for rate in associable_rates)}; "
                "those rows are uniquely addressable by source file and row index",
            ),
            event_criterion(
                EventCriterionId.WITHIN_CLIENT_ORDERING,
                all(rate < maximum_out_of_order_fraction for rate in out_of_order_rates),
                "per-device out-of-order fractions "
                f"{tuple(round(rate, 6) for rate in out_of_order_rates)} (maximum allowed "
                f"{maximum_out_of_order_fraction}); processed captures are segment concatenations, "
                "so capture-boundary seams are expected",
            ),
            event_criterion(
                EventCriterionId.OBSERVATION_DURATION,
                bool(durations) and all(duration > 0.0 for duration in durations),
                "per-capture observation durations in seconds: "
                f"{tuple(round(value, 3) for value in durations)}",
            ),
            event_criterion(
                EventCriterionId.NON_OVERLAPPING_EVALUATION_PERIOD,
                all(audit.separable_benign_period for audit in audits),
                "every attack-capable device has benign rows strictly outside the attack phase, "
                "so the final evaluation period is separable from the attacks",
            ),
        )
    )


def load_gotham_timed_rows(
    raw_root: Path,
    layout: GothamDatasetConfig,
) -> tuple[GothamTimedRow, ...]:
    rows: list[GothamTimedRow] = []
    for path in gotham_capture_paths(raw_root, layout):
        frame = pl.read_csv(
            path,
            columns=[layout.device_column, layout.label_column, layout.timestamp_column],
        )
        timestamps = _parse_timestamps(frame[layout.timestamp_column])
        epochs = _timestamp_epoch_seconds(timestamps)
        devices = frame[layout.device_column].to_numpy()
        labels = frame[layout.label_column].to_numpy()
        for source_row, (device, epoch, raw_label) in enumerate(
            zip(devices, epochs, labels, strict=True)
        ):
            if not np.isfinite(epoch) or str(device).strip() == "":
                continue
            rows.append(
                GothamTimedRow(
                    client_id=str(device),
                    timestamp=float(epoch),
                    label=(Label.BENIGN if str(raw_label) == layout.benign_label else Label.ATTACK),
                    source_file=path.name,
                    source_row=source_row,
                )
            )
    return tuple(rows)


def _require_feature_columns(
    path: Path,
    frame: pl.DataFrame,
    layout: GothamDatasetConfig,
    expected: tuple[ColumnName, ...] | None,
) -> tuple[ColumnName, ...]:
    missing = tuple(column for column in _layout_columns(layout) if column not in frame.columns)
    if missing:
        raise DatasetError(
            f"gotham capture {path.name} is missing required columns {missing}",
            path=path,
        )
    columns = _feature_columns(tuple(frame.columns), layout)
    if not columns:
        raise DatasetError(
            f"gotham capture {path.name} has no numeric feature columns",
            path=path,
        )
    if expected is not None and columns != expected:
        raise DatasetError(
            f"gotham capture {path.name} feature columns {columns} differ from {expected}",
            path=path,
        )
    return columns


def _parsed_rows_from_capture(
    path: Path,
    frame: pl.DataFrame,
    layout: GothamDatasetConfig,
    columns: tuple[ColumnName, ...],
) -> tuple[_ParsedNumericRow, ...]:
    numeric = frame.select(
        pl.col(column).fill_null("0").cast(pl.Float64, strict=False) for column in columns
    )
    values = np.asarray(numeric.to_numpy(), dtype=np.float64)
    devices = tuple(str(item).strip() for item in frame[layout.device_column].to_list())
    labels = tuple(str(item) for item in frame[layout.label_column].to_list())
    parsed: list[_ParsedNumericRow] = []
    for source_row in range(values.shape[0]):
        client_id = devices[source_row]
        if client_id == "" or not bool(np.all(np.isfinite(values[source_row]))):
            continue
        raw_label = labels[source_row]
        is_benign = raw_label == layout.benign_label
        parsed.append(
            _ParsedNumericRow(
                client_id=client_id,
                label=Label.BENIGN if is_benign else Label.ATTACK,
                attack_subtype=None if is_benign else _attack_subtype(raw_label),
                source_file=path.name,
                source_row=source_row,
                values=values[source_row],
            )
        )
    return tuple(parsed)


def _clients_from_parsed_rows(
    parsed: tuple[_ParsedNumericRow, ...],
) -> tuple[GothamNumericClient, ...]:
    ordered = sorted(parsed, key=lambda row: (row.client_id, row.source_file, row.source_row))
    clients: list[GothamNumericClient] = []
    for client_id, group in groupby(ordered, key=lambda row: row.client_id):
        rows = tuple(group)
        clients.append(
            GothamNumericClient(
                client_id=client_id,
                source_files=tuple(row.source_file for row in rows),
                source_rows=tuple(row.source_row for row in rows),
                labels=tuple(row.label for row in rows),
                attack_subtypes=tuple(row.attack_subtype for row in rows),
                features=FeatureMatrix(np.stack(tuple(row.values for row in rows))),
            )
        )
    return tuple(clients)


def load_gotham_numeric_features(
    raw_root: Path,
    layout: GothamDatasetConfig,
) -> tuple[GothamNumericClient, ...]:
    csv_paths = gotham_capture_paths(raw_root, layout)
    if not csv_paths:
        raise DatasetError(
            "no processed CSV files available under the raw dataset root",
            path=raw_root,
        )
    parsed: list[_ParsedNumericRow] = []
    feature_columns: tuple[ColumnName, ...] | None = None
    for path in csv_paths:
        frame = _read_gotham_csv(path)
        feature_columns = _require_feature_columns(path, frame, layout, feature_columns)
        parsed.extend(_parsed_rows_from_capture(path, frame, layout, feature_columns))
    if not parsed:
        raise DatasetError(
            "no finite numeric Gotham rows remain after dropping non-finite feature rows",
            path=raw_root,
        )
    return _clients_from_parsed_rows(tuple(parsed))
