from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from fabrid.config import (
    EXTERNAL_SOURCE_FILE,
    ZERO_ROW_COUNT,
    AttackSubtypeId,
    CicIotDiadDatasetConfig,
    ClientId,
    ColumnName,
    EnvironmentText,
    Label,
    Probability,
    ReplicationEvidenceLevel,
    RowCount,
)
from fabrid.datasets.registry import AttackFeatureBlock, DeviceDataset, FeatureMatrix
from fabrid.errors import DatasetError


@dataclass(frozen=True, slots=True)
class DeviceFeatures:
    client_id: ClientId
    features: FeatureMatrix


@dataclass(frozen=True, slots=True)
class PacketCsvIngestResult:
    devices: tuple[DeviceFeatures, ...]
    kept_columns: tuple[ColumnName, ...]
    dropped_rows: RowCount

    def __post_init__(self) -> None:
        client_ids = tuple(device.client_id for device in self.devices)
        if len(set(client_ids)) != len(client_ids):
            raise ValueError("packet CSV ingestion contains duplicate clients")


@dataclass(frozen=True, slots=True)
class AttackSubtypeCount:
    subtype: AttackSubtypeId
    rows: RowCount


@dataclass(frozen=True, slots=True)
class ExternalDeviceCounts:
    client_id: ClientId
    benign_rows: RowCount
    attacks: tuple[AttackSubtypeCount, ...]


@dataclass(frozen=True, slots=True)
class ExternalCensus:
    devices: tuple[ExternalDeviceCounts, ...]

    def __post_init__(self) -> None:
        client_ids = tuple(device.client_id for device in self.devices)
        if len(set(client_ids)) != len(client_ids):
            raise ValueError("external census contains duplicate clients")


@dataclass(frozen=True, slots=True)
class ExternalEvidenceAssessment:
    qualifying_clients: RowCount
    evidence_level: ReplicationEvidenceLevel
    eligible_devices: tuple[ClientId, ...]


@dataclass(frozen=True, slots=True)
class ExternalFederation:
    devices: tuple[DeviceDataset, ...]
    kept_columns: tuple[ColumnName, ...]
    dropped_rows: RowCount

    def __post_init__(self) -> None:
        client_ids = tuple(device.client_id for device in self.devices)
        if len(set(client_ids)) != len(client_ids):
            raise ValueError("external federation contains duplicate clients")


@dataclass(frozen=True, slots=True)
class DeviceFeatureParts:
    client_id: ClientId
    parts: tuple[FeatureMatrix, ...]


@dataclass(frozen=True, slots=True)
class DeviceAttackParts:
    client_id: ClientId
    blocks: tuple[AttackFeatureBlock, ...]


@dataclass(frozen=True, slots=True)
class CollectionCsv:
    path: Path
    label: Label
    attack_subtype: AttackSubtypeId | None

    def __post_init__(self) -> None:
        if self.label is Label.BENIGN:
            if self.attack_subtype is not None:
                raise ValueError("benign collection CSVs cannot carry an attack subtype")
        elif self.attack_subtype is None:
            raise ValueError("attack collection CSVs require an attack subtype")

    def require_attack_subtype(self) -> AttackSubtypeId:
        if self.attack_subtype is None:
            raise DatasetError("attack collection CSV is missing a subtype", path=self.path)
        return self.attack_subtype


@dataclass(frozen=True, slots=True)
class IngestedFederationTables:
    benign: tuple[DeviceFeatureParts, ...]
    attacks: tuple[DeviceAttackParts, ...]
    kept_columns: tuple[ColumnName, ...]
    dropped_rows: RowCount

    def benign_parts_for(self, client_id: ClientId) -> tuple[FeatureMatrix, ...]:
        for item in self.benign:
            if item.client_id == client_id:
                return item.parts
        return ()

    def attack_blocks_for(self, client_id: ClientId) -> tuple[AttackFeatureBlock, ...]:
        for item in self.attacks:
            if item.client_id == client_id:
                return item.blocks
        return ()


def _merge_benign_parts(parts: tuple[DeviceFeatureParts, ...]) -> tuple[DeviceFeatureParts, ...]:
    clients = tuple(sorted({item.client_id for item in parts}))
    return tuple(
        DeviceFeatureParts(
            client_id=client_id,
            parts=tuple(
                matrix for item in parts if item.client_id == client_id for matrix in item.parts
            ),
        )
        for client_id in clients
    )


def _merge_attack_parts(parts: tuple[DeviceAttackParts, ...]) -> tuple[DeviceAttackParts, ...]:
    clients = tuple(sorted({item.client_id for item in parts}))
    return tuple(
        DeviceAttackParts(
            client_id=client_id,
            blocks=tuple(
                block for item in parts if item.client_id == client_id for block in item.blocks
            ),
        )
        for client_id in clients
    )


def _read_csv_string_schema(path: Path, row_limit: RowCount | None = None) -> pl.DataFrame:
    preview = pl.read_csv(path, n_rows=1, ignore_errors=True)
    schema_overrides = {column: pl.String for column in preview.columns}
    return pl.read_csv(
        path,
        n_rows=None if row_limit is None else row_limit,
        schema_overrides=schema_overrides,
    )


@dataclass(frozen=True, slots=True)
class _ColumnParseTotals:
    column: ColumnName
    rows: RowCount
    numeric_rows: RowCount


def _column_parse_totals(
    frame: pl.DataFrame, layout: CicIotDiadDatasetConfig
) -> tuple[_ColumnParseTotals, ...]:
    totals: list[_ColumnParseTotals] = []
    for column in frame.columns:
        if (
            column == layout.device_column
            or column == layout.target_column
            or column in layout.excluded_features
        ):
            continue
        totals.append(
            _ColumnParseTotals(
                column=column,
                rows=frame.height,
                numeric_rows=frame.height
                - frame[column].cast(pl.Float64, strict=False).null_count(),
            )
        )
    return tuple(totals)


def global_feature_manifest(
    captures: tuple[CollectionCsv, ...],
    layout: CicIotDiadDatasetConfig,
    numeric_parse_threshold: Probability,
) -> tuple[ColumnName, ...]:
    total_rows: defaultdict[ColumnName, RowCount] = defaultdict(lambda: ZERO_ROW_COUNT)
    numeric_rows: defaultdict[ColumnName, RowCount] = defaultdict(lambda: ZERO_ROW_COUNT)
    for capture in captures:
        frame = _read_csv_string_schema(capture.path)
        for parse_totals in _column_parse_totals(frame, layout):
            total_rows[parse_totals.column] += parse_totals.rows
            numeric_rows[parse_totals.column] += parse_totals.numeric_rows
    manifest = tuple(
        sorted(
            column
            for column in total_rows
            if numeric_rows[column] / total_rows[column] >= numeric_parse_threshold
        )
    )
    if not manifest:
        raise ValueError("no feature columns survived the dataset-wide parse gate")
    return manifest


def _numeric_feature_matrix(frame: pl.DataFrame, columns: tuple[ColumnName, ...]) -> np.ndarray:
    numeric = frame.select([pl.col(column).cast(pl.Float64, strict=False) for column in columns])
    return numeric.to_numpy()


def _complete_numeric_row_mask(numeric_matrix: np.ndarray) -> np.ndarray:
    complete = np.logical_not(np.isnan(numeric_matrix)).all(axis=1)
    return np.asarray(complete)


def _device_feature_groups(
    frame: pl.DataFrame,
    numeric_matrix: np.ndarray,
    device_column: ColumnName,
    complete_mask: np.ndarray,
) -> tuple[DeviceFeatures, ...]:
    positions = np.flatnonzero(complete_mask)
    device_values: np.ndarray = frame[device_column].to_numpy()[positions]
    features = numeric_matrix[positions]
    groups: list[DeviceFeatures] = []
    for device_value in np.unique(device_values):
        groups.append(
            DeviceFeatures(
                client_id=_normalize_device(str(device_value)),
                features=FeatureMatrix(features[np.flatnonzero(device_values == device_value)]),
            )
        )
    return tuple(groups)


def read_packet_csv(
    path: Path,
    layout: CicIotDiadDatasetConfig,
    manifest: tuple[ColumnName, ...],
    row_limit: RowCount | None = None,
) -> PacketCsvIngestResult:
    frame = _read_csv_string_schema(path, row_limit)
    if layout.device_column not in frame.columns:
        raise ValueError(f"device column {layout.device_column!r} not found in {path}")
    missing = tuple(column for column in manifest if column not in frame.columns)
    if missing:
        raise DatasetError(
            f"external CSV {path} lacks frozen manifest feature(s) {missing}",
            path=path,
        )
    frame = _normalized_device_frame(frame, layout.device_column)
    numeric_matrix = _numeric_feature_matrix(frame, manifest)
    complete_mask = _complete_numeric_row_mask(numeric_matrix)
    dropped_rows = int((~complete_mask).sum())
    devices = sorted(
        _device_feature_groups(frame, numeric_matrix, layout.device_column, complete_mask),
        key=lambda device: device.client_id,
    )
    return PacketCsvIngestResult(
        devices=tuple(devices),
        kept_columns=manifest,
        dropped_rows=dropped_rows,
    )


_NON_DEVICE_DESTINATIONS = frozenset(
    {
        "ff:ff:ff:ff:ff:ff",
        "01:80:c2:00:00:00",
        "01:80:c2:00:00:03",
        "00:00:00:00:00:00",
    }
)


def _is_non_device_destination(destination: ClientId) -> bool:
    if destination in _NON_DEVICE_DESTINATIONS:
        return True
    return destination.startswith(("33:33:", "01:00:5e:"))


def _read_attack_capture_csv(
    path: Path,
    layout: CicIotDiadDatasetConfig,
    manifest: tuple[ColumnName, ...],
    row_limit: RowCount | None,
) -> PacketCsvIngestResult:
    frame = _read_csv_string_schema(path, row_limit)
    if layout.target_column not in frame.columns:
        raise ValueError(f"target column {layout.target_column!r} not found in {path}")
    missing = tuple(column for column in manifest if column not in frame.columns)
    if missing:
        raise DatasetError(
            f"external CSV {path} lacks frozen manifest feature(s) {missing}",
            path=path,
        )
    frame = _normalized_device_frame(frame, layout.target_column)
    numeric_matrix = _numeric_feature_matrix(frame, manifest)
    complete_mask = _complete_numeric_row_mask(numeric_matrix)
    dropped_rows = int((~complete_mask).sum())
    target_values = frame[layout.target_column].to_numpy()[complete_mask]
    devices: list[DeviceFeatures] = []
    for target in np.unique(target_values):
        destination = _normalize_device(str(target))
        if _is_non_device_destination(destination):
            continue
        positions = np.flatnonzero(target_values == target)
        devices.append(
            DeviceFeatures(
                client_id=destination,
                features=FeatureMatrix(numeric_matrix[complete_mask][positions]),
            )
        )
    return PacketCsvIngestResult(
        devices=tuple(devices),
        kept_columns=manifest,
        dropped_rows=dropped_rows,
    )


def _normalize_attack_subtype_id(subcategory: EnvironmentText) -> AttackSubtypeId:
    normalized = re.sub(r"[^a-z0-9]+", "_", subcategory.lower()).strip("_")
    return normalized


def device_eligibility(
    benign_rows: RowCount,
    attack_rows: RowCount,
    minimum_benign: RowCount,
    minimum_attack: RowCount,
) -> bool:
    return benign_rows >= minimum_benign and attack_rows >= minimum_attack


def _packet_collection_root(root: Path, layout: CicIotDiadDatasetConfig) -> Path:
    resolved = root / layout.packet_collection_directory
    if not resolved.is_dir():
        raise DatasetError(
            f"packet-based device collection not found under {root}: "
            f"missing {layout.packet_collection_directory}",
            path=root,
        )
    return resolved


def _classify_collection_csv(
    path: Path,
    collection_root: Path,
    layout: CicIotDiadDatasetConfig,
) -> CollectionCsv | None:
    parts = path.relative_to(collection_root).parts
    if len(parts) == 2 and parts[0] == layout.benign_category:
        return CollectionCsv(path=path, label=Label.BENIGN, attack_subtype=None)
    if len(parts) == 3:
        return CollectionCsv(
            path=path,
            label=Label.ATTACK,
            attack_subtype=_normalize_attack_subtype_id(parts[1]),
        )
    return None


def iter_collection_csvs(root: Path, layout: CicIotDiadDatasetConfig) -> tuple[CollectionCsv, ...]:
    collection_root = _packet_collection_root(root, layout)
    captures: list[CollectionCsv] = []
    for csv_path in sorted(collection_root.rglob(layout.csv_glob)):
        capture = _classify_collection_csv(csv_path, collection_root, layout)
        if capture is not None:
            captures.append(capture)
    return tuple(captures)


def _normalize_device(device_value: ClientId) -> ClientId:
    return re.sub(r"[^a-z0-9._: -]+", "_", device_value.strip().lower())


def _normalized_device_frame(frame: pl.DataFrame, device_column: ColumnName) -> pl.DataFrame:
    return frame.with_columns(pl.col(device_column).str.strip_chars().str.to_lowercase())


@dataclass(frozen=True, slots=True)
class _ClientRowTotal:
    client_id: ClientId
    rows: RowCount


@dataclass(frozen=True, slots=True)
class _ClientSubtypeTotal:
    client_id: ClientId
    subtype: AttackSubtypeId
    rows: RowCount


def _merged_benign_counts(parts: tuple[_ClientRowTotal, ...]) -> tuple[_ClientRowTotal, ...]:
    clients = tuple(sorted({part.client_id for part in parts}))
    return tuple(
        _ClientRowTotal(
            client_id=client_id,
            rows=sum(part.rows for part in parts if part.client_id == client_id),
        )
        for client_id in clients
    )


def _merged_attack_counts(
    parts: tuple[_ClientSubtypeTotal, ...],
) -> tuple[_ClientSubtypeTotal, ...]:
    keys = tuple(sorted({(part.client_id, part.subtype) for part in parts}))
    return tuple(
        _ClientSubtypeTotal(
            client_id=client_id,
            subtype=subtype,
            rows=sum(
                part.rows
                for part in parts
                if part.client_id == client_id and part.subtype == subtype
            ),
        )
        for client_id, subtype in keys
    )


def device_row_census(
    root: Path,
    layout: CicIotDiadDatasetConfig,
) -> ExternalCensus:
    benign_parts: list[_ClientRowTotal] = []
    attack_parts: list[_ClientSubtypeTotal] = []
    for capture in iter_collection_csvs(root, layout):
        if capture.label is Label.BENIGN:
            frame = pl.read_csv(
                capture.path,
                columns=[layout.device_column],
                schema_overrides={layout.device_column: pl.String},
            )
            frame = _normalized_device_frame(frame, layout.device_column)
            counts = frame[layout.device_column].value_counts()
            for row in counts.iter_rows(named=True):
                benign_parts.append(
                    _ClientRowTotal(
                        client_id=_normalize_device(str(row[layout.device_column])),
                        rows=int(row["count"]),
                    )
                )
            continue
        frame = pl.read_csv(
            capture.path,
            columns=[layout.target_column],
            schema_overrides={layout.target_column: pl.String},
        )
        frame = _normalized_device_frame(frame, layout.target_column)
        subtype = capture.require_attack_subtype()
        counts = frame[layout.target_column].value_counts()
        for row in counts.iter_rows(named=True):
            destination = _normalize_device(str(row[layout.target_column]))
            if _is_non_device_destination(destination):
                continue
            attack_parts.append(
                _ClientSubtypeTotal(
                    client_id=destination,
                    subtype=subtype,
                    rows=int(row["count"]),
                )
            )
    benign_counts = _merged_benign_counts(tuple(benign_parts))
    attack_counts = _merged_attack_counts(tuple(attack_parts))
    devices: list[ExternalDeviceCounts] = []
    client_ids = tuple(
        sorted(
            {item.client_id for item in benign_counts} | {item.client_id for item in attack_counts}
        )
    )
    for device in client_ids:
        attacks = tuple(
            sorted(
                (
                    AttackSubtypeCount(subtype=item.subtype, rows=item.rows)
                    for item in attack_counts
                    if item.client_id == device
                ),
                key=lambda item: item.subtype,
            )
        )
        devices.append(
            ExternalDeviceCounts(
                client_id=device,
                benign_rows=next(
                    (item.rows for item in benign_counts if item.client_id == device),
                    0,
                ),
                attacks=attacks,
            )
        )
    return ExternalCensus(tuple(devices))


def assess_external_evidence(
    census: ExternalCensus,
    minimum_benign_rows: RowCount,
    minimum_attack_rows: RowCount,
    minimum_confirmatory_clients: RowCount,
) -> ExternalEvidenceAssessment:
    eligible: list[ClientId] = []
    for device in census.devices:
        attack_rows = sum(item.rows for item in device.attacks)
        if device_eligibility(
            benign_rows=device.benign_rows,
            attack_rows=attack_rows,
            minimum_benign=minimum_benign_rows,
            minimum_attack=minimum_attack_rows,
        ):
            eligible.append(device.client_id)
    qualifying = len(eligible)
    return ExternalEvidenceAssessment(
        qualifying_clients=qualifying,
        evidence_level=(
            ReplicationEvidenceLevel.CONFIRMATORY
            if qualifying >= minimum_confirmatory_clients
            else ReplicationEvidenceLevel.SUPPORTIVE
        ),
        eligible_devices=tuple(eligible),
    )


def _read_capture_csv(
    capture: CollectionCsv,
    layout: CicIotDiadDatasetConfig,
    manifest: tuple[ColumnName, ...],
    row_limit: RowCount | None,
) -> PacketCsvIngestResult:
    if capture.label is Label.BENIGN:
        return read_packet_csv(
            path=capture.path,
            layout=layout,
            manifest=manifest,
            row_limit=row_limit,
        )
    return _read_attack_capture_csv(
        path=capture.path,
        layout=layout,
        manifest=manifest,
        row_limit=row_limit,
    )


def _ingest_device_parts(
    capture: CollectionCsv, ingest: PacketCsvIngestResult
) -> tuple[tuple[DeviceFeatureParts, ...], tuple[DeviceAttackParts, ...]]:
    if capture.label is Label.BENIGN:
        return (
            tuple(
                DeviceFeatureParts(client_id=device.client_id, parts=(device.features,))
                for device in ingest.devices
            ),
            (),
        )
    subtype = capture.require_attack_subtype()
    return (
        (),
        tuple(
            DeviceAttackParts(
                client_id=device.client_id,
                blocks=(
                    AttackFeatureBlock(
                        subtype=subtype,
                        source_file=capture.path.name,
                        features=device.features,
                    ),
                ),
            )
            for device in ingest.devices
        ),
    )


def _ingest_federation_csvs(
    root: Path,
    layout: CicIotDiadDatasetConfig,
    numeric_parse_threshold: Probability,
    row_limit: RowCount | None,
) -> IngestedFederationTables:
    captures = iter_collection_csvs(root, layout)
    if not captures:
        raise DatasetError("no packet CSVs found under the raw dataset root", path=root)
    manifest = global_feature_manifest(captures, layout, numeric_parse_threshold)
    benign_parts: list[DeviceFeatureParts] = []
    attack_parts: list[DeviceAttackParts] = []
    dropped_rows = 0
    for capture in captures:
        ingest = _read_capture_csv(
            capture,
            layout,
            manifest,
            row_limit,
        )
        dropped_rows += ingest.dropped_rows
        capture_benign, capture_attacks = _ingest_device_parts(capture, ingest)
        benign_parts.extend(capture_benign)
        attack_parts.extend(capture_attacks)
    return IngestedFederationTables(
        benign=_merge_benign_parts(tuple(benign_parts)),
        attacks=_merge_attack_parts(tuple(attack_parts)),
        kept_columns=manifest,
        dropped_rows=dropped_rows,
    )


def _merged_device_dataset(
    device_key: ClientId,
    ingested: IngestedFederationTables,
) -> DeviceDataset:
    benign_parts = ingested.benign_parts_for(device_key)
    benign = (
        FeatureMatrix(np.vstack([part.values for part in benign_parts]))
        if benign_parts
        else FeatureMatrix(np.empty((0, len(ingested.kept_columns))))
    )
    _require_feature_width(benign, ingested.kept_columns, device_key)
    blocks = ingested.attack_blocks_for(device_key)
    merged_blocks: list[AttackFeatureBlock] = []
    for subtype in sorted({block.subtype for block in blocks}):
        subtype_blocks = tuple(block for block in blocks if block.subtype == subtype)
        merged = FeatureMatrix(np.vstack([block.features.values for block in subtype_blocks]))
        _require_feature_width(merged, ingested.kept_columns, device_key)
        merged_blocks.append(
            AttackFeatureBlock(
                subtype=subtype,
                source_file="+".join(block.source_file for block in subtype_blocks),
                features=merged,
            )
        )
    return DeviceDataset(
        client_id=device_key,
        benign_source_file=EXTERNAL_SOURCE_FILE,
        benign=benign,
        attacks=tuple(merged_blocks),
    )


def prepare_external_federation(
    root: Path,
    layout: CicIotDiadDatasetConfig,
    numeric_parse_threshold: Probability,
    eligible_devices: tuple[ClientId, ...],
    row_limit: RowCount | None = None,
) -> ExternalFederation:
    if not eligible_devices:
        raise DatasetError("external federation requires eligible devices", path=root)
    ingested = _ingest_federation_csvs(
        root,
        layout,
        numeric_parse_threshold,
        row_limit,
    )
    present = {item.client_id for item in ingested.benign} | {
        item.client_id for item in ingested.attacks
    }
    devices = tuple(
        _merged_device_dataset(device_key, ingested)
        for device_key in sorted(present)
        if device_key in eligible_devices
    )
    if not devices:
        raise DatasetError(
            "eligible external devices produced no feature matrices",
            path=root,
        )
    return ExternalFederation(
        devices=devices,
        kept_columns=ingested.kept_columns,
        dropped_rows=ingested.dropped_rows,
    )


def _require_feature_width(
    matrix: FeatureMatrix,
    kept_columns: tuple[ColumnName, ...],
    device_key: ClientId,
) -> None:
    if matrix.feature_count != len(kept_columns):
        raise ValueError(
            f"device {device_key} feature width {matrix.feature_count} "
            f"does not match kept column count {len(kept_columns)}"
        )
