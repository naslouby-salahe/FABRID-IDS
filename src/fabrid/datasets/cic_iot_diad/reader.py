from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fabrid.datasets.common import FeatureMatrix
from fabrid.domain.enums import FeatureColumnStatus
from fabrid.domain.identifiers import ClientId, ColumnName
from fabrid.domain.values import Probability, RowCount


@dataclass(frozen=True, slots=True)
class ColumnParseReport:
    column: ColumnName
    parse_success: Probability
    status: FeatureColumnStatus


@dataclass(frozen=True, slots=True)
class FeatureColumns:
    values: tuple[ColumnName, ...]

    def __post_init__(self) -> None:
        if len(set(self.values)) != len(self.values):
            raise ValueError("feature columns contain duplicates")

    def contains(self, column: ColumnName) -> bool:
        return column in self.values


@dataclass(frozen=True, slots=True)
class DeviceFeatures:
    client_id: ClientId
    features: FeatureMatrix


@dataclass(frozen=True, slots=True)
class PacketCsvIngestResult:
    devices: tuple[DeviceFeatures, ...]
    kept_columns: FeatureColumns
    column_reports: tuple[ColumnParseReport, ...]
    dropped_rows: RowCount

    def __post_init__(self) -> None:
        client_ids = tuple(device.client_id for device in self.devices)
        if len(set(client_ids)) != len(client_ids):
            raise ValueError("packet CSV ingestion contains duplicate clients")


def _numeric_parse_success(series: pd.Series) -> Probability:
    parsed = pd.to_numeric(series, errors="coerce")
    return Probability(float(parsed.notna().mean()))


def select_numeric_feature_columns(
    frame: pd.DataFrame,
    excluded_features: FeatureColumns,
    threshold: Probability,
    device_column: ColumnName,
) -> tuple[ColumnParseReport, ...]:
    reports: list[ColumnParseReport] = []
    for raw_column in frame.columns:
        column = ColumnName(str(raw_column))
        if column == device_column or excluded_features.contains(column):
            reports.append(
                ColumnParseReport(
                    column=column,
                    parse_success=Probability(0.0),
                    status=FeatureColumnStatus.EXCLUDED,
                )
            )
            continue
        parse_success = _numeric_parse_success(frame[raw_column])
        reports.append(
            ColumnParseReport(
                column=column,
                parse_success=parse_success,
                status=(
                    FeatureColumnStatus.KEPT
                    if parse_success.value >= threshold.value
                    else FeatureColumnStatus.PARSE_REJECTED
                ),
            )
        )
    return tuple(reports)


def read_packet_csv(
    path: Path,
    excluded_features: FeatureColumns,
    numeric_parse_threshold: Probability,
    device_column: ColumnName,
    row_limit: RowCount | None = None,
) -> PacketCsvIngestResult:
    frame = pd.read_csv(
        path,
        nrows=None if row_limit is None else row_limit.value,
        low_memory=False,
    )
    if device_column.value not in frame.columns:
        raise ValueError(
            f"device column {device_column.value!r} not found in {path}"
        )

    reports = select_numeric_feature_columns(
        frame,
        excluded_features,
        numeric_parse_threshold,
        device_column,
    )
    kept_columns = FeatureColumns(
        tuple(
            report.column
            for report in reports
            if report.status is FeatureColumnStatus.KEPT
        )
    )
    if not kept_columns.values:
        raise ValueError(f"no feature columns survived filtering for {path}")

    numeric_frame = frame[
        [column.value for column in kept_columns.values]
    ].apply(pd.to_numeric, errors="coerce")
    complete_mask = numeric_frame.notna().all(axis=1)
    dropped_rows = RowCount(int((~complete_mask).sum()))

    device_series = frame.loc[complete_mask, device_column.value]
    complete_features = numeric_frame.loc[complete_mask]
    devices: list[DeviceFeatures] = []
    for device_value, indices in device_series.groupby(device_series).groups.items():
        devices.append(
            DeviceFeatures(
                client_id=ClientId(str(device_value)),
                features=FeatureMatrix(
                    complete_features.loc[indices].to_numpy(dtype=np.float64)
                ),
            )
        )

    return PacketCsvIngestResult(
        devices=tuple(sorted(devices, key=lambda device: device.client_id.value)),
        kept_columns=kept_columns,
        column_reports=reports,
        dropped_rows=dropped_rows,
    )
