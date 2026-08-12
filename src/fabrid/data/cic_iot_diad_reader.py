"""CIC IoT-DIAD 2024 packet-level CSV ingestion (external replication candidate,
EXTERNAL-001..006). Unlike N-BaIoT's purely numeric CSVs, this dataset's packet-level feature
files mix numeric telemetry with categorical/identity columns and interleave multiple devices'
rows within a single file, so ingestion must: select feature columns by per-column numeric-parse
success rate (EXTERNAL-005), enforce the frozen leakage exclusion list (EXTERNAL-004), then group
surviving numeric rows by device identity (EXTERNAL-002).

No silent coercion: a row with an unparseable value in a kept feature column is dropped, not
filled with NaN/0 (roadmap section 78's "no silent coercion" constraint).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fabrid.evaluation.record_level import ClientId


@dataclass(frozen=True, slots=True)
class ColumnParseReport:
    column: str
    parse_success_fraction: float
    kept: bool


@dataclass(frozen=True, slots=True)
class PacketCsvIngestResult:
    device_features: dict[ClientId, np.ndarray]
    kept_feature_columns: tuple[str, ...]
    column_reports: tuple[ColumnParseReport, ...]
    rows_dropped_unparseable: int


def _numeric_parse_success_fraction(series: pd.Series) -> float:
    parsed = pd.to_numeric(series, errors="coerce")
    return float(parsed.notna().mean())


def select_numeric_feature_columns(
    frame: pd.DataFrame,
    excluded_features: frozenset[str],
    numeric_parse_success_threshold: float,
    device_column: str,
) -> tuple[ColumnParseReport, ...]:
    """One report per column in `frame`: excluded-by-config columns (plus `device_column`,
    always excluded regardless of whether it is separately named in `excluded_features` — the
    client-identity key must never leak into the feature vector) are reported with `kept=False`
    and `parse_success_fraction=0.0` (never evaluated, not a parse failure); everything else is
    kept iff its numeric-parse success rate meets the threshold.
    """
    all_excluded = excluded_features | {device_column}
    reports: list[ColumnParseReport] = []
    for column in frame.columns:
        if column in all_excluded:
            reports.append(ColumnParseReport(column=column, parse_success_fraction=0.0, kept=False))
            continue
        fraction = _numeric_parse_success_fraction(frame[column])
        reports.append(
            ColumnParseReport(
                column=column,
                parse_success_fraction=fraction,
                kept=fraction >= numeric_parse_success_threshold,
            )
        )
    return tuple(reports)


def read_packet_csv(
    path: Path,
    excluded_features: frozenset[str],
    numeric_parse_success_threshold: float,
    device_column: str,
    nrows: int | None = None,
) -> PacketCsvIngestResult:
    frame = pd.read_csv(path, nrows=nrows, low_memory=False)
    if device_column not in frame.columns:
        raise ValueError(f"device column {device_column!r} not found in {path}")

    reports = select_numeric_feature_columns(
        frame, excluded_features, numeric_parse_success_threshold, device_column
    )
    kept_columns = tuple(r.column for r in reports if r.kept)
    if not kept_columns:
        raise ValueError(f"no feature columns survived numeric-parse filtering for {path}")

    numeric_frame = frame[list(kept_columns)].apply(pd.to_numeric, errors="coerce")
    complete_row_mask = numeric_frame.notna().all(axis=1)
    rows_dropped_unparseable = int((~complete_row_mask).sum())

    device_features: dict[ClientId, np.ndarray] = {}
    device_series = frame.loc[complete_row_mask, device_column]
    complete_numeric_frame = numeric_frame.loc[complete_row_mask]
    for device_value, group_index in device_series.groupby(device_series).groups.items():
        device_features[ClientId(str(device_value))] = complete_numeric_frame.loc[
            group_index
        ].to_numpy(dtype=np.float64)

    return PacketCsvIngestResult(
        device_features=device_features,
        kept_feature_columns=kept_columns,
        column_reports=reports,
        rows_dropped_unparseable=rows_dropped_unparseable,
    )
