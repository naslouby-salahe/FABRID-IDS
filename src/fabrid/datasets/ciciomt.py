from __future__ import annotations

from pathlib import Path

import polars as pl

from fabrid.config import CiciomtDatasetConfig, ColumnName, EventCriterionId, RowCount
from fabrid.datasets.registry import (
    EventProvenanceEvidence,
    event_criterion,
    unavailable_event_evidence,
)


def _header_columns(path: Path) -> tuple[ColumnName, ...]:
    frame = pl.read_csv(path, n_rows=0)
    return tuple(frame.columns)


def audit_ciciomt_event_provenance(
    raw_root: Path,
    layout: CiciomtDatasetConfig,
    maximum_files: RowCount | None = None,
) -> EventProvenanceEvidence:
    csv_paths = tuple(sorted(raw_root.rglob(layout.csv_glob)))
    if maximum_files is not None:
        csv_paths = csv_paths[:maximum_files]
    if not csv_paths:
        return unavailable_event_evidence(
            "no CICIoMT flow CSV files available under the raw dataset root"
        )
    headers = tuple(_header_columns(path) for path in csv_paths)
    timestamp_columns = set(layout.timestamp_columns)
    identity_columns = set(layout.identity_columns)
    has_timestamp = any(any(column in timestamp_columns for column in header) for header in headers)
    has_identity = any(any(column in identity_columns for column in header) for header in headers)
    return EventProvenanceEvidence(
        criteria=(
            event_criterion(
                EventCriterionId.IMMUTABLE_CLIENT_ID,
                has_identity,
                "CICIoMT flow CSVs expose no client identity column "
                f"(searched {sorted(identity_columns)}); flow features are pooled aggregates",
            ),
            event_criterion(
                EventCriterionId.PACKET_TIMESTAMP,
                has_timestamp,
                "CICIoMT flow CSVs expose no packet timestamp column "
                f"(searched {sorted(timestamp_columns)}); observed columns "
                f"sampled from {len(csv_paths)} files",
            ),
            event_criterion(
                EventCriterionId.INTERVAL_PROVENANCE,
                False,
                "attack identity is file-level (filename subtype with train/test split), "
                "not per-row interval labels within a capture",
            ),
            event_criterion(
                EventCriterionId.DETERMINISTIC_SCORE_ASSOCIATION,
                False,
                "CICIoMT flow CSVs have neither a packet timestamp nor a client identity, "
                "so a score cannot be joined to a timed client row",
            ),
            event_criterion(
                EventCriterionId.WITHIN_CLIENT_ORDERING,
                False,
                "no per-row timestamps exist to establish within-client ordering",
            ),
            event_criterion(
                EventCriterionId.OBSERVATION_DURATION,
                False,
                "no timestamp provenance exists to establish observation duration",
            ),
            event_criterion(
                EventCriterionId.NON_OVERLAPPING_EVALUATION_PERIOD,
                False,
                "evaluation intervals cannot be delimited without timestamp or interval provenance",
            ),
        )
    )
