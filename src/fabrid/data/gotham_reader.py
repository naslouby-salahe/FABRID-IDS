"""Gotham Dataset 2025 `processed/*.csv` ingestion. Unlike CICIoMT2024, these files carry a
real per-row device-identity column (`eth.src`), a real per-row attack-subtype label (`label`,
e.g. `"Telnet Brute Force"`, `"TCP Scan"`, `"Benign"`), and a real wall-clock timestamp
(`frame.time`) — the shape FABRID's decision layer expects (per-client benign vs. per-subtype
attack rows), unlike CICIoMT2024's pooled/session-only files (see `ciciomt_reader.py`).

Column selection reuses `cic_iot_diad_reader.select_numeric_feature_columns` (numeric-parse-rate
filtering plus always-excluding the identity column) rather than duplicating that logic — the
same mixed-numeric/categorical-column problem applies here (`frame.protocols`, `tcp.flags`, etc).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fabrid.data.cic_iot_diad_reader import select_numeric_feature_columns
from fabrid.evaluation.record_level import AttackSubtype, ClientId

_DEFAULT_DEVICE_COLUMN = "eth.src"
_DEFAULT_LABEL_COLUMN = "label"
_DEFAULT_BENIGN_LABEL = "Benign"
_DEFAULT_TIMESTAMP_COLUMN = "frame.time"


@dataclass(frozen=True, slots=True)
class GothamIngestResult:
    benign_features_by_client: dict[ClientId, np.ndarray]
    attack_features_by_client_and_subtype: dict[ClientId, dict[AttackSubtype, np.ndarray]]
    kept_feature_columns: tuple[str, ...]
    rows_dropped_unparseable: int


def read_processed_csv(
    path: Path,
    excluded_features: frozenset[str],
    numeric_parse_success_threshold: float,
    device_column: str = _DEFAULT_DEVICE_COLUMN,
    label_column: str = _DEFAULT_LABEL_COLUMN,
    benign_label: str = _DEFAULT_BENIGN_LABEL,
    timestamp_column: str = _DEFAULT_TIMESTAMP_COLUMN,
    nrows: int | None = None,
) -> GothamIngestResult:
    frame = pd.read_csv(path, nrows=nrows, low_memory=False)
    for required in (device_column, label_column):
        if required not in frame.columns:
            raise ValueError(f"column {required!r} not found in {path}")

    always_excluded = excluded_features | {label_column, timestamp_column}
    reports = select_numeric_feature_columns(
        frame, always_excluded, numeric_parse_success_threshold, device_column
    )
    kept_columns = tuple(r.column for r in reports if r.kept)
    if not kept_columns:
        raise ValueError(f"no feature columns survived numeric-parse filtering for {path}")

    numeric_frame = frame[list(kept_columns)].apply(pd.to_numeric, errors="coerce")
    complete_row_mask = numeric_frame.notna().all(axis=1)
    rows_dropped_unparseable = int((~complete_row_mask).sum())

    device_series = frame.loc[complete_row_mask, device_column]
    label_series = frame.loc[complete_row_mask, label_column]
    complete_numeric_frame = numeric_frame.loc[complete_row_mask]

    benign_by_client: dict[ClientId, np.ndarray] = {}
    attack_by_client: dict[ClientId, dict[AttackSubtype, np.ndarray]] = {}

    unique_devices = sorted(str(v) for v in device_series.unique())
    unique_labels = sorted(str(v) for v in label_series.unique())
    for device_value in unique_devices:
        for label_value in unique_labels:
            row_mask = (device_series.to_numpy() == device_value) & (
                label_series.to_numpy() == label_value
            )
            if not row_mask.any():
                continue
            client_id = ClientId(device_value)
            matrix = complete_numeric_frame.to_numpy(dtype=np.float64)[row_mask]
            if label_value == benign_label:
                benign_by_client[client_id] = matrix
            else:
                attack_by_client.setdefault(client_id, {})[AttackSubtype(label_value)] = matrix

    return GothamIngestResult(
        benign_features_by_client=benign_by_client,
        attack_features_by_client_and_subtype=attack_by_client,
        kept_feature_columns=kept_columns,
        rows_dropped_unparseable=rows_dropped_unparseable,
    )
