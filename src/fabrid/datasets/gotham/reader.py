from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fabrid.datasets.cic_iot_diad.reader import (
    FeatureColumns,
    select_numeric_feature_columns,
)
from fabrid.datasets.common import AttackFeatureBlock, DeviceDataset, FeatureMatrix
from fabrid.domain.identifiers import (
    AttackSubtypeId,
    ClientId,
    ColumnName,
    SourceFileId,
    SourceLabel,
)
from fabrid.domain.values import Probability, RowCount

_DEFAULT_DEVICE_COLUMN = ColumnName("eth.src")
_DEFAULT_LABEL_COLUMN = ColumnName("label")
_DEFAULT_BENIGN_LABEL = SourceLabel("Benign")
_DEFAULT_TIMESTAMP_COLUMN = ColumnName("frame.time")


@dataclass(frozen=True, slots=True)
class GothamIngestResult:
    devices: tuple[DeviceDataset, ...]
    kept_columns: FeatureColumns
    dropped_rows: RowCount

    def __post_init__(self) -> None:
        client_ids = tuple(device.client_id for device in self.devices)
        if len(set(client_ids)) != len(client_ids):
            raise ValueError("Gotham ingestion contains duplicate clients")


def read_processed_csv(
    path: Path,
    excluded_features: FeatureColumns,
    numeric_parse_threshold: Probability,
    device_column: ColumnName = _DEFAULT_DEVICE_COLUMN,
    label_column: ColumnName = _DEFAULT_LABEL_COLUMN,
    benign_label: SourceLabel = _DEFAULT_BENIGN_LABEL,
    timestamp_column: ColumnName = _DEFAULT_TIMESTAMP_COLUMN,
    row_limit: RowCount | None = None,
) -> GothamIngestResult:
    frame = pd.read_csv(
        path,
        nrows=None if row_limit is None else row_limit.value,
        low_memory=False,
    )
    for required in (device_column, label_column):
        if required.value not in frame.columns:
            raise ValueError(f"column {required.value!r} not found in {path}")

    selection_exclusions = FeatureColumns(
        tuple(
            dict.fromkeys(
                (
                    *excluded_features.values,
                    label_column,
                    timestamp_column,
                )
            )
        )
    )
    reports = select_numeric_feature_columns(
        frame,
        selection_exclusions,
        numeric_parse_threshold,
        device_column,
    )
    kept_columns = FeatureColumns(
        tuple(report.column for report in reports if report.kept)
    )
    if not kept_columns.values:
        raise ValueError(f"no feature columns survived filtering for {path}")

    numeric_frame = frame[
        [column.value for column in kept_columns.values]
    ].apply(pd.to_numeric, errors="coerce")
    complete_mask = numeric_frame.notna().all(axis=1)
    dropped_rows = RowCount(int((~complete_mask).sum()))
    filtered = frame.loc[complete_mask]
    numeric_values = numeric_frame.loc[complete_mask]

    devices: list[DeviceDataset] = []
    for raw_device in sorted(str(value) for value in filtered[device_column.value].unique()):
        client_id = ClientId(raw_device)
        device_mask = filtered[device_column.value].astype(str).to_numpy() == raw_device
        device_rows = filtered.loc[device_mask]
        device_features = numeric_values.loc[device_mask]

        benign_mask = (
            device_rows[label_column.value].astype(str).to_numpy() == benign_label.value
        )
        benign = FeatureMatrix(
            device_features.loc[benign_mask].to_numpy(dtype=np.float64)
        )

        attacks: list[AttackFeatureBlock] = []
        attack_labels = sorted(
            {
                str(value)
                for value in device_rows[label_column.value]
                if str(value) != benign_label.value
            }
        )
        for raw_label in attack_labels:
            label_mask = device_rows[label_column.value].astype(str).to_numpy() == raw_label
            attacks.append(
                AttackFeatureBlock(
                    subtype=AttackSubtypeId(raw_label),
                    source_file=SourceFileId(path.name),
                    features=FeatureMatrix(
                        device_features.loc[label_mask].to_numpy(dtype=np.float64)
                    ),
                )
            )

        devices.append(
            DeviceDataset(
                client_id=client_id,
                benign_source_file=SourceFileId(path.name),
                benign=benign,
                attacks=tuple(attacks),
            )
        )

    return GothamIngestResult(
        devices=tuple(devices),
        kept_columns=kept_columns,
        dropped_rows=dropped_rows,
    )
