from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from fabrid.config import AttackFileMapping, ClientId, NbaiotDatasetConfig
from fabrid.datasets.registry import (
    AttackFeatureBlock,
    DeviceDataset,
    FeatureMatrix,
)
from fabrid.errors import DatasetError


def _read_feature_csv(path: Path) -> FeatureMatrix:
    frame = pl.read_csv(path)
    return FeatureMatrix(frame.to_numpy().astype(np.float64, copy=False))


def _read_attack_family(
    family_dir: Path,
    files: tuple[AttackFileMapping, ...],
) -> tuple[AttackFeatureBlock, ...]:
    if not family_dir.exists():
        return ()
    blocks: list[AttackFeatureBlock] = []
    for mapping in files:
        path = family_dir / mapping.filename
        if path.exists():
            blocks.append(
                AttackFeatureBlock(
                    subtype=mapping.subtype,
                    source_file=f"{family_dir.name}/{mapping.filename}",
                    features=_read_feature_csv(path),
                )
            )
    return tuple(blocks)


def read_device_directory(
    client_id: ClientId,
    device_dir: Path,
    layout: NbaiotDatasetConfig,
) -> DeviceDataset:
    benign_path = device_dir / layout.benign_filename
    if not benign_path.exists():
        raise DatasetError(
            f"missing {layout.benign_filename} for client {client_id} at {device_dir}",
            path=benign_path,
        )
    attacks = (
        *_read_attack_family(device_dir / layout.bashlite_directory, layout.bashlite_files),
        *_read_attack_family(device_dir / layout.mirai_directory, layout.mirai_files),
    )
    return DeviceDataset(
        client_id=client_id,
        benign_source_file=layout.benign_filename,
        benign=_read_feature_csv(benign_path),
        attacks=attacks,
    )
