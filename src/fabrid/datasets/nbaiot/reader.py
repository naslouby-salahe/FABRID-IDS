from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fabrid.datasets.common import AttackFeatureBlock, DeviceDataset, FeatureMatrix
from fabrid.domain.identifiers import AttackSubtypeId, ClientId, SourceFileId

_BENIGN_FILENAME = "benign_traffic.csv"
_BASHLITE_DIRNAME = "gafgyt_attacks"
_MIRAI_DIRNAME = "mirai_attacks"


@dataclass(frozen=True, slots=True)
class _AttackFile:
    filename: SourceFileId
    subtype: AttackSubtypeId


_BASHLITE_FILES = (
    _AttackFile(SourceFileId("combo"), AttackSubtypeId("bashlite_combo")),
    _AttackFile(SourceFileId("junk"), AttackSubtypeId("bashlite_junk")),
    _AttackFile(SourceFileId("scan"), AttackSubtypeId("bashlite_scan")),
    _AttackFile(SourceFileId("tcp"), AttackSubtypeId("bashlite_tcp")),
    _AttackFile(SourceFileId("udp"), AttackSubtypeId("bashlite_udp")),
)
_MIRAI_FILES = (
    _AttackFile(SourceFileId("ack"), AttackSubtypeId("mirai_ack")),
    _AttackFile(SourceFileId("scan"), AttackSubtypeId("mirai_scan")),
    _AttackFile(SourceFileId("syn"), AttackSubtypeId("mirai_syn")),
    _AttackFile(SourceFileId("udp"), AttackSubtypeId("mirai_udp")),
    _AttackFile(SourceFileId("udpplain"), AttackSubtypeId("mirai_udpplain")),
)


def _read_feature_csv(path: Path) -> FeatureMatrix:
    values = pd.read_csv(path).to_numpy(dtype=np.float64)
    return FeatureMatrix(values)


def _read_attack_family(
    family_dir: Path,
    files: tuple[_AttackFile, ...],
) -> tuple[AttackFeatureBlock, ...]:
    if not family_dir.exists():
        return ()

    blocks: list[AttackFeatureBlock] = []
    for file in files:
        path = family_dir / f"{file.filename.value}.csv"
        if path.exists():
            blocks.append(
                AttackFeatureBlock(
                    subtype=file.subtype,
                    source_file=SourceFileId(
                        f"{family_dir.name}/{file.filename.value}.csv"
                    ),
                    features=_read_feature_csv(path),
                )
            )
    return tuple(blocks)


def read_device_directory(client_id: ClientId, device_dir: Path) -> DeviceDataset:
    benign_path = device_dir / _BENIGN_FILENAME
    if not benign_path.exists():
        raise FileNotFoundError(
            f"missing {_BENIGN_FILENAME} for client {client_id.value} at {device_dir}"
        )

    attacks = (
        *_read_attack_family(device_dir / _BASHLITE_DIRNAME, _BASHLITE_FILES),
        *_read_attack_family(device_dir / _MIRAI_DIRNAME, _MIRAI_FILES),
    )
    return DeviceDataset(
        client_id=client_id,
        benign_source_file=SourceFileId(_BENIGN_FILENAME),
        benign=_read_feature_csv(benign_path),
        attacks=attacks,
    )
