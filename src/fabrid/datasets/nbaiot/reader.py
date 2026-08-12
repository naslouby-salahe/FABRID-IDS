from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fabrid.datasets.common import AttackFeatureBlock, DeviceDataset, FeatureMatrix
from fabrid.domain.identifiers import AttackSubtypeId, ClientId

_BENIGN_FILENAME = "benign_traffic.csv"
_BASHLITE_DIRNAME = "gafgyt_attacks"
_MIRAI_DIRNAME = "mirai_attacks"


@dataclass(frozen=True, slots=True)
class _AttackFile:
    filename: str
    subtype: AttackSubtypeId


_BASHLITE_FILES = (
    _AttackFile("combo", AttackSubtypeId("bashlite_combo")),
    _AttackFile("junk", AttackSubtypeId("bashlite_junk")),
    _AttackFile("scan", AttackSubtypeId("bashlite_scan")),
    _AttackFile("tcp", AttackSubtypeId("bashlite_tcp")),
    _AttackFile("udp", AttackSubtypeId("bashlite_udp")),
)
_MIRAI_FILES = (
    _AttackFile("ack", AttackSubtypeId("mirai_ack")),
    _AttackFile("scan", AttackSubtypeId("mirai_scan")),
    _AttackFile("syn", AttackSubtypeId("mirai_syn")),
    _AttackFile("udp", AttackSubtypeId("mirai_udp")),
    _AttackFile("udpplain", AttackSubtypeId("mirai_udpplain")),
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
        path = family_dir / f"{file.filename}.csv"
        if path.exists():
            blocks.append(
                AttackFeatureBlock(
                    subtype=file.subtype,
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
        benign=_read_feature_csv(benign_path),
        attacks=attacks,
    )
