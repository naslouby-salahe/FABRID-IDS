"""Raw N-BaIoT CSV ingestion: standalone reader, no external research-stack dependency.

Reads each device's benign/attack CSVs directly, preserving source row order
(source_row = original CSV data-row index, 0-based, header excluded).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fabrid.evaluation.record_level import AttackSubtype, ClientId

_BENIGN_FILENAME = "benign_traffic.csv"
_BASHLITE_DIRNAME = "gafgyt_attacks"
_MIRAI_DIRNAME = "mirai_attacks"

_BASHLITE_ATTACK_FILES: dict[str, str] = {
    "combo": "bashlite_combo",
    "junk": "bashlite_junk",
    "scan": "bashlite_scan",
    "tcp": "bashlite_tcp",
    "udp": "bashlite_udp",
}
_MIRAI_ATTACK_FILES: dict[str, str] = {
    "ack": "mirai_ack",
    "scan": "mirai_scan",
    "syn": "mirai_syn",
    "udp": "mirai_udp",
    "udpplain": "mirai_udpplain",
}


@dataclass(frozen=True, slots=True)
class RawDeviceData:
    client_id: ClientId
    benign_features: np.ndarray
    attack_features_by_subtype: dict[AttackSubtype, np.ndarray]


def _read_feature_csv(path: Path) -> np.ndarray:
    return pd.read_csv(path).to_numpy(dtype=np.float64)


def _read_attack_family(
    family_dir: Path, filename_to_subtype: dict[str, str]
) -> dict[AttackSubtype, np.ndarray]:
    features: dict[AttackSubtype, np.ndarray] = {}
    if not family_dir.exists():
        return features
    for filename, subtype in filename_to_subtype.items():
        path = family_dir / f"{filename}.csv"
        if path.exists():
            features[AttackSubtype(subtype)] = _read_feature_csv(path)
    return features


def read_device_directory(client_id: ClientId, device_dir: Path) -> RawDeviceData:
    benign_path = device_dir / _BENIGN_FILENAME
    if not benign_path.exists():
        raise FileNotFoundError(
            f"missing {_BENIGN_FILENAME} for client {client_id} at {device_dir}"
        )
    benign_features = _read_feature_csv(benign_path)

    attack_features_by_subtype: dict[AttackSubtype, np.ndarray] = {}
    attack_features_by_subtype.update(
        _read_attack_family(device_dir / _BASHLITE_DIRNAME, _BASHLITE_ATTACK_FILES)
    )
    attack_features_by_subtype.update(
        _read_attack_family(device_dir / _MIRAI_DIRNAME, _MIRAI_ATTACK_FILES)
    )

    return RawDeviceData(
        client_id=client_id,
        benign_features=benign_features,
        attack_features_by_subtype=attack_features_by_subtype,
    )
