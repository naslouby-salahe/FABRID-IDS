from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fabrid.datasets.common import FeatureMatrix
from fabrid.domain.enums import SourceSplit
from fabrid.domain.identifiers import AttackSubtypeId, ClientId, SourceFileId

_PROFILING_SUFFIX = ".pcap.csv"
_ATTACK_FILENAME_PATTERN = re.compile(
    r"^(?P<subtype>.+)_(?P<split>train|test)\.pcap\.csv$"
)


@dataclass(frozen=True, slots=True)
class ProfilingSession:
    client_id: ClientId
    source_file: SourceFileId
    features: FeatureMatrix


@dataclass(frozen=True, slots=True)
class ProfilingSessions:
    sessions: tuple[ProfilingSession, ...]

    def __post_init__(self) -> None:
        client_ids = tuple(session.client_id for session in self.sessions)
        if len(set(client_ids)) != len(client_ids):
            raise ValueError("profiling sessions contain duplicate session ids")


@dataclass(frozen=True, slots=True)
class PooledAttackFile:
    subtype: AttackSubtypeId
    source_split: SourceSplit
    source_file: SourceFileId
    features: FeatureMatrix


def _read_numeric_csv(path: Path) -> FeatureMatrix:
    return FeatureMatrix(pd.read_csv(path).to_numpy(dtype=np.float64))


def session_id_from_profiling_filename(filename: SourceFileId) -> ClientId:
    if not filename.value.endswith(_PROFILING_SUFFIX):
        raise ValueError(
            f"expected a {_PROFILING_SUFFIX} file, got {filename.value!r}"
        )
    return ClientId(filename.value[: -len(_PROFILING_SUFFIX)])


def read_profiling_directory(profiling_dir: Path) -> ProfilingSessions:
    sessions = tuple(
        ProfilingSession(
            client_id=session_id_from_profiling_filename(SourceFileId(path.name)),
            source_file=SourceFileId(path.name),
            features=_read_numeric_csv(path),
        )
        for path in sorted(profiling_dir.glob(f"*{_PROFILING_SUFFIX}"))
    )
    return ProfilingSessions(sessions)


def parse_attack_filename(
    filename: SourceFileId,
) -> tuple[AttackSubtypeId, SourceSplit]:
    match = _ATTACK_FILENAME_PATTERN.match(filename.value)
    if match is None:
        raise ValueError(
            f"filename {filename.value!r} does not match the CICIoMT attack convention"
        )
    return (
        AttackSubtypeId(match.group("subtype")),
        SourceSplit(match.group("split")),
    )


def read_attacks_directory(attacks_csv_dir: Path) -> tuple[PooledAttackFile, ...]:
    files: list[PooledAttackFile] = []
    for path in sorted(attacks_csv_dir.rglob(f"*{_PROFILING_SUFFIX}")):
        source_file = SourceFileId(path.name)
        subtype, source_split = parse_attack_filename(source_file)
        files.append(
            PooledAttackFile(
                subtype=subtype,
                source_split=source_split,
                source_file=source_file,
                features=_read_numeric_csv(path),
            )
        )
    return tuple(files)
