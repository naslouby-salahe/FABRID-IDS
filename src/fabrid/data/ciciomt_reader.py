"""CICIoMT2024 CSV ingestion. Structurally different from N-BaIoT/CIC IoT-DIAD 2024: every row
is a pure numeric flow-feature vector with no device/label column at all — identity and split
membership live entirely in the filename, and the two halves of the dataset have incompatible
shapes for FABRID's per-client benign+attack contract:

- `profiling/CSV/*.pcap.csv`: one file per capture *session* (benign only). Device names embed
  inconsistently in the filename (e.g. `Blink_Camera_LAN_MIC.pcap.csv` vs
  `SenseUBaby_Power.pcap.csv` vs `Active.pcap.csv`, a network-state capture with no device at
  all) — there is no reliable, non-guessed rule to merge sessions into one "device" grouping, so
  each file is treated as its own client/session identifier rather than inventing a merge
  heuristic.
- `attacks/CSV/{train,test}/*.pcap.csv`: one file per attack subtype, pooled across the whole
  network/broker, not per device. FABRID's decision layer needs per-client benign *and* attack
  data from the same client; this dataset does not provide that shape for its attack traffic, so
  this module does not attempt to fabricate a per-client attack split from pooled data. It is
  read as network-level attack data, for whatever pooled-baseline use it is fit for, not wired
  into `frontier`/`allocation` as a FABRID client population.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fabrid.evaluation.record_level import AttackSubtype, ClientId

_PROFILING_SUFFIX = ".pcap.csv"
_ATTACK_FILENAME_RE = re.compile(r"^(?P<subtype>.+)_(?P<split>train|test)\.pcap\.csv$")


def _read_numeric_csv(path: Path) -> np.ndarray:
    return pd.read_csv(path).to_numpy(dtype=np.float64)


def session_id_from_profiling_filename(filename: str) -> ClientId:
    if not filename.endswith(_PROFILING_SUFFIX):
        raise ValueError(f"expected a {_PROFILING_SUFFIX} file, got {filename!r}")
    return ClientId(filename[: -len(_PROFILING_SUFFIX)])


def read_profiling_directory(profiling_dir: Path) -> dict[ClientId, np.ndarray]:
    """One entry per `*.pcap.csv` file in `profiling_dir` (non-recursive), keyed by the file
    stem. All-benign; there is no attack counterpart per session in this dataset.
    """
    sessions: dict[ClientId, np.ndarray] = {}
    for path in sorted(profiling_dir.glob(f"*{_PROFILING_SUFFIX}")):
        sessions[session_id_from_profiling_filename(path.name)] = _read_numeric_csv(path)
    return sessions


@dataclass(frozen=True, slots=True)
class PooledAttackFile:
    """One attack-subtype file's pooled (not per-client) feature matrix."""

    attack_subtype: AttackSubtype
    is_train_split: bool
    features: np.ndarray


def parse_attack_filename(filename: str) -> tuple[AttackSubtype, bool]:
    match = _ATTACK_FILENAME_RE.match(filename)
    if match is None:
        raise ValueError(f"filename {filename!r} does not match '<subtype>_<train|test>.pcap.csv'")
    return AttackSubtype(match.group("subtype")), match.group("split") == "train"


def read_attacks_directory(attacks_csv_dir: Path) -> tuple[PooledAttackFile, ...]:
    """Reads every `*_train.pcap.csv`/`*_test.pcap.csv` file under `attacks_csv_dir` (searched
    recursively, matching the real `train/`/`test/` subdirectory layout). Includes the
    `Benign_train`/`Benign_test` files — they are not attacks, but the same pooled, non-per-
    client shape applies, so the caller must filter on `attack_subtype` if only true attacks are
    wanted.
    """
    files: list[PooledAttackFile] = []
    for path in sorted(attacks_csv_dir.rglob(f"*{_PROFILING_SUFFIX}")):
        subtype, is_train = parse_attack_filename(path.name)
        files.append(
            PooledAttackFile(
                attack_subtype=subtype, is_train_split=is_train, features=_read_numeric_csv(path)
            )
        )
    return tuple(files)
