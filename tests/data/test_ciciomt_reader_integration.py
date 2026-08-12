"""Integration check against the real downloaded CICIoMT2024 CSVs.

Skipped automatically when the shared raw-data symlink is not present.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fabrid.data.ciciomt_reader import read_attacks_directory, read_profiling_directory
from fabrid.evaluation.record_level import AttackSubtype

_BASE = Path(__file__).parents[2] / "data" / "raw" / "CICIoMT2024" / "WiFI_and_MQTT"
_PROFILING_DIR = _BASE / "profiling" / "CSV"
_ATTACKS_DIR = _BASE / "attacks" / "CSV"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _PROFILING_DIR.exists(), reason="raw CICIoMT2024 data not present"),
]


def test_real_profiling_directory_reads_all_sessions() -> None:
    sessions = read_profiling_directory(_PROFILING_DIR)
    assert len(sessions) > 1
    for matrix in sessions.values():
        assert matrix.shape[0] > 0
        assert matrix.shape[1] > 0


def test_real_attacks_directory_includes_benign_and_attack_subtypes() -> None:
    files = read_attacks_directory(_ATTACKS_DIR)
    assert len(files) > 1
    subtypes = {f.attack_subtype for f in files}
    assert AttackSubtype("Benign") in subtypes
    assert any(s != AttackSubtype("Benign") for s in subtypes)
    assert any(f.is_train_split for f in files)
    assert any(not f.is_train_split for f in files)
