from __future__ import annotations

import logging
from pathlib import Path

from pydantic import TypeAdapter

from fabrid.artifacts.json import read_typed_json
from fabrid.validation.reproducibility import ProgressState, report_progress
from tests.support import isolated_paths


def test_report_progress_logs_tick_and_writes_state(tmp_path: Path) -> None:
    paths = isolated_paths(tmp_path)
    records: list[logging.LogRecord] = []

    class _Handler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    logger = logging.getLogger("test-progress")
    logger.handlers = [_Handler()]
    logger.setLevel(logging.INFO)
    logger.propagate = False
    report_progress(logger, paths, "event tables", 3, 8, detail="seed 3")
    state_path = paths.progress_state_path()
    assert state_path.is_file()
    state = read_typed_json(state_path, TypeAdapter(ProgressState))
    assert state.phase == "event tables"
    assert state.completed == 3
    assert state.total == 8
    assert state.detail == "seed 3"
    assert state.updated_at
    message = records[0].getMessage()
    assert "[PROGRESS]" in message
    assert "3/8" in message
    assert "37.5%" in message
    report_progress(logger, paths, "event tables", 8, 8)
    state = read_typed_json(state_path, TypeAdapter(ProgressState))
    assert state.completed == 8
    assert state.detail == "in progress"
