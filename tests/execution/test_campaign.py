from __future__ import annotations

from pathlib import Path

from fabrid.config import ExperimentId
from fabrid.execution.application import ApplicationContext
from fabrid.execution.campaign import primary_seed_experiments
from tests.support import isolated_paths, smoke_application


def test_primary_seed_experiments_excludes_global_external_and_event_branches(
    tmp_path: Path,
) -> None:
    application = smoke_application()
    context = ApplicationContext(
        config=application,
        paths=isolated_paths(tmp_path),
        repository_root=tmp_path,
    )
    pending = frozenset(
        (
            ExperimentId.MATCHED_BUDGET,
            ExperimentId.EXTERNAL_REPLICATION,
            ExperimentId.EVENT_LEVEL,
        )
    )
    assert primary_seed_experiments(context, pending) == frozenset((ExperimentId.MATCHED_BUDGET,))
