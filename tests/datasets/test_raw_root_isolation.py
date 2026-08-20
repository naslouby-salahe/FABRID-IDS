from __future__ import annotations

from pathlib import Path

import pytest

from fabrid.config import DatasetId
from fabrid.datasets.registry import resolve_raw_dataset_root
from fabrid.errors import DatasetError
from fabrid.execution.prerequisites import prepare_federation
from tests.support import isolated_context, smoke_application


def test_resolve_raw_dataset_root_rejects_cic_directory_as_nbaiot_parent() -> None:
    application = smoke_application()
    catalog = application.datasets
    cic_root = Path("/data/raw") / catalog.cic_iot_diad.directory_name
    with pytest.raises(DatasetError, match="dataset directory"):
        resolve_raw_dataset_root(cic_root, DatasetId.NBAIOT, catalog)


def test_prepare_federation_rejects_cic_root(tmp_path: Path) -> None:
    application = smoke_application()
    cic_root = tmp_path / application.datasets.cic_iot_diad.directory_name
    cic_root.mkdir()
    context = isolated_context(application, tmp_path, raw_data_root=cic_root)
    with pytest.raises(DatasetError, match="dataset directory"):
        prepare_federation(context)
