from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import pytest
from pydantic import BaseModel, ConfigDict

from fabrid.artifacts.parquet import read_parquet_models, write_parquet_models


@dataclass(frozen=True, slots=True)
class HybridModel(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    client_id: str
    logical_bytes: int


def test_write_parquet_models_skips_classvar_annotations(tmp_path: Path) -> None:
    records = (
        HybridModel(client_id="dev-1", logical_bytes=1000),
        HybridModel(client_id="dev-2", logical_bytes=2000),
    )
    path = tmp_path / "payload.parquet"
    write_parquet_models(path, records)
    restored = read_parquet_models(path, HybridModel)
    assert restored == records


def test_write_parquet_models_rejects_empty(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty parquet artifact"):
        write_parquet_models(tmp_path / "empty.parquet", ())
