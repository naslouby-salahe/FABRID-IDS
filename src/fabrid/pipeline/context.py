from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fabrid.artifacts.layout import ArtifactLayout
from fabrid.domain.enums import DatasetId


@dataclass(frozen=True, slots=True)
class PipelinePaths:
    raw_data_root: Path
    outputs_root: Path

    @property
    def artifacts(self) -> ArtifactLayout:
        return ArtifactLayout(self.outputs_root)

    def raw_dataset_root(self, dataset_id: DatasetId) -> Path:
        if dataset_id is DatasetId.NBAIOT:
            return self.raw_data_root / "N-BaIoT"
        if dataset_id is DatasetId.CIC_IOT_DIAD:
            return self.raw_data_root / "CIC-IoT-DIAD2024"
        if dataset_id is DatasetId.GOTHAM:
            return self.raw_data_root / "Gotham"
        if dataset_id is DatasetId.CICIOMT:
            return self.raw_data_root / "CICIoMT2024"
        raise ValueError(f"unsupported dataset {dataset_id.value}")
