from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fabrid.domain.coordinates import (
    AllocationCoordinate,
    DetectorCoordinate,
    ExperimentCoordinate,
    ScoreCoordinate,
)
from fabrid.domain.enums import ArtifactKind, AttackSplit, BenignSplit
from fabrid.domain.identifiers import ArtifactName

ScoreSplit = BenignSplit | AttackSplit


@dataclass(frozen=True, slots=True)
class ArtifactLayout:
    root: Path

    def campaign_root(self, coordinate: ExperimentCoordinate) -> Path:
        return self.root / coordinate.campaign_id.value

    def protocol_path(self, coordinate: ExperimentCoordinate) -> Path:
        return self.campaign_root(coordinate) / "protocol.json"

    def dataset_root(self, coordinate: ExperimentCoordinate) -> Path:
        return self.campaign_root(coordinate) / "datasets" / coordinate.dataset_id.value

    def detector_root(self, coordinate: DetectorCoordinate, campaign_root: Path) -> Path:
        return (
            campaign_root
            / "detectors"
            / coordinate.dataset_id.value
            / f"seed-{coordinate.detector_seed.value:03d}"
        )

    def score_path(
        self,
        coordinate: ScoreCoordinate,
        split: ScoreSplit,
        campaign_root: Path,
    ) -> Path:
        return (
            campaign_root
            / "scores"
            / coordinate.dataset_id.value
            / f"seed-{coordinate.detector_seed.value:03d}"
            / coordinate.client_id.value
            / f"{split.value}.parquet"
        )

    def allocation_path(self, coordinate: AllocationCoordinate) -> Path:
        experiment = coordinate.experiment
        return (
            self.campaign_root(experiment)
            / "allocations"
            / experiment.experiment_id.value
            / f"seed-{experiment.detector_seed.value:03d}"
            / experiment.budget_id.value
            / f"{coordinate.policy.value}.json"
        )

    def result_path(self, coordinate: ExperimentCoordinate) -> Path:
        return (
            self.campaign_root(coordinate)
            / "results"
            / f"{coordinate.experiment_id.value}.parquet"
        )

    def analysis_path(
        self,
        coordinate: ExperimentCoordinate,
        name: ArtifactName,
    ) -> Path:
        return (
            self.campaign_root(coordinate)
            / "analysis"
            / coordinate.experiment_id.value
            / f"{name.value}.parquet"
        )

    def publication_dir(
        self,
        coordinate: ExperimentCoordinate,
        kind: ArtifactKind,
    ) -> Path:
        if kind not in {ArtifactKind.TABLE, ArtifactKind.FIGURE}:
            raise ValueError(f"publication directory does not support {kind.value}")
        return self.campaign_root(coordinate) / "publication" / f"{kind.value}s"

    def audit_dir(self, coordinate: ExperimentCoordinate) -> Path:
        return self.campaign_root(coordinate) / "audit"
