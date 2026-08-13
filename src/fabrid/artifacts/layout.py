from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fabrid.domain.coordinates import (
    AllocationCoordinate,
    DetectorCoordinate,
    ExperimentCoordinate,
    ScoreCoordinate,
)
from fabrid.domain.enums import ArtifactKind, AttackSplit, BenignSplit, DatasetId
from fabrid.domain.identifiers import ArtifactName, CampaignId

ScoreSplit = BenignSplit | AttackSplit


@dataclass(frozen=True, slots=True)
class ArtifactLayout:
    root: Path

    def campaign_root(self, campaign_id: CampaignId) -> Path:
        return self.root / campaign_id.value

    def protocol_path(self, campaign_id: CampaignId) -> Path:
        return self.campaign_root(campaign_id) / "protocol.json"

    def dataset_root(self, campaign_id: CampaignId, dataset_id: DatasetId) -> Path:
        return self.campaign_root(campaign_id) / "datasets" / dataset_id.value

    def split_manifest_path(
        self,
        campaign_id: CampaignId,
        dataset_id: DatasetId,
    ) -> Path:
        return self.dataset_root(campaign_id, dataset_id) / "splits.json"

    def feature_manifest_path(
        self,
        campaign_id: CampaignId,
        dataset_id: DatasetId,
    ) -> Path:
        return self.dataset_root(campaign_id, dataset_id) / "features.json"

    def detector_root(
        self,
        campaign_id: CampaignId,
        coordinate: DetectorCoordinate,
    ) -> Path:
        return (
            self.campaign_root(campaign_id)
            / "detectors"
            / coordinate.dataset_id.value
            / f"seed-{coordinate.detector_seed.value:03d}"
        )

    def score_path(
        self,
        campaign_id: CampaignId,
        coordinate: ScoreCoordinate,
        split: ScoreSplit,
    ) -> Path:
        return (
            self.campaign_root(campaign_id)
            / "scores"
            / coordinate.dataset_id.value
            / f"seed-{coordinate.detector_seed.value:03d}"
            / coordinate.client_id.value
            / f"{split.value}.parquet"
        )

    def experiment_root(self, coordinate: ExperimentCoordinate) -> Path:
        return (
            self.campaign_root(coordinate.campaign_id)
            / "experiments"
            / coordinate.experiment_id.value
            / coordinate.variant_id.value
            / coordinate.dataset_id.value
            / f"seed-{coordinate.detector_seed.value:03d}"
            / coordinate.budget_id.value
            / coordinate.weight_mode.value
        )

    def allocation_path(self, coordinate: AllocationCoordinate) -> Path:
        return (
            self.experiment_root(coordinate.experiment)
            / "allocations"
            / f"{coordinate.policy.value}.json"
        )

    def result_path(self, coordinate: ExperimentCoordinate) -> Path:
        return self.experiment_root(coordinate) / "results.parquet"

    def evaluation_summary_path(self, coordinate: ExperimentCoordinate) -> Path:
        return self.experiment_root(coordinate) / "evaluation.json"

    def analysis_path(
        self,
        coordinate: ExperimentCoordinate,
        name: ArtifactName,
    ) -> Path:
        return self.experiment_root(coordinate) / "analysis" / f"{name.value}.parquet"

    def publication_dir(
        self,
        coordinate: ExperimentCoordinate,
        kind: ArtifactKind,
    ) -> Path:
        if kind not in {ArtifactKind.TABLE, ArtifactKind.FIGURE}:
            raise ValueError(f"publication directory does not support {kind.value}")
        return self.experiment_root(coordinate) / "publication" / f"{kind.value}s"

    def audit_dir(self, campaign_id: CampaignId) -> Path:
        return self.campaign_root(campaign_id) / "audit"
