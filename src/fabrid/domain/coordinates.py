from __future__ import annotations

from dataclasses import dataclass

from fabrid.domain.enums import AllocationPolicy, DatasetId, ExperimentId, WeightMode
from fabrid.domain.identifiers import BudgetId, CampaignId, ClientId
from fabrid.domain.values import DetectorSeed, FalsePositiveBudget


@dataclass(frozen=True, slots=True)
class DetectorCoordinate:
    dataset_id: DatasetId
    detector_seed: DetectorSeed


@dataclass(frozen=True, slots=True)
class ScoreCoordinate:
    dataset_id: DatasetId
    detector_seed: DetectorSeed
    client_id: ClientId


@dataclass(frozen=True, slots=True)
class ExperimentCoordinate:
    campaign_id: CampaignId
    experiment_id: ExperimentId
    dataset_id: DatasetId
    detector_seed: DetectorSeed
    budget_id: BudgetId
    budget: FalsePositiveBudget
    weight_mode: WeightMode


@dataclass(frozen=True, slots=True)
class AllocationCoordinate:
    experiment: ExperimentCoordinate
    policy: AllocationPolicy
