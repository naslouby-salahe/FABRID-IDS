from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fabrid.config import (
    ApplicationConfig,
    AttackSplit,
    AttackSplitConfig,
    AttackSubtypeId,
    BenignSplit,
    BenignSplitConfig,
    ClientId,
    DatasetCatalog,
    DatasetId,
    EnvironmentText,
    EventCriterionId,
    FeatureCount,
    GateStatus,
    RowCount,
    SourceFileId,
    SourceRowIndex,
    SplitFraction,
    WeightMode,
)
from fabrid.errors import DatasetError


@dataclass(frozen=True, slots=True)
class FeatureMatrix:
    values: np.ndarray

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ValueError(f"feature matrix must be two-dimensional, got {self.values.ndim}")
        if not np.issubdtype(self.values.dtype, np.number):
            raise ValueError("feature matrix must contain numeric values")
        if not np.isfinite(self.values).all():
            raise ValueError("feature matrix must contain only finite values")

    @classmethod
    def from_cached_values(cls, values: np.ndarray) -> FeatureMatrix:
        if values.ndim != 2:
            raise ValueError(f"feature matrix must be two-dimensional, got {values.ndim}")
        if not np.issubdtype(values.dtype, np.number):
            raise ValueError("feature matrix must contain numeric values")
        matrix = object.__new__(cls)
        object.__setattr__(matrix, "values", values)
        return matrix

    @property
    def row_count(self) -> RowCount:
        return self.values.shape[0]

    @property
    def feature_count(self) -> FeatureCount:
        return self.values.shape[1]


@dataclass(frozen=True, slots=True)
class AttackFeatureBlock:
    subtype: AttackSubtypeId
    source_file: SourceFileId
    features: FeatureMatrix


@dataclass(frozen=True, slots=True)
class DeviceDataset:
    client_id: ClientId
    benign_source_file: SourceFileId
    benign: FeatureMatrix
    attacks: tuple[AttackFeatureBlock, ...]

    def __post_init__(self) -> None:
        subtypes = tuple(block.subtype for block in self.attacks)
        if len(set(subtypes)) != len(subtypes):
            raise ValueError("device dataset contains duplicate attack subtypes")
        if any(block.features.feature_count != self.benign.feature_count for block in self.attacks):
            raise ValueError("benign and attack feature matrices must share feature width")

    def attack(self, subtype: AttackSubtypeId) -> AttackFeatureBlock:
        for block in self.attacks:
            if block.subtype == subtype:
                return block
        raise KeyError(subtype)


@dataclass(frozen=True, slots=True)
class ClientPopulation:
    clients: tuple[ClientId, ...]

    def __post_init__(self) -> None:
        if not self.clients:
            raise ValueError("client population must not be empty")
        if len(set(self.clients)) != len(self.clients):
            raise ValueError("client population contains duplicate clients")

    @property
    def size(self) -> RowCount:
        return len(self.clients)


@dataclass(frozen=True, slots=True)
class EventCriterionEvidence:
    criterion: EventCriterionId
    status: GateStatus
    detail: EnvironmentText

    def __post_init__(self) -> None:
        if not self.detail.strip():
            raise ValueError(f"{self.criterion.value} evidence requires a non-empty detail")


@dataclass(frozen=True, slots=True)
class EventProvenanceEvidence:
    criteria: tuple[EventCriterionEvidence, ...]

    def __post_init__(self) -> None:
        expected = tuple(EventCriterionId)
        observed = tuple(item.criterion for item in self.criteria)
        if observed != expected:
            raise ValueError("event provenance evidence must contain every criterion exactly once")

    def status_of(self, criterion: EventCriterionId) -> GateStatus:
        for item in self.criteria:
            if item.criterion is criterion:
                return item.status
        raise KeyError(criterion)

    def detail_of(self, criterion: EventCriterionId) -> EnvironmentText:
        for item in self.criteria:
            if item.criterion is criterion:
                return item.detail
        raise KeyError(criterion)

    def passed(self, criterion: EventCriterionId) -> bool:
        return self.status_of(criterion) is GateStatus.PASS


def resolve_raw_dataset_root(
    raw_data_root: Path,
    dataset_id: DatasetId,
    catalog: DatasetCatalog,
) -> Path:
    dataset_directories = {
        catalog.nbaiot.directory_name,
        catalog.cic_iot_diad.directory_name,
        catalog.gotham.directory_name,
        catalog.ciciomt.directory_name,
    }
    if raw_data_root.name in dataset_directories:
        raise DatasetError(
            f"raw data root {raw_data_root} is the {raw_data_root.name} dataset directory; "
            "pass the repository raw-data root so dataset trees remain siblings",
            path=raw_data_root,
        )
    return raw_data_root / catalog.directory_name(dataset_id)


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    dataset_id: DatasetId
    population: ClientPopulation | None
    weight_evidence_level: WeightMode


@dataclass(frozen=True, slots=True)
class DatasetRegistry:
    specs: tuple[DatasetSpec, ...]

    def __post_init__(self) -> None:
        if not self.specs:
            raise ValueError("dataset registry must contain at least one dataset")
        dataset_ids = tuple(spec.dataset_id for spec in self.specs)
        if len(set(dataset_ids)) != len(dataset_ids):
            raise ValueError("dataset registry contains duplicate dataset identities")

    def spec(self, dataset_id: DatasetId) -> DatasetSpec:
        for spec in self.specs:
            if spec.dataset_id is dataset_id:
                return spec
        raise KeyError(dataset_id)


def event_criterion(
    criterion: EventCriterionId,
    passed: bool,
    detail: EnvironmentText,
) -> EventCriterionEvidence:
    return EventCriterionEvidence(
        criterion=criterion,
        status=GateStatus.PASS if passed else GateStatus.FAIL,
        detail=detail,
    )


def unavailable_event_evidence(detail: EnvironmentText) -> EventProvenanceEvidence:
    return EventProvenanceEvidence(
        criteria=tuple(
            EventCriterionEvidence(
                criterion=criterion,
                status=GateStatus.FAIL,
                detail=detail,
            )
            for criterion in EventCriterionId
        )
    )


def build_dataset_registry(config: ApplicationConfig) -> DatasetRegistry:
    return DatasetRegistry(
        (
            DatasetSpec(
                dataset_id=DatasetId.NBAIOT,
                population=ClientPopulation(config.datasets.nbaiot.devices),
                weight_evidence_level=config.datasets.nbaiot.weight_mode,
            ),
            DatasetSpec(
                dataset_id=DatasetId.CIC_IOT_DIAD,
                population=None,
                weight_evidence_level=config.datasets.cic_iot_diad.weight_mode,
            ),
            DatasetSpec(
                dataset_id=DatasetId.GOTHAM,
                population=None,
                weight_evidence_level=config.datasets.gotham.weight_mode,
            ),
            DatasetSpec(
                dataset_id=DatasetId.CICIOMT,
                population=None,
                weight_evidence_level=config.datasets.ciciomt.weight_mode,
            ),
        )
    )


@dataclass(frozen=True, slots=True)
class BenignSplitCounts:
    train: RowCount
    frontier: RowCount
    final_cal: RowCount
    test: RowCount

    @property
    def total(self) -> RowCount:
        return self.train + self.frontier + self.final_cal + self.test


@dataclass(frozen=True, slots=True)
class AttackSplitCounts:
    validation: RowCount
    test: RowCount

    @property
    def total(self) -> RowCount:
        return self.validation + self.test


@dataclass(frozen=True, slots=True)
class BenignSplitBoundaries:
    train_end: RowCount
    frontier_end: RowCount
    final_cal_end: RowCount
    total_rows: RowCount

    def __post_init__(self) -> None:
        values = (self.train_end, self.frontier_end, self.final_cal_end, self.total_rows)
        if values != tuple(sorted(values)):
            raise ValueError("benign split boundaries must be monotonically increasing")
        if self.total_rows < 0:
            raise ValueError("benign population size may not be negative")

    def split_of(self, row_index: SourceRowIndex) -> BenignSplit:
        if row_index >= self.total_rows:
            raise ValueError(f"row index {row_index} is outside the benign population")
        if row_index < self.train_end:
            return BenignSplit.TRAIN
        if row_index < self.frontier_end:
            return BenignSplit.FRONTIER
        if row_index < self.final_cal_end:
            return BenignSplit.FINAL_CAL
        return BenignSplit.TEST

    def counts(self) -> BenignSplitCounts:
        return BenignSplitCounts(
            train=self.train_end,
            frontier=self.frontier_end - self.train_end,
            final_cal=self.final_cal_end - self.frontier_end,
            test=self.total_rows - self.final_cal_end,
        )


@dataclass(frozen=True, slots=True)
class AttackSplitBoundary:
    validation_end: RowCount
    total_rows: RowCount

    def __post_init__(self) -> None:
        if self.validation_end > self.total_rows:
            raise ValueError("attack validation boundary may not exceed total rows")
        if self.total_rows < 0:
            raise ValueError("attack population size may not be negative")

    def split_of(self, row_index: SourceRowIndex) -> AttackSplit:
        if row_index >= self.total_rows:
            raise ValueError(f"row index {row_index} is outside the attack population")
        if row_index < self.validation_end:
            return AttackSplit.VALIDATION
        return AttackSplit.TEST

    def counts(self) -> AttackSplitCounts:
        return AttackSplitCounts(
            validation=self.validation_end,
            test=self.total_rows - self.validation_end,
        )


@dataclass(frozen=True, slots=True)
class AttackSubtypeBoundary:
    subtype: AttackSubtypeId
    boundary: AttackSplitBoundary


@dataclass(frozen=True, slots=True)
class DeviceSplitPlan:
    benign: BenignSplitBoundaries
    attacks: tuple[AttackSubtypeBoundary, ...]

    def __post_init__(self) -> None:
        subtypes = tuple(item.subtype for item in self.attacks)
        if len(set(subtypes)) != len(subtypes):
            raise ValueError("device split plan contains duplicate attack subtypes")

    def attack_boundary(self, subtype: AttackSubtypeId) -> AttackSplitBoundary:
        for item in self.attacks:
            if item.subtype == subtype:
                return item.boundary
        raise KeyError(subtype)


def compute_benign_split_boundaries(
    total_rows: RowCount,
    train_end: SplitFraction,
    frontier_end: SplitFraction,
    final_cal_end: SplitFraction,
) -> BenignSplitBoundaries:
    return BenignSplitBoundaries(
        train_end=math.floor(train_end * total_rows),
        frontier_end=math.floor(frontier_end * total_rows),
        final_cal_end=math.floor(final_cal_end * total_rows),
        total_rows=total_rows,
    )


def compute_attack_split_boundary(
    total_rows: RowCount,
    validation_end: SplitFraction,
) -> AttackSplitBoundary:
    return AttackSplitBoundary(
        validation_end=math.floor(validation_end * total_rows),
        total_rows=total_rows,
    )


def plan_device_splits(
    device: DeviceDataset,
    benign_splits: BenignSplitConfig,
    attack_split: AttackSplitConfig,
) -> DeviceSplitPlan:
    benign_plan = compute_benign_split_boundaries(
        device.benign.row_count,
        benign_splits.train_end,
        benign_splits.frontier_end,
        benign_splits.final_cal_end,
    )
    return DeviceSplitPlan(
        benign=benign_plan,
        attacks=tuple(
            AttackSubtypeBoundary(
                subtype=block.subtype,
                boundary=compute_attack_split_boundary(
                    block.features.row_count, attack_split.validation_end
                ),
            )
            for block in device.attacks
        ),
    )
