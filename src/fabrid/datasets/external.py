from __future__ import annotations

from dataclasses import dataclass

from fabrid.domain.enums import (
    ClientPartitionKind,
    DatasetEligibilityStatus,
    DatasetId,
    ReplicationEvidenceLevel,
)
from fabrid.domain.identifiers import ColumnName
from fabrid.domain.values import Probability, RowCount


@dataclass(frozen=True, slots=True)
class ExternalDatasetEligibility:
    minimum_benign_rows: RowCount
    minimum_attack_rows: RowCount
    minimum_attack_validation_rows: RowCount
    minimum_eligible_subtypes: RowCount
    minimum_rows_per_eligible_subtype: RowCount
    minimum_confirmatory_clients: RowCount


@dataclass(frozen=True, slots=True)
class ExternalDatasetSpecification:
    dataset_id: DatasetId
    partition_kind: ClientPartitionKind
    client_column: ColumnName
    eligibility: ExternalDatasetEligibility
    excluded_features: tuple[ColumnName, ...]
    numeric_parse_success_threshold: Probability

    def __post_init__(self) -> None:
        if len(set(self.excluded_features)) != len(self.excluded_features):
            raise ValueError("external-dataset exclusions contain duplicates")
        if self.client_column not in self.excluded_features:
            raise ValueError("client partition column must be excluded from model features")

    def device_eligibility(
        self,
        benign_rows: RowCount,
        attack_rows: RowCount,
    ) -> DatasetEligibilityStatus:
        if (
            benign_rows.value >= self.eligibility.minimum_benign_rows.value
            and attack_rows.value >= self.eligibility.minimum_attack_rows.value
        ):
            return DatasetEligibilityStatus.ELIGIBLE
        return DatasetEligibilityStatus.INELIGIBLE

    def evidence_level(self, qualifying_clients: RowCount) -> ReplicationEvidenceLevel:
        if qualifying_clients.value >= self.eligibility.minimum_confirmatory_clients.value:
            return ReplicationEvidenceLevel.CONFIRMATORY
        return ReplicationEvidenceLevel.SUPPORTIVE
