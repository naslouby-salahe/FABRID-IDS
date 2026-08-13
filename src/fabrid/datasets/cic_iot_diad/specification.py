from __future__ import annotations

from fabrid.datasets.external import ExternalDatasetEligibility, ExternalDatasetSpecification
from fabrid.domain.enums import ClientPartitionKind, DatasetId
from fabrid.domain.identifiers import ColumnName
from fabrid.domain.values import Probability, RowCount


SPECIFICATION = ExternalDatasetSpecification(
    dataset_id=DatasetId.CIC_IOT_DIAD,
    partition_kind=ClientPartitionKind.PHYSICAL_DEVICE,
    client_column=ColumnName("src_mac"),
    eligibility=ExternalDatasetEligibility(
        minimum_benign_rows=RowCount(10_000),
        minimum_attack_rows=RowCount(1_000),
        minimum_attack_validation_rows=RowCount(200),
        minimum_eligible_subtypes=RowCount(2),
        minimum_rows_per_eligible_subtype=RowCount(50),
        minimum_confirmatory_clients=RowCount(10),
    ),
    excluded_features=(
        ColumnName("src_mac"),
        ColumnName("device_mac"),
        ColumnName("anomaly_label"),
        ColumnName("device_identification_label"),
        ColumnName("source_row_id"),
        ColumnName("source_file"),
        ColumnName("split_id"),
    ),
    numeric_parse_success_threshold=Probability(0.999),
)
