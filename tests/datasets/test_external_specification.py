from __future__ import annotations

from fabrid.datasets.cic_iot_diad.specification import SPECIFICATION
from fabrid.domain.enums import DatasetEligibilityStatus, ReplicationEvidenceLevel
from fabrid.domain.identifiers import ColumnName
from fabrid.domain.values import RowCount


def test_cic_iot_diad_replication_contract_matches_roadmap() -> None:
    eligibility = SPECIFICATION.eligibility

    assert eligibility.minimum_benign_rows == RowCount(10_000)
    assert eligibility.minimum_attack_rows == RowCount(1_000)
    assert eligibility.minimum_attack_validation_rows == RowCount(200)
    assert eligibility.minimum_eligible_subtypes == RowCount(2)
    assert eligibility.minimum_rows_per_eligible_subtype == RowCount(50)
    assert eligibility.minimum_confirmatory_clients == RowCount(10)
    assert SPECIFICATION.numeric_parse_success_threshold.value == 0.999
    assert SPECIFICATION.client_column == ColumnName("src_mac")
    assert SPECIFICATION.client_column in SPECIFICATION.excluded_features


def test_external_device_eligibility_requires_benign_and_attack_minima() -> None:
    assert SPECIFICATION.device_eligibility(
        benign_rows=RowCount(10_000),
        attack_rows=RowCount(1_000),
    ) is DatasetEligibilityStatus.ELIGIBLE
    assert SPECIFICATION.device_eligibility(
        benign_rows=RowCount(9_999),
        attack_rows=RowCount(1_000),
    ) is DatasetEligibilityStatus.INELIGIBLE
    assert SPECIFICATION.device_eligibility(
        benign_rows=RowCount(10_000),
        attack_rows=RowCount(999),
    ) is DatasetEligibilityStatus.INELIGIBLE


def test_external_replication_evidence_downgrades_below_ten_clients() -> None:
    assert SPECIFICATION.evidence_level(RowCount(10)) is ReplicationEvidenceLevel.CONFIRMATORY
    assert SPECIFICATION.evidence_level(RowCount(9)) is ReplicationEvidenceLevel.SUPPORTIVE
