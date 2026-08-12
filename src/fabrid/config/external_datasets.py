"""Typed loader for external-replication-candidate dataset contracts (`datasets.yaml`'s
`cic_iot_diad_2024`/`cic_iomt_2024`-style sections): eligibility thresholds, the leakage
exclusion list, and the numeric-parse-success threshold, all frozen but previously unread by
any code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fabrid.config.attack_folds import DATASETS_PATH
from fabrid.config.protocol import read_yaml_mapping


@dataclass(frozen=True, slots=True)
class ExternalDatasetEligibility:
    min_benign_rows: int
    min_attack_rows: int
    min_attack_validation_rows: int
    min_eligible_subtypes: int
    min_rows_per_eligible_subtype: int
    min_qualifying_clients_for_confirmatory: int

    def __post_init__(self) -> None:
        for name, value in (
            ("min_benign_rows", self.min_benign_rows),
            ("min_attack_rows", self.min_attack_rows),
            ("min_attack_validation_rows", self.min_attack_validation_rows),
            ("min_eligible_subtypes", self.min_eligible_subtypes),
            ("min_rows_per_eligible_subtype", self.min_rows_per_eligible_subtype),
            (
                "min_qualifying_clients_for_confirmatory",
                self.min_qualifying_clients_for_confirmatory,
            ),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")


@dataclass(frozen=True, slots=True)
class ExternalDatasetConfig:
    dataset_id: str
    client_key: str
    eligibility: ExternalDatasetEligibility
    excluded_features: frozenset[str]
    numeric_parse_success_threshold: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.numeric_parse_success_threshold <= 1.0):
            raise ValueError(
                "numeric_parse_success_threshold must be in [0, 1], got "
                f"{self.numeric_parse_success_threshold}"
            )


def load_external_dataset_config(
    dataset_key: str, path: Path = DATASETS_PATH
) -> ExternalDatasetConfig:
    payload = read_yaml_mapping(path)
    raw = payload[dataset_key]
    eligibility_raw = raw["eligibility"]
    return ExternalDatasetConfig(
        dataset_id=str(raw["dataset_id"]),
        client_key=str(raw["client_key"]),
        eligibility=ExternalDatasetEligibility(
            min_benign_rows=int(eligibility_raw["min_benign_rows"]),
            min_attack_rows=int(eligibility_raw["min_attack_rows"]),
            min_attack_validation_rows=int(eligibility_raw["min_attack_validation_rows"]),
            min_eligible_subtypes=int(eligibility_raw["min_eligible_subtypes"]),
            min_rows_per_eligible_subtype=int(eligibility_raw["min_rows_per_eligible_subtype"]),
            min_qualifying_clients_for_confirmatory=int(
                eligibility_raw["min_qualifying_clients_for_confirmatory"]
            ),
        ),
        excluded_features=frozenset(str(f) for f in raw["excluded_features"]),
        numeric_parse_success_threshold=float(raw["numeric_parse_success_threshold"]),
    )


def device_is_eligible(
    benign_row_count: int, attack_row_count: int, eligibility: ExternalDatasetEligibility
) -> bool:
    return (
        benign_row_count >= eligibility.min_benign_rows
        and attack_row_count >= eligibility.min_attack_rows
    )
