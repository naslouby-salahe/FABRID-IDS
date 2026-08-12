from __future__ import annotations

from fabrid.config.external_datasets import (
    ExternalDatasetEligibility,
    device_is_eligible,
    load_external_dataset_config,
)


def test_load_external_dataset_config_reads_cic_iot_diad_section() -> None:
    config = load_external_dataset_config("cic_iot_diad_2024")

    assert config.dataset_id == "cic-iot-diad-2024"
    assert config.client_key == "device_mac"
    assert config.eligibility.min_benign_rows == 10000
    assert config.eligibility.min_attack_rows == 1000
    assert config.eligibility.min_qualifying_clients_for_confirmatory == 10
    assert "device_mac" in config.excluded_features
    assert "anomaly_label" in config.excluded_features
    assert config.numeric_parse_success_threshold == 0.999


def test_device_is_eligible_requires_both_thresholds() -> None:
    eligibility = ExternalDatasetEligibility(
        min_benign_rows=10000,
        min_attack_rows=1000,
        min_attack_validation_rows=200,
        min_eligible_subtypes=2,
        min_rows_per_eligible_subtype=50,
        min_qualifying_clients_for_confirmatory=10,
    )
    assert device_is_eligible(10000, 1000, eligibility)
    assert not device_is_eligible(9999, 1000, eligibility)
    assert not device_is_eligible(10000, 999, eligibility)
