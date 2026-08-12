from __future__ import annotations

import pytest

from fabrid.evaluation.record_level import ClientId
from fabrid.reporting.tables import (
    DatasetPopulationRow,
    SystemOverheadRow,
    build_table_2_dataset_populations,
    build_table_6_system_overhead,
    dataset_population_row_from_score_artifacts,
)

_ALPHA_GRID_SIZE = 207


def _population_row(dataset: str = "n-baiot") -> DatasetPopulationRow:
    return DatasetPopulationRow(
        dataset=dataset,
        natural_clients=9,
        eligible_clients=9,
        benign_rows=555_932,
        attack_rows=6_506_674,
        attack_types=10,
        physical_or_emulated="Physical",
        timestamp_provenance="Source-order only, no wall-clock timestamps",
        weight_evidence_level="C",
    )


def test_dataset_population_row_rejects_eligible_exceeding_natural() -> None:
    with pytest.raises(ValueError, match="eligible_clients"):
        DatasetPopulationRow(
            dataset="x",
            natural_clients=5,
            eligible_clients=6,
            benign_rows=100,
            attack_rows=100,
            attack_types=1,
            physical_or_emulated="Physical",
            timestamp_provenance="",
            weight_evidence_level="C",
        )


def test_build_table_2_requires_at_least_one_row() -> None:
    with pytest.raises(ValueError, match="at least one row"):
        build_table_2_dataset_populations(())


def test_build_table_2_rejects_duplicate_dataset_names() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        build_table_2_dataset_populations((_population_row("n-baiot"), _population_row("n-baiot")))


def test_build_table_2_passes_through_valid_rows() -> None:
    rows = (_population_row("n-baiot"), _population_row("cic-iot-diad-2024"))
    result = build_table_2_dataset_populations(rows)
    assert result == rows


def test_dataset_population_row_from_score_artifacts_sums_real_counts() -> None:
    benign = {ClientId("1"): 100, ClientId("2"): 200}
    attack = {ClientId("1"): 50, ClientId("2"): 75}
    row = dataset_population_row_from_score_artifacts(
        dataset="n-baiot",
        eligible_clients=2,
        benign_row_counts=benign,
        attack_row_counts=attack,
        attack_types=10,
        physical_or_emulated="Physical",
        timestamp_provenance="Source-order only",
        weight_evidence_level="C",
    )
    assert row.natural_clients == 2
    assert row.benign_rows == 300
    assert row.attack_rows == 125


def test_system_overhead_row_rejects_negative_runtime() -> None:
    with pytest.raises(ValueError, match="allocation_runtime_seconds"):
        SystemOverheadRow(
            client_count=9,
            utility_payload_bytes=896,
            serialized_upload_bytes=8064,
            allocation_runtime_seconds=-1.0,
            peak_memory_bytes=0,
            response_bytes=0,
        )


def test_build_table_6_computes_payload_bytes_and_uses_measured_values() -> None:
    result = build_table_6_system_overhead(
        client_counts=(9, 105),
        alpha_grid_size=_ALPHA_GRID_SIZE,
        measured_allocation_runtime_seconds={9: 0.05, 105: 0.4},
        measured_peak_memory_bytes={9: 1_000_000, 105: 5_000_000},
        measured_response_bytes={9: 90, 105: 1050},
    )
    assert len(result) == 2
    row_9 = result[0]
    assert row_9.client_count == 9
    assert row_9.utility_payload_bytes == 896
    assert row_9.serialized_upload_bytes == 8064
    assert row_9.allocation_runtime_seconds == pytest.approx(0.05)
    row_105 = result[1]
    assert row_105.utility_payload_bytes == 896
    assert row_105.serialized_upload_bytes == 94080


def test_build_table_6_missing_measurement_raises() -> None:
    with pytest.raises(KeyError):
        build_table_6_system_overhead(
            client_counts=(9,),
            alpha_grid_size=_ALPHA_GRID_SIZE,
            measured_allocation_runtime_seconds={},
            measured_peak_memory_bytes={9: 1},
            measured_response_bytes={9: 1},
        )
