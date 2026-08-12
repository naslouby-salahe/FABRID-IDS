"""Primary result table generation (roadmap section 95). Every builder takes already-computed
typed inputs — never raw file paths or manually-entered numbers — matching TABLE-001..006's
"generated programmatically from artifacts" requirement. Tables whose real input (a completed
confirmatory experiment run, or a chosen result-schema extension) does not exist yet are
intentionally not built here rather than populated with placeholder values.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from fabrid.evaluation.record_level import ClientId
from fabrid.evaluation.workload import ClientUploadPayload, federation_upload_bytes


@dataclass(frozen=True, slots=True)
class DatasetPopulationRow:
    """Table 2 — Dataset populations."""

    dataset: str
    natural_clients: int
    eligible_clients: int
    benign_rows: int
    attack_rows: int
    attack_types: int
    physical_or_emulated: str
    timestamp_provenance: str
    weight_evidence_level: str

    def __post_init__(self) -> None:
        if not (0 <= self.eligible_clients <= self.natural_clients):
            raise ValueError(
                "eligible_clients must be in [0, natural_clients], got "
                f"{self.eligible_clients} of {self.natural_clients}"
            )
        if self.benign_rows < 0 or self.attack_rows < 0:
            raise ValueError(
                f"row counts must be non-negative, got benign={self.benign_rows}, "
                f"attack={self.attack_rows}"
            )


def build_table_2_dataset_populations(
    rows: tuple[DatasetPopulationRow, ...],
) -> tuple[DatasetPopulationRow, ...]:
    """Identity pass-through that validates the full table shape (non-empty, no duplicate
    dataset names) — the actual per-dataset values are computed by the caller from real
    `datasets.yaml` counts and persisted `ScoreArtifact` row counts, not invented here.
    """
    if not rows:
        raise ValueError("build_table_2_dataset_populations requires at least one row")
    dataset_names = [row.dataset for row in rows]
    if len(dataset_names) != len(set(dataset_names)):
        raise ValueError(f"duplicate dataset names in Table 2: {dataset_names}")
    return rows


@dataclass(frozen=True, slots=True)
class SystemOverheadRow:
    """Table 6 — System overhead."""

    client_count: int
    utility_payload_bytes: int
    serialized_upload_bytes: int
    allocation_runtime_seconds: float
    peak_memory_bytes: int
    response_bytes: int

    def __post_init__(self) -> None:
        if self.client_count < 1:
            raise ValueError(f"client_count must be positive, got {self.client_count}")
        if self.allocation_runtime_seconds < 0:
            raise ValueError(
                f"allocation_runtime_seconds must be non-negative, got "
                f"{self.allocation_runtime_seconds}"
            )
        if self.peak_memory_bytes < 0:
            raise ValueError(
                f"peak_memory_bytes must be non-negative, got {self.peak_memory_bytes}"
            )


def build_table_6_system_overhead(
    client_counts: tuple[int, ...],
    alpha_grid_size: int,
    measured_allocation_runtime_seconds: Mapping[int, float],
    measured_peak_memory_bytes: Mapping[int, int],
    measured_response_bytes: Mapping[int, int],
) -> tuple[SystemOverheadRow, ...]:
    """`utility_payload_bytes`/`serialized_upload_bytes` (federation total, pre-serialization)
    are computed directly from the fixed message layout (`evaluation/workload.py`) for each `K`
    in `client_counts`. `allocation_runtime_seconds`/`peak_memory_bytes`/`response_bytes` are
    real measurements the caller must supply per `K` (instrumentation, not something this
    module can derive) — a `K` missing from any of the three mappings raises rather than
    silently reporting zero.
    """
    payload = ClientUploadPayload(alpha_grid_size=alpha_grid_size)
    rows: list[SystemOverheadRow] = []
    for client_count in client_counts:
        if client_count not in measured_allocation_runtime_seconds:
            raise KeyError(f"no measured allocation runtime for K={client_count}")
        if client_count not in measured_peak_memory_bytes:
            raise KeyError(f"no measured peak memory for K={client_count}")
        if client_count not in measured_response_bytes:
            raise KeyError(f"no measured response bytes for K={client_count}")
        rows.append(
            SystemOverheadRow(
                client_count=client_count,
                utility_payload_bytes=payload.total_bytes(),
                serialized_upload_bytes=federation_upload_bytes(payload, client_count),
                allocation_runtime_seconds=measured_allocation_runtime_seconds[client_count],
                peak_memory_bytes=measured_peak_memory_bytes[client_count],
                response_bytes=measured_response_bytes[client_count],
            )
        )
    return tuple(rows)


def dataset_population_row_from_score_artifacts(
    dataset: str,
    eligible_clients: int,
    benign_row_counts: Mapping[ClientId, int],
    attack_row_counts: Mapping[ClientId, int],
    attack_types: int,
    physical_or_emulated: str,
    timestamp_provenance: str,
    weight_evidence_level: str,
) -> DatasetPopulationRow:
    """Sums real per-client row counts (e.g. from persisted `ScoreArtifact`s) into one Table 2
    row, instead of requiring the caller to sum them by hand.
    """
    return DatasetPopulationRow(
        dataset=dataset,
        natural_clients=len(benign_row_counts),
        eligible_clients=eligible_clients,
        benign_rows=sum(benign_row_counts.values()),
        attack_rows=sum(attack_row_counts.values()),
        attack_types=attack_types,
        physical_or_emulated=physical_or_emulated,
        timestamp_provenance=timestamp_provenance,
        weight_evidence_level=weight_evidence_level,
    )
