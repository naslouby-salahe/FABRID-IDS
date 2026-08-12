"""Primary result table generation (roadmap section 95). Every builder takes already-computed
typed inputs — never raw file paths or manually-entered numbers — matching TABLE-001..006's
"generated programmatically from artifacts" requirement. Tables whose real input (a completed
confirmatory experiment run, or a chosen result-schema extension) does not exist yet are
intentionally not built here rather than populated with placeholder values.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from fabrid.evaluation.record_level import ClientId, FprDispersion
from fabrid.evaluation.workload import ClientUploadPayload, federation_upload_bytes
from fabrid.experiments.generalization import RotationResult
from fabrid.experiments.main_experiment import SeedBudgetResult
from fabrid.schemas.allocation import AllocationPolicy


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


@dataclass(frozen=True, slots=True)
class MatchedBudgetRow:
    """Table 3 — Matched-budget N-BaIoT."""

    budget: float
    policy: AllocationPolicy
    macro_recall: float
    worst_client_recall: float
    mean_client_fpr: float
    bur: float | None
    max_client_fpr: float
    cv_fpr: float | None


def _mean(values: tuple[float, ...]) -> float:
    return sum(values) / len(values)


def build_table_3_matched_budget(
    results_by_budget_and_policy: Mapping[
        tuple[float, AllocationPolicy], tuple[SeedBudgetResult, ...]
    ],
    fpr_dispersion_by_budget_and_policy: Mapping[tuple[float, AllocationPolicy], FprDispersion],
) -> tuple[MatchedBudgetRow, ...]:
    """One row per (budget, policy), each metric averaged across every seed present in
    `results_by_budget_and_policy` for that cell (seeds where the policy was `SOLVER_INVALID` at
    that budget are simply absent from the corresponding `SeedBudgetResult`s' per-policy dicts —
    the caller controls which seeds/results are passed in; this function does not itself know
    about exclusion). `MeanClientFPR`/`MaxClientFPR`/`CV_FPR` come from a separately-supplied
    `FprDispersion` per cell (built from real per-client FPR values via
    `evaluation.record_level.fpr_dispersion`), since `SeedBudgetResult` only carries the
    federation-aggregate FPR, not the full per-client distribution.
    """
    rows: list[MatchedBudgetRow] = []
    for (budget, policy), results in results_by_budget_and_policy.items():
        macro_recalls = tuple(
            r.macro_recall_by_policy[policy] for r in results if policy in r.macro_recall_by_policy
        )
        worst_recalls = tuple(
            r.worst_client_recall_by_policy[policy]
            for r in results
            if policy in r.worst_client_recall_by_policy
        )
        burs = tuple(
            bur_value for r in results if (bur_value := r.bur_by_policy.get(policy)) is not None
        )
        if not macro_recalls or not worst_recalls:
            raise ValueError(f"no non-excluded seed results for (budget={budget}, policy={policy})")

        dispersion = fpr_dispersion_by_budget_and_policy[(budget, policy)]
        rows.append(
            MatchedBudgetRow(
                budget=budget,
                policy=policy,
                macro_recall=_mean(macro_recalls),
                worst_client_recall=_mean(worst_recalls),
                mean_client_fpr=dispersion.median,
                bur=_mean(burs) if burs else None,
                max_client_fpr=dispersion.maximum,
                cv_fpr=dispersion.coefficient_of_variation,
            )
        )
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class AttackSubtypeDisjointRow:
    """Table 4 — Attack-subtype-disjoint."""

    rotation_label: str
    policy: AllocationPolicy
    macro_recall: float
    worst_client_recall: float
    bur: float


def build_table_4_attack_subtype_disjoint(
    results_by_rotation_and_policy: Mapping[
        tuple[str, AllocationPolicy], tuple[RotationResult, ...]
    ],
) -> tuple[AttackSubtypeDisjointRow, ...]:
    """One row per (rotation, policy), each metric averaged across every seed's
    `RotationResult` present for that cell (a policy `SOLVER_INVALID` at a given seed is simply
    absent from that `RotationResult`'s per-policy dicts, same exclusion convention as Table 3).
    """
    rows: list[AttackSubtypeDisjointRow] = []
    for (rotation_label, policy), results in results_by_rotation_and_policy.items():
        macro_recalls = tuple(
            r.macro_recall_by_policy[policy] for r in results if policy in r.macro_recall_by_policy
        )
        worst_recalls = tuple(
            r.worst_client_recall_by_policy[policy]
            for r in results
            if policy in r.worst_client_recall_by_policy
        )
        burs = tuple(r.bur_by_policy[policy] for r in results if policy in r.bur_by_policy)
        if not macro_recalls or not worst_recalls or not burs:
            raise ValueError(
                f"no non-excluded seed results for (rotation={rotation_label}, policy={policy})"
            )
        rows.append(
            AttackSubtypeDisjointRow(
                rotation_label=rotation_label,
                policy=policy,
                macro_recall=_mean(macro_recalls),
                worst_client_recall=_mean(worst_recalls),
                bur=_mean(burs),
            )
        )
    return tuple(rows)
