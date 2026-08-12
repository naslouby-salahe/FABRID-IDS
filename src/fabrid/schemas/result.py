"""Primary per-client result row schema: every roadmap-required field, no hand-entered numbers.

Generated programmatically from persisted score artifacts, allocation runs,
and calibration results — never hand-typed into a manuscript table.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from fabrid.evaluation.record_level import AttackSubtype, ClientId
from fabrid.schemas.allocation import AllocationPolicy


class WeightMode(StrEnum):
    EQUAL_CLIENT = "equal_client"
    DATASET_COUNT_PROXY = "dataset_count_proxy"
    OPERATIONAL_RATE = "operational_rate"


class SolverStatus(StrEnum):
    OPTIMAL = "optimal"
    SOLVER_INVALID = "solver_invalid"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True, slots=True)
class ResultRow:
    experiment_id: str
    dataset_id: str
    seed: int
    budget_id: str
    budget_value: float
    weight_mode: WeightMode
    policy: AllocationPolicy
    client_id: ClientId
    alpha_selected: float
    threshold: float
    calibration_n: int
    nominal_weight: float
    realized_weight: float
    n_benign_test: int
    n_attack_test: int
    attack_subtype: AttackSubtype | None
    true_positive: int
    false_negative: int
    false_positive: int
    true_negative: int
    fpr: float
    tpr: float
    macro_attack_recall: float
    false_alert_count: int
    solver_status: SolverStatus
    solver_objective: float | None
    solver_gap: float | None
    solver_runtime_ms: float | None
    model_sha256: str
    score_sha256: str
    split_sha256: str
    feature_sha256: str
    protocol_sha256: str
    git_commit: str

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}")
        if self.true_positive < 0 or self.false_negative < 0:
            raise ValueError("true_positive/false_negative must be non-negative")
        if self.false_positive < 0 or self.true_negative < 0:
            raise ValueError("false_positive/true_negative must be non-negative")
        if not (0.0 <= self.fpr <= 1.0):
            raise ValueError(f"fpr must be in [0, 1], got {self.fpr}")
        if not (0.0 <= self.tpr <= 1.0):
            raise ValueError(f"tpr must be in [0, 1], got {self.tpr}")
