from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from typing import TYPE_CHECKING, cast

import numpy as np
from scipy import optimize
from scipy.optimize import Bounds, LinearConstraint

from fabrid.allocation.problem import Allocation
from fabrid.config import (
    MIP_FEASIBILITY_TOLERANCE,
    AllocationPolicy,
    CandidateCount,
    ClientCount,
    ClientId,
    FalsePositiveBudget,
    SolverConfig,
    SolverGap,
    SolverObjective,
    SolverRuntimeMilliseconds,
    TargetFalsePositiveRate,
)
from fabrid.errors import SolverInvalidError

if TYPE_CHECKING:
    from scipy.optimize._milp import _OptionsMILP as _MilpOptions
else:
    _MilpOptions = dict


class SolverStatus(StrEnum):
    OPTIMAL = "optimal"
    SOLVER_INVALID = "solver_invalid"
    NOT_APPLICABLE = "not_applicable"


class SolverStage(StrEnum):
    MACRO_PRIMARY_UTILITY = "macro_primary_utility"
    MACRO_MINIMUM_BUDGET = "macro_minimum_budget"
    MACRO_LOWEST_VARIANCE = "macro_lowest_variance"
    MACRO_TIE_BREAK = "macro_tie_break"
    CVAR_UTILITY = "cvar_utility"
    CVAR_MEAN_UTILITY = "cvar_mean_utility"
    CVAR_MINIMUM_BUDGET = "cvar_minimum_budget"
    CVAR_LOWEST_VARIANCE = "cvar_lowest_variance"
    CVAR_TIE_BREAK = "cvar_tie_break"


class MilpStatusCode(IntEnum):
    OPTIMAL = 0
    ITERATION_LIMIT = 1
    INFEASIBLE = 2
    UNBOUNDED = 3
    OTHER = 4


@dataclass(frozen=True, slots=True)
class SolverStageEvidence:
    stage: SolverStage
    objective: SolverObjective
    gap: SolverGap
    runtime_ms: SolverRuntimeMilliseconds


@dataclass(frozen=True, slots=True)
class SolverEvidence:
    status: SolverStatus
    stages: tuple[SolverStageEvidence, ...]

    def __post_init__(self) -> None:
        if self.status is SolverStatus.OPTIMAL and not self.stages:
            raise ValueError("optimal solver evidence requires at least one solver stage")
        if self.status is not SolverStatus.OPTIMAL and self.stages:
            raise ValueError("non-optimal solver evidence may not carry accepted solver stages")
        stage_ids = tuple(stage.stage for stage in self.stages)
        if len(set(stage_ids)) != len(stage_ids):
            raise ValueError("solver evidence contains duplicate stages")

    @property
    def final_objective(self) -> SolverObjective | None:
        return None if not self.stages else self.stages[-1].objective

    @property
    def final_gap(self) -> SolverGap | None:
        return None if not self.stages else self.stages[-1].gap

    @property
    def total_runtime(self) -> SolverRuntimeMilliseconds | None:
        if not self.stages:
            return None
        return sum(stage.runtime_ms for stage in self.stages)


@dataclass(frozen=True, slots=True)
class AllocationDecisionSnapshot:
    client_id: ClientId
    target_rate: TargetFalsePositiveRate


@dataclass(frozen=True, slots=True)
class AllocationSnapshot:
    policy: AllocationPolicy
    decisions: tuple[AllocationDecisionSnapshot, ...]
    solver: SolverEvidence


@dataclass(frozen=True, slots=True)
class MilpSolution:
    variables: np.ndarray
    evidence: SolverStageEvidence


@dataclass(frozen=True, slots=True)
class OptimizedAllocation:
    allocation: Allocation
    solver: SolverEvidence

    def __post_init__(self) -> None:
        if self.solver.status is not SolverStatus.OPTIMAL:
            raise ValueError("optimized allocation requires optimal solver evidence")


def not_applicable_solver_evidence() -> SolverEvidence:
    return SolverEvidence(status=SolverStatus.NOT_APPLICABLE, stages=())


def solve_milp(
    stage: SolverStage,
    minimize_coefficients: np.ndarray,
    constraints: tuple[LinearConstraint, ...],
    integrality: np.ndarray,
    bounds: Bounds,
    settings: SolverConfig,
) -> MilpSolution:
    started = time.perf_counter()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Unrecognized options detected",
            category=RuntimeWarning,
        )
        result = optimize.milp(
            minimize_coefficients,
            constraints=constraints,
            integrality=integrality,
            bounds=bounds,
            options=cast(
                _MilpOptions,
                {
                    "output_flag": False,
                    "mip_rel_gap": settings.requested_gap,
                    "time_limit": settings.time_limit_seconds,
                    "mip_feasibility_tolerance": MIP_FEASIBILITY_TOLERANCE,
                },
            ),
        )
    runtime_ms = (time.perf_counter() - started) * 1_000.0
    solved = bool(result.success)
    status = MilpStatusCode(int(result.status))
    objective = result.fun
    mip_gap = result.mip_gap
    variables = result.x
    message = result.message
    if objective is None or mip_gap is None:
        raise SolverInvalidError(
            f"SOLVER_INVALID: success={solved}, status={status.name}, "
            f"objective={objective}, mip_gap={mip_gap}, message={message}"
        )
    accepted = (
        solved
        and status is MilpStatusCode.OPTIMAL
        and (
            mip_gap <= settings.accepted_gap
            or mip_gap * abs(objective) <= settings.accepted_absolute_gap
        )
    )
    if not accepted or variables is None:
        raise SolverInvalidError(
            f"SOLVER_INVALID: success={solved}, status={status.name}, "
            f"objective={objective}, mip_gap={mip_gap}, message={message}"
        )
    return MilpSolution(
        variables=np.asarray(variables),
        evidence=SolverStageEvidence(
            stage=stage,
            objective=objective,
            gap=mip_gap,
            runtime_ms=runtime_ms,
        ),
    )


def one_hot_constraint(
    client_count: ClientCount, candidate_count: CandidateCount
) -> LinearConstraint:
    matrix = np.zeros((client_count, client_count * candidate_count))
    for client_index in range(client_count):
        start = client_index * candidate_count
        matrix[client_index, start : start + candidate_count] = 1.0
    return LinearConstraint(matrix, lb=1.0, ub=1.0)


def budget_constraint(cost: np.ndarray, budget: FalsePositiveBudget) -> LinearConstraint:
    return LinearConstraint(cost.reshape(1, -1), lb=-np.inf, ub=budget)


def _constant_bound(bound: np.ndarray) -> np.float64:
    values = np.asarray(bound, dtype=np.float64).reshape(-1)
    if values.size == 0 or not bool(np.all(values == values[0])):
        raise ValueError("cannot pad a constraint with non-constant bounds")
    return values[0]


def pad_constraint_columns(
    constraint: LinearConstraint,
    matrix: np.ndarray,
    extra_columns: CandidateCount,
) -> LinearConstraint:
    padded = np.hstack([matrix, np.zeros((matrix.shape[0], extra_columns))])
    return LinearConstraint(
        padded,
        lb=_constant_bound(constraint.lb),
        ub=_constant_bound(constraint.ub),
    )


def lexicographic_weights(client_count: ClientCount, candidate_count: CandidateCount) -> np.ndarray:
    per_client_scale = np.power(
        float(candidate_count + 1),
        -np.arange(client_count, dtype=np.float64),
    )
    return np.repeat(per_client_scale, candidate_count)
