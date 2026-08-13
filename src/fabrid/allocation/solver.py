from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from fabrid.domain.identifiers import FailureReason
from fabrid.domain.values import SolverGap, SolverObjective, SolverRuntimeMilliseconds
from fabrid.protocol.models import SolverSettings

_OPTIMAL_STATUS = 0
_HIGHS_FEASIBILITY_TOLERANCE = 1e-9


class SolverInvalidError(Exception):
    def __init__(self, reason: FailureReason) -> None:
        super().__init__(reason.value)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class MilpSolution:
    variables: np.ndarray
    objective: SolverObjective
    gap: SolverGap
    runtime: SolverRuntimeMilliseconds


def solve_milp(
    minimize_coefficients: np.ndarray,
    constraints: tuple[LinearConstraint, ...],
    integrality: np.ndarray,
    bounds: Bounds,
    settings: SolverSettings,
) -> MilpSolution:
    started = time.perf_counter()
    result = milp(
        minimize_coefficients,
        constraints=constraints,
        integrality=integrality,
        bounds=bounds,
        options={  # pyright: ignore[reportArgumentType]
            "mip_rel_gap": settings.requested_gap.value,
            "time_limit": settings.time_limit.value,
            "mip_feasibility_tolerance": _HIGHS_FEASIBILITY_TOLERANCE,
            "primal_feasibility_tolerance": _HIGHS_FEASIBILITY_TOLERANCE,
        },
    )
    runtime = SolverRuntimeMilliseconds((time.perf_counter() - started) * 1_000.0)

    objective_value = result.fun
    mip_gap = result.mip_gap
    accepted = (
        bool(result.success)
        and result.status == _OPTIMAL_STATUS
        and objective_value is not None
        and mip_gap is not None
        and mip_gap <= settings.accepted_gap.value
    )
    if not accepted or objective_value is None or mip_gap is None or result.x is None:
        raise SolverInvalidError(
            FailureReason(
                "SOLVER_INVALID: "
                f"success={result.success}, status={result.status}, "
                f"mip_gap={mip_gap}, message={result.message}"
            )
        )

    return MilpSolution(
        variables=np.asarray(result.x),
        objective=SolverObjective(float(objective_value)),
        gap=SolverGap(float(mip_gap)),
        runtime=runtime,
    )
