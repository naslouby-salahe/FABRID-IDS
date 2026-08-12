"""Thin, strict wrapper around `scipy.optimize.milp`.

A solve is accepted only if the solver reports success, optimal status, and a
MIP relative gap within tolerance. Anything else is a rejected solve — never
silently treated as an optimum.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from fabrid.config.protocol import SolverSettings

_OPTIMAL_STATUS = 0
_BINARY_LOWER_BOUND = 0.0
_BINARY_UPPER_BOUND = 1.0


class SolverInvalidError(Exception):
    """Raised when the MILP solve does not meet the accept criteria (`SOLVER_INVALID`)."""


@dataclass(frozen=True, slots=True)
class MilpSolution:
    x: np.ndarray
    objective_value: float
    mip_gap: float


def solve_milp(
    minimize_coefficients: np.ndarray,
    constraints: list[LinearConstraint],
    integrality: np.ndarray,
    bounds: Bounds,
    settings: SolverSettings,
) -> MilpSolution:
    """Solve `min c^T x` s.t. `constraints`, given per-variable integrality and bounds.

    Raises `SolverInvalidError` if the solve is not accepted.
    """
    result = milp(
        minimize_coefficients,
        constraints=constraints,
        integrality=integrality,
        bounds=bounds,
        options={"mip_rel_gap": settings.mip_rel_gap, "time_limit": settings.time_limit_seconds},
    )

    objective_value = result.fun
    mip_gap = result.mip_gap
    accepted = (
        bool(result.success)
        and result.status == _OPTIMAL_STATUS
        and objective_value is not None
        and mip_gap is not None
        and mip_gap <= settings.accept_mip_gap_leq
    )
    if not accepted or objective_value is None or mip_gap is None:
        raise SolverInvalidError(
            f"SOLVER_INVALID: success={result.success}, status={result.status}, "
            f"mip_gap={mip_gap}, message={result.message}"
        )

    return MilpSolution(
        x=np.asarray(result.x),
        objective_value=float(objective_value),
        mip_gap=float(mip_gap),
    )


def solve_binary_milp(
    minimize_coefficients: np.ndarray,
    constraints: list[LinearConstraint],
    settings: SolverSettings,
) -> MilpSolution:
    """Solve `min c^T x` s.t. `constraints`, `x in {0,1}^n`."""
    n_vars = minimize_coefficients.shape[0]
    solution = solve_milp(
        minimize_coefficients,
        constraints,
        integrality=np.ones(n_vars),
        bounds=Bounds(_BINARY_LOWER_BOUND, _BINARY_UPPER_BOUND),
        settings=settings,
    )
    return MilpSolution(
        x=np.round(solution.x).astype(np.int64),
        objective_value=solution.objective_value,
        mip_gap=solution.mip_gap,
    )
