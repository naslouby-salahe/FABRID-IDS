"""Shared MILP formulation building blocks for FABRID_MACRO and FABRID_MINIMAX.

Both policies select exactly one candidate target rate per eligible client
under one shared weighted budget; this module builds the constraint matrices
and lexicographic tie-break objective common to both.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import LinearConstraint


def one_hot_constraint(n_clients: int, n_candidates: int) -> LinearConstraint:
    """sum_j x_{k,j} = 1 for every client k, over the flattened (k, j) variable order."""
    matrix = np.zeros((n_clients, n_clients * n_candidates))
    for client_index in range(n_clients):
        start = client_index * n_candidates
        matrix[client_index, start : start + n_candidates] = 1.0
    return LinearConstraint(matrix, lb=1.0, ub=1.0)


def budget_constraint(cost: np.ndarray, budget: float) -> LinearConstraint:
    return LinearConstraint(cost.reshape(1, -1), lb=-np.inf, ub=budget)


def pad_constraint_columns(
    constraint: LinearConstraint, matrix: np.ndarray, extra_columns: int
) -> LinearConstraint:
    """Extend a constraint's dense variable block with `extra_columns` zero-coefficient columns.

    `matrix` is the same dense array the constraint was originally built from
    (constraint.A may be sparse and expose no typed `.shape`, so the caller
    passes the known dense source instead of round-tripping through it).
    """
    padded = np.hstack([matrix, np.zeros((matrix.shape[0], extra_columns))])
    # LinearConstraint.lb/ub are typed as `float` in scipy's stubs but are
    # actually stored as per-row arrays once the constraint has >1 row.
    return LinearConstraint(padded, lb=constraint.lb, ub=constraint.ub)  # type: ignore[arg-type]


def lexicographic_weights(n_clients: int, n_candidates: int) -> np.ndarray:
    """Client 0 dominates client 1 dominates ...; magnitude spaced to preserve ordering."""
    per_client_scale = np.power(float(n_candidates + 1), np.arange(n_clients, dtype=np.float64))
    return np.repeat(per_client_scale[::-1], n_candidates)
