from __future__ import annotations

import numpy as np
from scipy.optimize import LinearConstraint

from fabrid.domain.values import FalsePositiveBudget, RowCount


def one_hot_constraint(
    client_count: RowCount,
    candidate_count: RowCount,
) -> LinearConstraint:
    matrix = np.zeros(
        (client_count.value, client_count.value * candidate_count.value)
    )
    for client_index in range(client_count.value):
        start = client_index * candidate_count.value
        matrix[
            client_index,
            start : start + candidate_count.value,
        ] = 1.0
    return LinearConstraint(matrix, lb=1.0, ub=1.0)


def budget_constraint(
    cost: np.ndarray,
    budget: FalsePositiveBudget,
) -> LinearConstraint:
    return LinearConstraint(cost.reshape(1, -1), lb=-np.inf, ub=budget.value)


def pad_constraint_columns(
    constraint: LinearConstraint,
    matrix: np.ndarray,
    extra_columns: RowCount,
) -> LinearConstraint:
    padded = np.hstack(
        [matrix, np.zeros((matrix.shape[0], extra_columns.value))]
    )
    return LinearConstraint(
        padded,
        lb=constraint.lb,
        ub=constraint.ub,
    )  # type: ignore[arg-type]


def lexicographic_weights(
    client_count: RowCount,
    candidate_count: RowCount,
) -> np.ndarray:
    per_client_scale = np.power(
        float(candidate_count.value + 1),
        np.arange(client_count.value, dtype=np.float64),
    )
    return np.repeat(per_client_scale[::-1], candidate_count.value)
