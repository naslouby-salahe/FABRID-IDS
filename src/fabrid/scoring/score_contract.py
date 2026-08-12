"""The score contract: strict decision rule and cross-policy AUROC invariance.

Every threshold policy in the system must alert under exactly `s > tau`, and
because policies only choose thresholds/target-rates over one frozen score
set, their AUROC (which depends only on score ranking, not threshold) must be
numerically identical.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy.stats import rankdata

AUROC_IDENTITY_TOLERANCE = 1e-12


def decide(scores: np.ndarray, threshold: float) -> np.ndarray:
    """alert iff s > tau; ties at the threshold are non-alerts."""
    return scores > threshold


def compute_auroc(scores: np.ndarray, is_attack: np.ndarray) -> float:
    """Rank-based AUROC (Mann-Whitney U statistic), tie-averaged."""
    n_positive = int(np.sum(is_attack))
    n_negative = int(is_attack.size - n_positive)
    if n_positive == 0 or n_negative == 0:
        raise ValueError("compute_auroc requires at least one attack and one benign sample")

    ranks = rankdata(scores)
    sum_ranks_positive = float(np.sum(ranks[is_attack]))
    return (sum_ranks_positive - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative)


def compute_auprc(scores: np.ndarray, is_attack: np.ndarray) -> float:
    """Precision-recall AUC via the trapezoidal rule over the score-sorted precision/recall
    curve (attack = positive class), matching the score contract's strict `>` decision rule at
    every candidate threshold. Depends only on score ranking, like AUROC.
    """
    n_positive = int(np.sum(is_attack))
    if n_positive == 0:
        raise ValueError("compute_auprc requires at least one attack sample")
    if is_attack.size - n_positive == 0:
        raise ValueError("compute_auprc requires at least one benign sample")

    order = np.argsort(-scores, kind="stable")
    sorted_is_attack = is_attack[order]
    cumulative_tp = np.cumsum(sorted_is_attack)
    cumulative_fp = np.cumsum(~sorted_is_attack)

    recall = cumulative_tp / n_positive
    precision = cumulative_tp / (cumulative_tp + cumulative_fp)

    recall_with_origin = np.concatenate(([0.0], recall))
    precision_with_origin = np.concatenate(([1.0], precision))
    return float(np.trapezoid(precision_with_origin, recall_with_origin))


def assert_auroc_invariant(auroc_by_policy: Mapping[str, float]) -> None:
    if not auroc_by_policy:
        raise ValueError("assert_auroc_invariant requires at least one policy")
    values = list(auroc_by_policy.values())
    spread = max(values) - min(values)
    if spread >= AUROC_IDENTITY_TOLERANCE:
        raise ValueError(
            f"AUROC invariant violated: spread {spread} >= {AUROC_IDENTITY_TOLERANCE} "
            f"across policies {dict(auroc_by_policy)}"
        )
