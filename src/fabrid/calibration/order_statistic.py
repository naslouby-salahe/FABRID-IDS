"""Finite-sample order-statistic threshold calibration.

Given a benign calibration score set and a target rate ``alpha``, computes the
threshold ``tau`` such that the decision rule ``alert iff s > tau`` is expected
to admit approximately an ``alpha`` fraction of the calibration scores. Uses
strict ``>`` everywhere; ties at the threshold are non-alerts by construction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

INFINITE_THRESHOLD = math.inf


@dataclass(frozen=True, slots=True)
class Threshold:
    """A calibrated decision threshold. ``value`` may be ``+inf`` (zero alerts)."""

    value: float

    def alerts(self, scores: np.ndarray) -> np.ndarray:
        """Strict `>` decision rule; ties at threshold are non-alerts."""
        return scores > self.value


def calibrate_threshold(benign_scores: np.ndarray, alpha: float) -> Threshold:
    """r = ceil((n+1)(1-alpha)); tau = s_(r) if r<=n else +inf.

    alpha == 0 always yields +inf, regardless of sample size.
    """
    if alpha < 0:
        raise ValueError(f"alpha must be non-negative, got {alpha}")
    n = benign_scores.shape[0]
    if alpha == 0:
        return Threshold(INFINITE_THRESHOLD)
    if n == 0:
        return Threshold(INFINITE_THRESHOLD)

    rank = math.ceil((n + 1) * (1.0 - alpha))
    if rank > n:
        return Threshold(INFINITE_THRESHOLD)
    if rank < 1:
        # r <= 0 corresponds to a target rate so large that even the smallest
        # calibration score must alert; s_(0) is conventionally -inf.
        return Threshold(-math.inf)

    sorted_scores = np.sort(benign_scores)
    return Threshold(float(sorted_scores[rank - 1]))


def minimum_resolvable_rate(n: int) -> float:
    """Smallest nonzero alpha that can yield a finite tau: approximately 1/(n+1)."""
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if n == 0:
        return math.inf
    return 1.0 / (n + 1)
