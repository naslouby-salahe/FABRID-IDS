from __future__ import annotations

import math

import numpy as np

from fabrid.domain.scores import AlertVector, ScoreVector
from fabrid.domain.values import RowCount, TargetFalsePositiveRate, Threshold


def alerts_above_threshold(
    scores: ScoreVector,
    threshold: Threshold,
) -> AlertVector:
    return AlertVector(scores.values > threshold.value)


def calibrate_threshold(
    benign_scores: ScoreVector,
    target_rate: TargetFalsePositiveRate,
) -> Threshold:
    row_count = benign_scores.row_count.value
    if target_rate.value == 0.0 or row_count == 0:
        return Threshold(math.inf)

    rank = math.ceil((row_count + 1) * (1.0 - target_rate.value))
    if rank > row_count:
        return Threshold(math.inf)
    if rank < 1:
        return Threshold(-math.inf)

    sorted_scores = np.sort(benign_scores.values)
    return Threshold(float(sorted_scores[rank - 1]))


def minimum_resolvable_rate(
    row_count: RowCount,
) -> TargetFalsePositiveRate | None:
    if row_count.value == 0:
        return None
    return TargetFalsePositiveRate(1.0 / (row_count.value + 1))
