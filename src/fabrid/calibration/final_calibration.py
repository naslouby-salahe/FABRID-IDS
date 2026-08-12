"""Independent final threshold calibration, performed only after alpha* is frozen.

Discards the provisional `BENIGN_FRONTIER` threshold and recomputes tau_k
from `BENIGN_FINAL_CAL` alone. No attack-validation data enters this stage.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from fabrid.calibration.order_statistic import Threshold, calibrate_threshold
from fabrid.evaluation.record_level import ClientId
from fabrid.schemas.allocation import Allocation


@dataclass(frozen=True, slots=True)
class FinalCalibrationResult:
    client_id: ClientId
    alpha_selected: float
    threshold: Threshold
    calibration_n: int
    calibration_sha256: str


def _hash_calibration_scores(scores: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(scores, dtype=np.float64).tobytes()).hexdigest()


def calibrate_final_thresholds(
    allocation: Allocation, benign_final_cal_scores: Mapping[ClientId, np.ndarray]
) -> dict[ClientId, FinalCalibrationResult]:
    if allocation.decisions.keys() != benign_final_cal_scores.keys():
        raise ValueError("allocation and benign_final_cal_scores must cover the same client set")

    results: dict[ClientId, FinalCalibrationResult] = {}
    for client_id, decision in allocation.decisions.items():
        scores = benign_final_cal_scores[client_id]
        threshold = calibrate_threshold(scores, decision.alpha_selected)
        results[client_id] = FinalCalibrationResult(
            client_id=client_id,
            alpha_selected=decision.alpha_selected,
            threshold=threshold,
            calibration_n=scores.shape[0],
            calibration_sha256=_hash_calibration_scores(scores),
        )
    return results
