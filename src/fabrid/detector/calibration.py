from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from fabrid.artifacts.json import digest_text
from fabrid.config import (
    TIGHT_TOLERANCE,
    ArtifactDigest,
    BenignSplit,
    ClientId,
    Probability,
    RowCount,
    TargetFalsePositiveRate,
    Threshold,
)
from fabrid.detector.scoring import ScorePartitionArtifact


def alerts_above_threshold(scores: np.ndarray, threshold: Threshold) -> np.ndarray:
    return scores > threshold


def sorted_order_statistic_threshold(
    sorted_scores: np.ndarray, target_rate: TargetFalsePositiveRate
) -> Threshold:
    row_count = sorted_scores.size
    if target_rate <= TIGHT_TOLERANCE or row_count == 0:
        return math.inf
    rank = math.ceil((row_count + 1) * (1.0 - target_rate))
    if rank > row_count:
        return math.inf
    if rank < 1:
        return -math.inf
    return float(sorted_scores[rank - 1])


def calibrate_threshold(
    benign_scores: np.ndarray, target_rate: TargetFalsePositiveRate
) -> Threshold:
    return sorted_order_statistic_threshold(np.sort(benign_scores), target_rate)


def minimum_resolvable_rate(row_count: RowCount) -> Probability | None:
    if row_count == 0:
        return None
    return 1.0 / (row_count + 1)


@dataclass(frozen=True, slots=True)
class FinalCalibrationDecision:
    client_id: ClientId
    target_rate: TargetFalsePositiveRate


@dataclass(frozen=True, slots=True)
class FinalCalibrationInputs:
    _WINDOWS_PER_CLIENT = 2
    clients: tuple[ScorePartitionArtifact, ...]

    def __post_init__(self) -> None:
        if not self.clients:
            raise ValueError("final calibration requires at least one client")
        if any(
            artifact.split not in (BenignSplit.FRONTIER, BenignSplit.FINAL_CAL)
            for artifact in self.clients
        ):
            raise ValueError("final calibration accepts FRONTIER and FINAL_CAL artifacts only")
        client_ids = tuple(artifact.coordinate.client_id for artifact in self.clients)
        if len(set(client_ids)) * self._WINDOWS_PER_CLIENT != len(client_ids):
            raise ValueError(
                "final calibration requires exactly one frontier and one "
                "final-calibration artifact per client"
            )
        for client_id in set(client_ids):
            splits = {
                artifact.split
                for artifact in self.clients
                if artifact.coordinate.client_id == client_id
            }
            if splits != {BenignSplit.FRONTIER, BenignSplit.FINAL_CAL}:
                raise ValueError(
                    f"final calibration requires both FRONTIER and FINAL_CAL for {client_id}"
                )

    def for_client(
        self, client_id: ClientId
    ) -> tuple[ScorePartitionArtifact, ScorePartitionArtifact]:
        windows = tuple(
            artifact for artifact in self.clients if artifact.coordinate.client_id == client_id
        )
        if len(windows) != 2:
            raise KeyError(client_id)
        return windows


@dataclass(frozen=True, slots=True)
class FinalCalibrationResult:
    client_id: ClientId
    target_rate: TargetFalsePositiveRate
    threshold: Threshold
    calibration_count: RowCount
    calibration_digest: ArtifactDigest


@dataclass(frozen=True, slots=True)
class FinalCalibrationResults:
    clients: tuple[FinalCalibrationResult, ...]

    def for_client(self, client_id: ClientId) -> FinalCalibrationResult:
        for result in self.clients:
            if result.client_id == client_id:
                return result
        raise KeyError(client_id)


def calibrate_final_thresholds(
    decisions: tuple[FinalCalibrationDecision, ...],
    inputs: FinalCalibrationInputs,
) -> FinalCalibrationResults:
    decision_clients = {decision.client_id for decision in decisions}
    calibration_clients = {artifact.coordinate.client_id for artifact in inputs.clients}
    if decision_clients != calibration_clients:
        raise ValueError("allocation decisions and final calibration must cover the same clients")
    results: list[FinalCalibrationResult] = []
    for decision in decisions:
        frontier, final_cal = inputs.for_client(decision.client_id)
        scores = np.concatenate((frontier.score_values(), final_cal.score_values()))
        results.append(
            FinalCalibrationResult(
                client_id=decision.client_id,
                target_rate=decision.target_rate,
                threshold=calibrate_threshold(scores, decision.target_rate),
                calibration_count=scores.size,
                calibration_digest=digest_text((frontier.digest(), final_cal.digest())),
            )
        )
    return FinalCalibrationResults(tuple(results))
