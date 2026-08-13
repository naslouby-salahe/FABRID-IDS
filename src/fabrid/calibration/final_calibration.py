from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fabrid.allocation.contracts import Allocation
from fabrid.artifacts.score import ScorePartitionArtifact
from fabrid.calibration.order_statistic import calibrate_threshold
from fabrid.domain.enums import BenignSplit
from fabrid.domain.identifiers import ArtifactDigest, ClientId
from fabrid.domain.scores import ScoreVector
from fabrid.domain.values import RowCount, TargetFalsePositiveRate, Threshold


@dataclass(frozen=True, slots=True)
class FinalCalibrationInputs:
    clients: tuple[ScorePartitionArtifact, ...]

    def __post_init__(self) -> None:
        if not self.clients:
            raise ValueError("final calibration requires at least one client")
        if any(artifact.split is not BenignSplit.FINAL_CAL for artifact in self.clients):
            raise ValueError("final calibration accepts BENIGN_FINAL_CAL artifacts only")
        client_ids = tuple(artifact.coordinate.client_id for artifact in self.clients)
        if len(set(client_ids)) != len(client_ids):
            raise ValueError("final calibration contains duplicate clients")

    def for_client(self, client_id: ClientId) -> ScorePartitionArtifact:
        for artifact in self.clients:
            if artifact.coordinate.client_id == client_id:
                return artifact
        raise KeyError(client_id.value)


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
        raise KeyError(client_id.value)


def _scores(artifact: ScorePartitionArtifact) -> ScoreVector:
    return ScoreVector(
        np.fromiter(
            (record.score.value for record in artifact.records),
            dtype=np.float64,
            count=len(artifact.records),
        )
    )


def calibrate_final_thresholds(
    allocation: Allocation,
    inputs: FinalCalibrationInputs,
) -> FinalCalibrationResults:
    allocation_clients = {decision.client_id for decision in allocation.decisions}
    calibration_clients = {
        artifact.coordinate.client_id for artifact in inputs.clients
    }
    if allocation_clients != calibration_clients:
        raise ValueError("allocation and final calibration must cover the same clients")

    results: list[FinalCalibrationResult] = []
    for decision in allocation.decisions:
        artifact = inputs.for_client(decision.client_id)
        scores = _scores(artifact)
        results.append(
            FinalCalibrationResult(
                client_id=decision.client_id,
                target_rate=decision.target_rate,
                threshold=calibrate_threshold(scores, decision.target_rate),
                calibration_count=scores.row_count,
                calibration_digest=artifact.digest(),
            )
        )
    return FinalCalibrationResults(tuple(results))
