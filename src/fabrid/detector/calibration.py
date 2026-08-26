from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import binom

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


def nominal_order_statistic_rank(
    row_count: RowCount, target_rate: TargetFalsePositiveRate
) -> RowCount | None:
    if row_count == 0 or target_rate <= TIGHT_TOLERANCE:
        return None
    rank = math.ceil((row_count + 1) * (1.0 - target_rate))
    return rank if rank <= row_count else None


def finite_sample_safe_order_statistic_rank(
    row_count: RowCount,
    target_rate: TargetFalsePositiveRate,
    confidence: Probability,
) -> RowCount | None:
    if row_count == 0 or target_rate <= TIGHT_TOLERANCE:
        return None
    if not 0.0 < confidence < 1.0:
        raise ValueError("finite-sample confidence must be strictly between zero and one")
    benign_cdf_at_target = 1.0 - target_rate
    low = 1
    high = row_count
    best: RowCount | None = None
    while low <= high:
        rank = (low + high) // 2
        coverage = float(binom.cdf(rank - 1, row_count, benign_cdf_at_target))
        if coverage >= confidence:
            best = rank
            high = rank - 1
        else:
            low = rank + 1
    return best


def finite_sample_safe_threshold(
    benign_scores: np.ndarray,
    target_rate: TargetFalsePositiveRate,
    confidence: Probability,
) -> Threshold:
    rank = finite_sample_safe_order_statistic_rank(benign_scores.size, target_rate, confidence)
    if rank is None:
        return math.inf
    return float(np.sort(benign_scores)[rank - 1])


def minimum_resolvable_rate(row_count: RowCount) -> Probability | None:
    if row_count == 0:
        return None
    return 1.0 / (row_count + 1)


@dataclass(frozen=True, slots=True)
class CalibrationReachability:
    sample_count: RowCount
    distinct_score_count: RowCount
    tied_score_count: RowCount
    strict_exceedance_counts: tuple[RowCount, ...]
    attainable_rates: tuple[Probability, ...]
    smallest_attainable_nonzero_rate: Probability | None
    requested_rate: TargetFalsePositiveRate
    attained_rate: Probability
    requested_attainment_error: Probability
    order_statistic_rank: RowCount | None = None


def calibration_reachability(
    benign_scores: np.ndarray,
    target_rate: TargetFalsePositiveRate,
    threshold: Threshold,
    order_statistic_rank: RowCount | None = None,
) -> CalibrationReachability:
    if benign_scores.size == 0:
        return CalibrationReachability(
            sample_count=0,
            distinct_score_count=0,
            tied_score_count=0,
            strict_exceedance_counts=(0,),
            attainable_rates=(0.0,),
            smallest_attainable_nonzero_rate=None,
            requested_rate=target_rate,
            attained_rate=0.0,
            requested_attainment_error=target_rate,
            order_statistic_rank=order_statistic_rank,
        )
    _, counts = np.unique(benign_scores, return_counts=True)
    alert_counts = np.cumsum(counts[::-1])[:-1]
    attainable = alert_counts / benign_scores.size
    smallest = None if attainable.size == 0 else float(np.min(attainable))
    attained = (
        0.0
        if math.isinf(threshold) and threshold > 0.0
        else float(np.count_nonzero(benign_scores > threshold) / benign_scores.size)
    )
    return CalibrationReachability(
        sample_count=benign_scores.size,
        distinct_score_count=counts.size,
        tied_score_count=int(np.count_nonzero(counts > 1)),
        strict_exceedance_counts=(0, *(int(count) for count in alert_counts)),
        attainable_rates=(0.0, *(float(rate) for rate in attainable), 1.0),
        smallest_attainable_nonzero_rate=smallest,
        requested_rate=target_rate,
        attained_rate=attained,
        requested_attainment_error=abs(target_rate - attained),
        order_statistic_rank=order_statistic_rank,
    )


@dataclass(frozen=True, slots=True)
class FinalCalibrationDecision:
    client_id: ClientId
    target_rate: TargetFalsePositiveRate


@dataclass(frozen=True, slots=True)
class FinalCalibrationInputs:
    clients: tuple[ScorePartitionArtifact, ...]

    def __post_init__(self) -> None:
        if not self.clients:
            raise ValueError("final calibration requires at least one client")
        if any(artifact.split is not BenignSplit.FINAL_CAL for artifact in self.clients):
            raise ValueError("final calibration accepts FINAL_CAL artifacts only")
        client_ids = tuple(artifact.coordinate.client_id for artifact in self.clients)
        if len(set(client_ids)) != len(client_ids):
            raise ValueError("final calibration requires exactly one artifact per client")

    def for_client(self, client_id: ClientId) -> ScorePartitionArtifact:
        for artifact in self.clients:
            if artifact.coordinate.client_id == client_id:
                return artifact
        raise KeyError(client_id)


@dataclass(frozen=True, slots=True)
class FinalCalibrationResult:
    client_id: ClientId
    target_rate: TargetFalsePositiveRate
    threshold: Threshold
    calibration_count: RowCount
    order_statistic_rank: RowCount | None
    calibration_digest: ArtifactDigest
    reachability: CalibrationReachability


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
    finite_sample_confidence: Probability | None = None,
) -> FinalCalibrationResults:
    decision_clients = {decision.client_id for decision in decisions}
    calibration_clients = {artifact.coordinate.client_id for artifact in inputs.clients}
    if decision_clients != calibration_clients:
        raise ValueError("allocation decisions and final calibration must cover the same clients")
    results: list[FinalCalibrationResult] = []
    for decision in decisions:
        final_cal = inputs.for_client(decision.client_id)
        scores = final_cal.score_values()
        if finite_sample_confidence is None:
            threshold = calibrate_threshold(scores, decision.target_rate)
            rank = nominal_order_statistic_rank(scores.size, decision.target_rate)
        else:
            threshold = finite_sample_safe_threshold(
                scores, decision.target_rate, finite_sample_confidence
            )
            rank = finite_sample_safe_order_statistic_rank(
                scores.size, decision.target_rate, finite_sample_confidence
            )
        results.append(
            FinalCalibrationResult(
                client_id=decision.client_id,
                target_rate=decision.target_rate,
                threshold=threshold,
                calibration_count=scores.size,
                order_statistic_rank=rank,
                calibration_digest=final_cal.digest(),
                reachability=calibration_reachability(
                    scores, decision.target_rate, threshold, rank
                ),
            )
        )
    return FinalCalibrationResults(tuple(results))
