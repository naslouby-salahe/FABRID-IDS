from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import binom

from fabrid.artifacts.paths import ScoreCoordinate
from fabrid.config import BenignSplit, DatasetId, Label
from fabrid.detector.calibration import (
    FinalCalibrationDecision,
    FinalCalibrationInputs,
    FinalCalibrationResult,
    alerts_above_threshold,
    calibrate_final_thresholds,
    calibrate_threshold,
    calibration_reachability,
    finite_sample_safe_order_statistic_rank,
    finite_sample_safe_threshold,
    minimum_resolvable_rate,
    sorted_order_statistic_threshold,
)
from fabrid.detector.scoring import ScorePartitionArtifact, ScoreRecord


def test_order_statistic_threshold_guarantees_target_rate() -> None:
    rng = np.random.default_rng(7)
    scores = np.sort(rng.normal(size=10_000))
    target = 0.01
    threshold = sorted_order_statistic_threshold(scores, target)
    alert_fraction = float((scores > threshold).mean())
    assert alert_fraction <= target
    assert alert_fraction > 0.0


def test_calibrate_threshold_matches_sorted_equivalent() -> None:
    rng = np.random.default_rng(11)
    scores = rng.normal(size=5_000)
    assert calibrate_threshold(scores, 0.005) == sorted_order_statistic_threshold(
        np.sort(scores), 0.005
    )


def test_t17_final_calibration_resolution_yields_infinite_threshold() -> None:
    scores = np.array([0.1, 0.2, 0.3, 0.4])
    assert sorted_order_statistic_threshold(scores, 0.0) == math.inf
    assert sorted_order_statistic_threshold(np.array([]), 0.01) == math.inf
    assert sorted_order_statistic_threshold(scores, 0.1) == math.inf
    assert sorted_order_statistic_threshold(scores, 1.0) == -math.inf


def test_minimum_resolvable_rate() -> None:
    assert minimum_resolvable_rate(0) is None
    assert minimum_resolvable_rate(9) == pytest.approx(0.1)


def test_finite_sample_safe_rank_meets_and_minimizes_one_sided_coverage() -> None:
    row_count = 100
    target_rate = 0.1
    confidence = 0.95
    rank = finite_sample_safe_order_statistic_rank(row_count, target_rate, confidence)
    assert rank is not None
    assert binom.cdf(rank - 1, row_count, 1.0 - target_rate) >= confidence
    if rank > 1:
        assert binom.cdf(rank - 2, row_count, 1.0 - target_rate) < confidence


def test_finite_sample_safe_threshold_is_never_lower_than_nominal() -> None:
    scores = np.linspace(0.0, 1.0, 1_000)
    safe = finite_sample_safe_threshold(scores, 0.05, 0.95)
    nominal = calibrate_threshold(scores, 0.05)
    assert safe >= nominal


def test_t18_duplicate_score_ties_are_non_alerts() -> None:
    scores = np.array([0.5, 0.5, 0.5, 0.9])
    alerts = alerts_above_threshold(scores, 0.5)
    assert alerts.tolist() == [False, False, False, True]
    assert float(alerts.mean()) == pytest.approx(0.25)


def test_calibration_reachability_reports_ties_and_attained_rate() -> None:
    scores = np.array([0.1, 0.5, 0.5, 0.9])
    reachability = calibration_reachability(scores, 0.3, threshold=0.5)
    assert reachability.sample_count == 4
    assert reachability.distinct_score_count == 3
    assert reachability.tied_score_count == 1
    assert reachability.strict_exceedance_counts == (0, 1, 3)
    assert reachability.attainable_rates == pytest.approx((0.0, 0.25, 0.75, 1.0))
    assert reachability.smallest_attainable_nonzero_rate == pytest.approx(0.25)
    assert reachability.attained_rate == pytest.approx(0.25)
    assert reachability.requested_attainment_error == pytest.approx(0.05)


def _artifact(client_id: str, scores: np.ndarray, split: BenignSplit) -> ScorePartitionArtifact:
    return ScorePartitionArtifact(
        coordinate=ScoreCoordinate(
            dataset_id=DatasetId.NBAIOT, detector_seed=0, client_id=client_id
        ),
        split=split,
        records=tuple(
            ScoreRecord(
                sample_id=f"{client_id}|{split.value}|{index}",
                source_file="f.csv",
                source_row=index,
                score=float(score),
                label=Label.BENIGN,
                attack_subtype=None,
                timestamp=None,
            )
            for index, score in enumerate(scores)
        ),
    )


def test_calibrate_final_thresholds_uses_only_sealed_final_calibration() -> None:
    inputs = FinalCalibrationInputs(
        clients=(
            _artifact(
                "a",
                np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
                BenignSplit.FINAL_CAL,
            ),
            _artifact(
                "b",
                np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]),
                BenignSplit.FINAL_CAL,
            ),
        )
    )
    results = calibrate_final_thresholds(
        decisions=(
            FinalCalibrationDecision(client_id="a", target_rate=0.1),
            FinalCalibrationDecision(client_id="b", target_rate=0.2),
        ),
        inputs=inputs,
    )
    result_a: FinalCalibrationResult = results.for_client("a")
    result_b: FinalCalibrationResult = results.for_client("b")
    assert result_a.calibration_count == 10
    assert result_a.threshold == pytest.approx(1.0)
    assert result_b.threshold == pytest.approx(0.09)
    assert result_a.calibration_digest != result_b.calibration_digest
    assert result_a.reachability.sample_count == result_a.calibration_count


def test_final_calibration_can_use_predeclared_finite_sample_safety() -> None:
    inputs = FinalCalibrationInputs(
        clients=(_artifact("a", np.linspace(0.0, 1.0, 100), BenignSplit.FINAL_CAL),)
    )
    decisions = (FinalCalibrationDecision(client_id="a", target_rate=0.1),)
    nominal = calibrate_final_thresholds(decisions, inputs).for_client("a")
    safe = calibrate_final_thresholds(decisions, inputs, finite_sample_confidence=0.95).for_client(
        "a"
    )
    assert safe.threshold >= nominal.threshold
    assert safe.order_statistic_rank is not None
    assert nominal.order_statistic_rank is not None
    assert safe.order_statistic_rank > nominal.order_statistic_rank
    assert safe.reachability.order_statistic_rank == safe.order_statistic_rank


def test_calibrate_final_thresholds_rejects_mismatched_clients() -> None:
    inputs = FinalCalibrationInputs(
        clients=(_artifact("a", np.array([0.1, 0.2, 0.3, 0.4]), BenignSplit.FINAL_CAL),)
    )
    decision = FinalCalibrationDecision(client_id="b", target_rate=0.1)
    with pytest.raises(ValueError):
        calibrate_final_thresholds(decisions=(decision,), inputs=inputs)


def test_final_calibration_inputs_rejects_non_calibration_windows() -> None:
    artifact = _artifact("a", np.array([0.1, 0.2]), BenignSplit.TEST)
    with pytest.raises(ValueError):
        FinalCalibrationInputs(clients=(artifact,))


def test_final_calibration_inputs_rejects_frontier_leakage() -> None:
    artifact = _artifact("a", np.array([0.1, 0.2]), BenignSplit.FRONTIER)
    with pytest.raises(ValueError, match="FINAL_CAL"):
        FinalCalibrationInputs(clients=(artifact,))


def test_final_calibration_inputs_rejects_duplicate_client() -> None:
    clients = (
        _artifact("a", np.array([0.1, 0.2]), BenignSplit.FINAL_CAL),
        _artifact("a", np.array([0.3, 0.4]), BenignSplit.FINAL_CAL),
    )
    with pytest.raises(ValueError):
        FinalCalibrationInputs(clients=clients)
