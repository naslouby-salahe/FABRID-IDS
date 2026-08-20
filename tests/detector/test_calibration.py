from __future__ import annotations

import math

import numpy as np
import pytest

from fabrid.artifacts.paths import ScoreCoordinate
from fabrid.config import BenignSplit, DatasetId, Label
from fabrid.detector.calibration import (
    FinalCalibrationDecision,
    FinalCalibrationInputs,
    FinalCalibrationResult,
    alerts_above_threshold,
    calibrate_final_thresholds,
    calibrate_threshold,
    minimum_resolvable_rate,
    sorted_order_statistic_threshold,
)
from fabrid.detector.scoring import ScorePartitionArtifact, ScoreRecord


def test_relabeled_preserves_fields_and_changes_only_sample_id() -> None:
    original = ScoreRecord(
        sample_id="cam|attack|7",
        source_file="f.csv",
        source_row=7,
        score=0.731,
        label=Label.ATTACK,
        attack_subtype="mirai",
        timestamp=None,
    )
    relabeled = original.relabeled("oracle|cam|attack|7")
    assert relabeled is not original
    assert relabeled.sample_id == "oracle|cam|attack|7"
    assert relabeled.source_file == original.source_file
    assert relabeled.source_row == original.source_row
    assert relabeled.score == original.score
    assert relabeled.label is Label.ATTACK
    assert relabeled.attack_subtype == original.attack_subtype
    assert relabeled.timestamp == original.timestamp
    assert original.sample_id == "cam|attack|7"


def test_relabeled_equivalent_to_dataclasses_replace() -> None:
    from dataclasses import replace

    benign = ScoreRecord(
        sample_id="cam|benign|3",
        source_file="f.csv",
        source_row=3,
        score=0.12,
        label=Label.BENIGN,
        attack_subtype=None,
        timestamp=None,
    )
    assert benign.relabeled("oracle|3") == replace(benign, sample_id="oracle|3")


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


def test_t18_duplicate_score_ties_are_non_alerts() -> None:
    scores = np.array([0.5, 0.5, 0.5, 0.9])
    alerts = alerts_above_threshold(scores, 0.5)
    assert alerts.tolist() == [False, False, False, True]
    assert float(alerts.mean()) == pytest.approx(0.25)


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


def test_calibrate_final_thresholds_pools_frontier_and_final_cal() -> None:
    inputs = FinalCalibrationInputs(
        clients=(
            _artifact(
                "a",
                np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
                BenignSplit.FINAL_CAL,
            ),
            _artifact(
                "a",
                np.array([0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]),
                BenignSplit.FRONTIER,
            ),
            _artifact(
                "b",
                np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]),
                BenignSplit.FINAL_CAL,
            ),
            _artifact(
                "b",
                np.array([0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11]),
                BenignSplit.FRONTIER,
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
    assert result_a.calibration_count == 20
    assert result_a.threshold == pytest.approx(1.4)
    assert result_b.threshold == pytest.approx(0.09)
    assert result_a.calibration_digest != result_b.calibration_digest


def test_calibrate_final_thresholds_rejects_mismatched_clients() -> None:
    inputs = FinalCalibrationInputs(
        clients=(
            _artifact("a", np.array([0.1, 0.2, 0.3, 0.4]), BenignSplit.FINAL_CAL),
            _artifact("a", np.array([0.5, 0.6, 0.7, 0.8]), BenignSplit.FRONTIER),
        )
    )
    decision = FinalCalibrationDecision(client_id="b", target_rate=0.1)
    with pytest.raises(ValueError):
        calibrate_final_thresholds(decisions=(decision,), inputs=inputs)


def test_final_calibration_inputs_rejects_non_calibration_windows() -> None:
    artifact = _artifact("a", np.array([0.1, 0.2]), BenignSplit.TEST)
    with pytest.raises(ValueError):
        FinalCalibrationInputs(clients=(artifact,))


def test_final_calibration_inputs_requires_both_windows_per_client() -> None:
    clients = (
        _artifact("a", np.array([0.1, 0.2]), BenignSplit.FINAL_CAL),
        _artifact("a", np.array([0.3, 0.4]), BenignSplit.FRONTIER),
        _artifact("b", np.array([0.5, 0.6]), BenignSplit.FINAL_CAL),
    )
    with pytest.raises(ValueError):
        FinalCalibrationInputs(clients=clients)
