from __future__ import annotations

import math

import numpy as np
import pytest

from fabrid.calibration.final_calibration import calibrate_final_thresholds
from fabrid.evaluation.record_level import ClientId
from fabrid.schemas.allocation import Allocation, AllocationDecision, AllocationPolicy


def _allocation(alphas: dict[str, float]) -> Allocation:
    decisions = {
        ClientId(c): AllocationDecision(client_id=ClientId(c), alpha_selected=a)
        for c, a in alphas.items()
    }
    return Allocation(policy=AllocationPolicy.EQ_FPR, decisions=decisions)


def test_final_calibration_uses_only_provided_scores() -> None:
    allocation = _allocation({"1": 0.2})
    scores = {ClientId("1"): np.array([1.0, 2.0, 3.0, 4.0, 5.0])}
    results = calibrate_final_thresholds(allocation, scores)
    result = results[ClientId("1")]
    assert result.calibration_n == 5
    assert result.threshold.value == pytest.approx(5.0)
    assert result.alpha_selected == pytest.approx(0.2)


def test_hash_changes_when_scores_change() -> None:
    allocation = _allocation({"1": 0.2})
    base = calibrate_final_thresholds(allocation, {ClientId("1"): np.array([1.0, 2.0, 3.0])})
    changed = calibrate_final_thresholds(allocation, {ClientId("1"): np.array([1.0, 2.0, 4.0])})
    assert base[ClientId("1")].calibration_sha256 != changed[ClientId("1")].calibration_sha256


def test_hash_stable_for_identical_scores() -> None:
    allocation = _allocation({"1": 0.2})
    scores = {ClientId("1"): np.array([1.0, 2.0, 3.0])}
    first = calibrate_final_thresholds(allocation, scores)
    second = calibrate_final_thresholds(allocation, scores)
    assert first[ClientId("1")].calibration_sha256 == second[ClientId("1")].calibration_sha256


def test_alpha_zero_yields_infinite_threshold() -> None:
    allocation = _allocation({"1": 0.0})
    scores = {ClientId("1"): np.array([1.0, 2.0, 3.0])}
    results = calibrate_final_thresholds(allocation, scores)
    assert results[ClientId("1")].threshold.value == math.inf


def test_mismatched_client_sets_rejected() -> None:
    allocation = _allocation({"1": 0.2})
    with pytest.raises(ValueError):
        calibrate_final_thresholds(allocation, {ClientId("2"): np.array([1.0])})
