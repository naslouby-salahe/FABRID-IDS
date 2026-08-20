from __future__ import annotations

import numpy as np

from fabrid.execution.prerequisites import ensemble_scores


def test_ensemble_scores_alert_on_either_signal() -> None:
    rng = np.random.default_rng(7)
    calibration_scores = rng.normal(0.0, 1.0, 10_000)
    calibration_feature = rng.normal(0.0, 1.0, 10_000)
    attack_scores = np.array([0.0, 0.01, -0.01])
    attack_feature = np.array([4.0, 3.8, 3.9])
    ensemble = ensemble_scores(
        attack_scores, attack_feature, calibration_scores, calibration_feature
    )
    assert np.all(ensemble > 0.99)


def test_ensemble_scores_order_statistic_flags_exact_alpha() -> None:
    rng = np.random.default_rng(11)
    calibration_scores = rng.normal(0.0, 1.0, 20_000)
    calibration_feature = rng.exponential(1.0, 20_000)
    ensemble = ensemble_scores(
        calibration_scores, calibration_feature, calibration_scores, calibration_feature
    )
    alpha = 0.005
    k = int(round((1.0 - alpha) * ensemble.size))
    threshold = np.partition(ensemble, k - 1)[k - 1]
    flagged = np.count_nonzero(ensemble > threshold) / ensemble.size
    assert abs(flagged - alpha) < 0.01


def test_ensemble_scores_preserves_reconstruction_separation() -> None:
    rng = np.random.default_rng(13)
    calibration_scores = rng.normal(0.0, 1.0, 10_000)
    calibration_feature = rng.normal(0.0, 1.0, 10_000)
    attack_scores = np.array([4.0, 5.0])
    attack_feature = np.array([0.0, 0.0])
    ensemble = ensemble_scores(
        attack_scores, attack_feature, calibration_scores, calibration_feature
    )
    assert np.all(ensemble > 0.99)
