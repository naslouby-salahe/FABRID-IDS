from __future__ import annotations

import numpy as np
import pytest

from fabrid.config import ScorerDefinition
from fabrid.execution.prerequisites import score_with_scorer


def test_combined_scorer_alerts_on_either_signal() -> None:
    rng = np.random.default_rng(7)
    calibration_scores = rng.normal(0.0, 1.0, 10_000)
    calibration_feature = rng.normal(0.0, 1.0, 10_000)
    attack_scores = np.array([0.0, 0.01, -0.01])
    attack_feature = np.array([4.0, 3.8, 3.9])
    ensemble = score_with_scorer(
        ScorerDefinition.AE_PLUS_AUXILIARY_F75,
        attack_scores,
        attack_feature,
        calibration_scores,
        calibration_feature,
    )
    assert np.all(ensemble > 0.99)


def test_combined_scorer_order_statistic_flags_exact_alpha() -> None:
    rng = np.random.default_rng(11)
    calibration_scores = rng.normal(0.0, 1.0, 20_000)
    calibration_feature = rng.exponential(1.0, 20_000)
    ensemble = score_with_scorer(
        ScorerDefinition.AE_PLUS_AUXILIARY_F75,
        calibration_scores,
        calibration_feature,
        calibration_scores,
        calibration_feature,
    )
    alpha = 0.005
    k = int(round((1.0 - alpha) * ensemble.size))
    threshold = np.partition(ensemble, k - 1)[k - 1]
    flagged = np.count_nonzero(ensemble > threshold) / ensemble.size
    assert abs(flagged - alpha) < 0.01


def test_combined_scorer_preserves_reconstruction_separation() -> None:
    rng = np.random.default_rng(13)
    calibration_scores = rng.normal(0.0, 1.0, 10_000)
    calibration_feature = rng.normal(0.0, 1.0, 10_000)
    attack_scores = np.array([4.0, 5.0])
    attack_feature = np.array([0.0, 0.0])
    ensemble = score_with_scorer(
        ScorerDefinition.AE_PLUS_AUXILIARY_F75,
        attack_scores,
        attack_feature,
        calibration_scores,
        calibration_feature,
    )
    assert np.all(ensemble > 0.99)


def test_ae_scorer_preserves_raw_reconstruction_scores() -> None:
    reconstruction = np.array([0.1, 0.3])
    assert np.array_equal(
        score_with_scorer(
            ScorerDefinition.AE_RECONSTRUCTION,
            reconstruction,
            None,
            np.array([0.0, 1.0]),
            None,
        ),
        reconstruction,
    )


def test_auxiliary_scorer_requires_development_reference() -> None:
    with pytest.raises(ValueError, match="requires auxiliary feature"):
        score_with_scorer(
            ScorerDefinition.AUXILIARY_F75,
            np.array([0.1]),
            None,
            np.array([0.0, 1.0]),
            None,
        )
