from __future__ import annotations

import pytest

from fabrid.evaluation.record_level import ClientId
from fabrid.evaluation.weight_sensitivity import (
    WeightConcentration,
    gamma_reweight,
    preregistered_gamma_sweep,
)

_REFERENCE = {ClientId("big"): 0.7, ClientId("small"): 0.3}


def test_gamma_zero_yields_equal_weights() -> None:
    result = gamma_reweight(_REFERENCE, gamma=0.0)
    assert result[ClientId("big")] == pytest.approx(0.5)
    assert result[ClientId("small")] == pytest.approx(0.5)


def test_gamma_one_recovers_original_normalized_weights() -> None:
    result = gamma_reweight(_REFERENCE, gamma=1.0)
    assert result[ClientId("big")] == pytest.approx(0.7)
    assert result[ClientId("small")] == pytest.approx(0.3)


def test_gamma_greater_than_one_amplifies_concentration() -> None:
    result = gamma_reweight(_REFERENCE, gamma=1.5)
    assert result[ClientId("big")] > 0.7


def test_gamma_between_zero_and_one_reduces_concentration() -> None:
    result = gamma_reweight(_REFERENCE, gamma=0.5)
    assert 0.5 < result[ClientId("big")] < 0.7


def test_weights_always_sum_to_one() -> None:
    for gamma in (0.0, 0.5, 1.0, 1.5, 2.0):
        result = gamma_reweight(_REFERENCE, gamma)
        assert sum(result.values()) == pytest.approx(1.0)


def test_preregistered_sweep_covers_all_four_gammas() -> None:
    sweep = preregistered_gamma_sweep(_REFERENCE)
    assert set(sweep.keys()) == set(WeightConcentration)
    assert sweep[WeightConcentration.ORIGINAL][ClientId("big")] == pytest.approx(0.7)


def test_nonpositive_weight_rejected() -> None:
    with pytest.raises(ValueError):
        gamma_reweight({ClientId("a"): 0.0}, gamma=1.0)


def test_empty_weights_rejected() -> None:
    with pytest.raises(ValueError):
        gamma_reweight({}, gamma=1.0)
