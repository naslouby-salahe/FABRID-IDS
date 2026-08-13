from __future__ import annotations

import pytest

from fabrid.allocation.weight_sensitivity import (
    ClientDatasetCount,
    DatasetCounts,
    WeightConcentration,
    dataset_count_reference_weights,
    gamma_reweight,
    preregistered_weight_scenarios,
)
from fabrid.domain.enums import ExperimentVariantId
from fabrid.domain.identifiers import ClientId
from fabrid.domain.values import RowCount


def _reference():
    return dataset_count_reference_weights(
        DatasetCounts(
            (
                ClientDatasetCount(ClientId("big"), RowCount(70)),
                ClientDatasetCount(ClientId("small"), RowCount(30)),
            )
        )
    )


def test_dataset_count_proxy_normalizes_positive_counts() -> None:
    reference = _reference()

    assert reference.for_client(ClientId("big")).value == pytest.approx(0.7)
    assert reference.for_client(ClientId("small")).value == pytest.approx(0.3)


def test_gamma_zero_yields_equal_client_weights() -> None:
    result = gamma_reweight(_reference(), WeightConcentration.EQUAL_CLIENT)

    assert result.for_client(ClientId("big")).value == pytest.approx(0.5)
    assert result.for_client(ClientId("small")).value == pytest.approx(0.5)


def test_gamma_reference_recovers_original_weights() -> None:
    result = gamma_reweight(_reference(), WeightConcentration.REFERENCE)

    assert result.for_client(ClientId("big")).value == pytest.approx(0.7)
    assert result.for_client(ClientId("small")).value == pytest.approx(0.3)


def test_gamma_changes_concentration_in_preregistered_directions() -> None:
    reduced = gamma_reweight(_reference(), WeightConcentration.REDUCED)
    amplified = gamma_reweight(_reference(), WeightConcentration.AMPLIFIED)

    assert 0.5 < reduced.for_client(ClientId("big")).value < 0.7
    assert amplified.for_client(ClientId("big")).value > 0.7


def test_preregistered_scenarios_cover_exact_four_variants() -> None:
    scenarios = preregistered_weight_scenarios(_reference())

    assert tuple(scenario.concentration for scenario in scenarios) == tuple(WeightConcentration)
    assert tuple(scenario.variant_id for scenario in scenarios) == (
        ExperimentVariantId.WEIGHT_GAMMA_0,
        ExperimentVariantId.WEIGHT_GAMMA_0P5,
        ExperimentVariantId.WEIGHT_GAMMA_1,
        ExperimentVariantId.WEIGHT_GAMMA_1P5,
    )
    assert all(
        sum(client.weight.value for client in scenario.weights.clients) == pytest.approx(1.0)
        for scenario in scenarios
    )


def test_dataset_count_proxy_rejects_zero_rows() -> None:
    with pytest.raises(ValueError):
        ClientDatasetCount(ClientId("empty"), RowCount(0))
