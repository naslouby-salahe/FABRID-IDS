from __future__ import annotations

import numpy as np
import pytest

from fabrid.allocation.problem import (
    ClientRowCount,
    dataset_count_weights,
    equal_client_weights,
    weight_gamma_transform,
)
from fabrid.datasets.registry import ClientPopulation


def test_equal_client_weights() -> None:
    population = ClientPopulation(("a", "b", "c"))
    weights = equal_client_weights(population)
    assert weights.for_client("a") == pytest.approx(1.0 / 3.0)
    assert weights.for_client("b") == pytest.approx(1.0 / 3.0)
    assert weights.for_client("c") == pytest.approx(1.0 / 3.0)
    assert sum(client.weight for client in weights.clients) == pytest.approx(1.0)


def test_dataset_count_weights() -> None:
    population = ClientPopulation(("a", "b"))
    weights = dataset_count_weights(population, (ClientRowCount("a", 3), ClientRowCount("b", 1)))
    assert weights.for_client("a") == pytest.approx(0.75)
    assert weights.for_client("b") == pytest.approx(0.25)


def test_dataset_count_weights_require_exact_population_coverage() -> None:
    population = ClientPopulation(("a", "b"))
    unknown_client = (ClientRowCount("a", 1), ClientRowCount("unknown", 100))
    with pytest.raises(ValueError):
        dataset_count_weights(population, unknown_client)
    zero_weight = (ClientRowCount("a", 1), ClientRowCount("b", 0))
    with pytest.raises(ValueError):
        dataset_count_weights(population, zero_weight)
    duplicate_client = (ClientRowCount("a", 1), ClientRowCount("a", 2))
    with pytest.raises(ValueError):
        dataset_count_weights(population, duplicate_client)


def test_weight_gamma_transform() -> None:
    population = ClientPopulation(("a", "b", "c"))
    reference = equal_client_weights(population)
    gamma_one = weight_gamma_transform(reference, 1.0)
    for client in population.clients:
        assert gamma_one.for_client(client) == pytest.approx(reference.for_client(client))
    gamma_zero = weight_gamma_transform(reference, 0.0)
    for client in population.clients:
        assert gamma_zero.for_client(client) == pytest.approx(1.0 / 3.0)
    unbalanced = dataset_count_weights(
        population, (ClientRowCount("a", 6), ClientRowCount("b", 3), ClientRowCount("c", 1))
    )
    for gamma in (0.0, 0.5, 1.0, 1.5):
        transformed = weight_gamma_transform(unbalanced, gamma)
        total = sum(client.weight for client in transformed.clients)
        assert total == pytest.approx(1.0)


def test_federation_weights_subset_preserves_relative_weights() -> None:
    population = ClientPopulation(("a", "b", "c"))
    weights = equal_client_weights(population)
    subset = weights.subset(ClientPopulation(("a", "c")))
    assert subset.for_client("a") == pytest.approx(1.0 / 3.0)
    assert subset.for_client("c") == pytest.approx(1.0 / 3.0)
    assert np.isclose(subset.for_client("a"), 1.0 / 3.0)
