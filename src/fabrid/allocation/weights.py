from __future__ import annotations

from fabrid.allocation.contracts import (
    AllocationWeights,
    ClientBudgetWeight,
    FederationWeights,
)
from fabrid.domain.population import ClientPopulation
from fabrid.domain.values import ClientWeight


def equal_client_weights(population: ClientPopulation) -> FederationWeights:
    weight = ClientWeight(1.0 / population.size.value)
    return FederationWeights(
        AllocationWeights(
            tuple(
                ClientBudgetWeight(client_id=client_id, weight=weight)
                for client_id in population.clients
            )
        )
    )
