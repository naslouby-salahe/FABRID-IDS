from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from fabrid.allocation.contracts import (
    AllocationWeights,
    ClientBudgetWeight,
    FederationWeights,
)
from fabrid.domain.enums import ExperimentVariantId
from fabrid.domain.identifiers import ClientId
from fabrid.domain.values import ClientWeight, RowCount


class WeightConcentration(float, Enum):
    EQUAL_CLIENT = 0.0
    REDUCED = 0.5
    REFERENCE = 1.0
    AMPLIFIED = 1.5

    @property
    def variant_id(self) -> ExperimentVariantId:
        if self is WeightConcentration.EQUAL_CLIENT:
            return ExperimentVariantId.WEIGHT_GAMMA_0
        if self is WeightConcentration.REDUCED:
            return ExperimentVariantId.WEIGHT_GAMMA_0P5
        if self is WeightConcentration.REFERENCE:
            return ExperimentVariantId.WEIGHT_GAMMA_1
        return ExperimentVariantId.WEIGHT_GAMMA_1P5


@dataclass(frozen=True, slots=True)
class ClientDatasetCount:
    client_id: ClientId
    rows: RowCount

    def __post_init__(self) -> None:
        if self.rows.value == 0:
            raise ValueError("dataset-count proxy requires strictly positive client counts")


@dataclass(frozen=True, slots=True)
class DatasetCounts:
    clients: tuple[ClientDatasetCount, ...]

    def __post_init__(self) -> None:
        if not self.clients:
            raise ValueError("dataset-count proxy requires at least one client")
        client_ids = tuple(client.client_id for client in self.clients)
        if len(set(client_ids)) != len(client_ids):
            raise ValueError("dataset-count proxy contains duplicate clients")


@dataclass(frozen=True, slots=True)
class WeightSensitivityScenario:
    concentration: WeightConcentration
    weights: FederationWeights

    @property
    def variant_id(self) -> ExperimentVariantId:
        return self.concentration.variant_id


def dataset_count_reference_weights(counts: DatasetCounts) -> FederationWeights:
    total_rows = sum(client.rows.value for client in counts.clients)
    return FederationWeights(
        AllocationWeights(
            tuple(
                ClientBudgetWeight(
                    client_id=client.client_id,
                    weight=ClientWeight(client.rows.value / total_rows),
                )
                for client in counts.clients
            )
        )
    )


def gamma_reweight(
    reference: FederationWeights,
    concentration: WeightConcentration,
) -> FederationWeights:
    if any(client.weight.value <= 0.0 for client in reference.clients):
        raise ValueError("weight sensitivity requires strictly positive reference weights")
    powered = tuple(
        (client.client_id, client.weight.value ** concentration.value)
        for client in reference.clients
    )
    total = sum(value for _, value in powered)
    return FederationWeights(
        AllocationWeights(
            tuple(
                ClientBudgetWeight(client_id, ClientWeight(value / total))
                for client_id, value in powered
            )
        )
    )


def preregistered_weight_scenarios(
    reference: FederationWeights,
) -> tuple[WeightSensitivityScenario, ...]:
    return tuple(
        WeightSensitivityScenario(
            concentration=concentration,
            weights=gamma_reweight(reference, concentration),
        )
        for concentration in WeightConcentration
    )
