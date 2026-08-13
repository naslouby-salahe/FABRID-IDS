from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from fabrid.allocation.contracts import Allocation
from fabrid.domain.identifiers import ClientId
from fabrid.domain.values import (
    AnalysisSeed,
    Probability,
    RowCount,
    TargetFalsePositiveRate,
)

_MAX_RANDOM_SEED = 2**31 - 1


@dataclass(frozen=True, slots=True)
class AllocationStabilitySummary:
    modal_target_rate: TargetFalsePositiveRate
    modal_frequency: Probability
    median_target_rate: TargetFalsePositiveRate
    percentile_5: TargetFalsePositiveRate
    percentile_95: TargetFalsePositiveRate
    instability: Probability


@dataclass(frozen=True, slots=True)
class ClientAllocationStability:
    client_id: ClientId
    summary: AllocationStabilitySummary


@dataclass(frozen=True, slots=True)
class AllocationStabilityAnalysis:
    clients: tuple[ClientAllocationStability, ...]

    def __post_init__(self) -> None:
        client_ids = tuple(client.client_id for client in self.clients)
        if len(set(client_ids)) != len(client_ids):
            raise ValueError("allocation stability contains duplicate clients")


def summarize_client_stability(
    selected_rates: tuple[TargetFalsePositiveRate, ...],
) -> AllocationStabilitySummary:
    if not selected_rates:
        raise ValueError("allocation stability requires at least one replicate")

    raw = np.asarray(tuple(rate.value for rate in selected_rates), dtype=np.float64)
    values, counts = np.unique(raw, return_counts=True)
    modal_index = int(np.argmax(counts))
    modal_frequency = Probability(float(counts[modal_index]) / len(selected_rates))
    return AllocationStabilitySummary(
        modal_target_rate=TargetFalsePositiveRate(float(values[modal_index])),
        modal_frequency=modal_frequency,
        median_target_rate=TargetFalsePositiveRate(float(np.median(raw))),
        percentile_5=TargetFalsePositiveRate(float(np.percentile(raw, 5))),
        percentile_95=TargetFalsePositiveRate(float(np.percentile(raw, 95))),
        instability=Probability(1.0 - modal_frequency.value),
    )


def run_allocation_sensitivity(
    resample_and_allocate: Callable[[AnalysisSeed], Allocation],
    replicates: RowCount,
    seed: AnalysisSeed,
) -> AllocationStabilityAnalysis:
    if replicates.value < 1:
        raise ValueError("allocation sensitivity requires at least one replicate")

    rng = np.random.default_rng(seed.value)
    replicate_seeds = tuple(
        AnalysisSeed(int(value))
        for value in rng.integers(0, _MAX_RANDOM_SEED, size=replicates.value)
    )
    allocations = tuple(resample_and_allocate(value) for value in replicate_seeds)
    first_client_ids = tuple(
        decision.client_id for decision in allocations[0].decisions
    )
    for allocation in allocations[1:]:
        if tuple(decision.client_id for decision in allocation.decisions) != first_client_ids:
            raise ValueError("all sensitivity replicates must allocate the same ordered clients")

    clients = tuple(
        ClientAllocationStability(
            client_id=client_id,
            summary=summarize_client_stability(
                tuple(
                    allocation.decision(client_id).target_rate
                    for allocation in allocations
                )
            ),
        )
        for client_id in first_client_ids
    )
    return AllocationStabilityAnalysis(clients)
