"""Allocation-sensitivity analysis: resample BENIGN_FRONTIER and ATTACK_VALIDATION independently.

Because the data are temporally dependent (N-BaIoT is a sequential
dataset), this is described as an allocation *sensitivity* analysis rather
than a formal IID bootstrap confidence interval.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

from fabrid.evaluation.record_level import ClientId

_DEFAULT_REPLICATES = 500


@dataclass(frozen=True, slots=True)
class AllocationStabilitySummary:
    modal_alpha: float
    modal_frequency: float
    median_alpha: float
    percentile_5: float
    percentile_95: float
    instability: float


def summarize_client_stability(selected_alphas: tuple[float, ...]) -> AllocationStabilitySummary:
    """Instability_k = 1 - max_j P(alpha_k = alpha_j)."""
    if not selected_alphas:
        raise ValueError("summarize_client_stability requires at least one replicate")

    values, counts = np.unique(np.asarray(selected_alphas), return_counts=True)
    modal_index = int(np.argmax(counts))
    modal_frequency = float(counts[modal_index]) / len(selected_alphas)

    return AllocationStabilitySummary(
        modal_alpha=float(values[modal_index]),
        modal_frequency=modal_frequency,
        median_alpha=float(np.median(selected_alphas)),
        percentile_5=float(np.percentile(selected_alphas, 5)),
        percentile_95=float(np.percentile(selected_alphas, 95)),
        instability=1.0 - modal_frequency,
    )


def run_allocation_sensitivity(
    resample_and_allocate: Callable[[int], Mapping[ClientId, float]],
    replicates: int = _DEFAULT_REPLICATES,
    seed: int = 0,
) -> dict[ClientId, AllocationStabilitySummary]:
    """`resample_and_allocate(replicate_seed)` must resample BENIGN_FRONTIER and
    ATTACK_VALIDATION independently (within attack subtype), rebuild the frontier,
    and return one selected alpha per client for that replicate.
    """
    if replicates < 1:
        raise ValueError(f"replicates must be at least 1, got {replicates}")

    rng = np.random.default_rng(seed)
    replicate_seeds = rng.integers(0, 2**31 - 1, size=replicates)

    alphas_by_client: dict[ClientId, list[float]] = {}
    for replicate_seed in replicate_seeds:
        allocation = resample_and_allocate(int(replicate_seed))
        for client_id, alpha in allocation.items():
            alphas_by_client.setdefault(client_id, []).append(alpha)

    return {
        client_id: summarize_client_stability(tuple(alphas))
        for client_id, alphas in alphas_by_client.items()
    }
