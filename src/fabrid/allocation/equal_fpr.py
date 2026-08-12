"""EQ_FPR baseline: every client receives the same target rate, alpha_k = B_FP."""

from __future__ import annotations

from collections.abc import Sequence

from fabrid.evaluation.record_level import ClientId
from fabrid.schemas.allocation import Allocation, AllocationDecision, AllocationPolicy


def allocate_equal_fpr(
    client_ids: Sequence[ClientId], budget: float, alpha_max: float
) -> Allocation:
    if not client_ids:
        raise ValueError("allocate_equal_fpr requires at least one client")
    if budget < 0:
        raise ValueError(f"budget must be non-negative, got {budget}")
    if budget > alpha_max:
        raise ValueError(f"budget {budget} exceeds the local target-rate cap {alpha_max}")

    decisions = {
        client_id: AllocationDecision(client_id=client_id, alpha_selected=budget)
        for client_id in client_ids
    }
    return Allocation(policy=AllocationPolicy.EQ_FPR, decisions=decisions)
