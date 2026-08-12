"""EQ_ALERT: conditional baseline, valid only under defensibly unequal client weights.

Under equal-client weighting EQ_ALERT is mathematically identical to EQ_FPR
and must not be used as a separately reported baseline; this module raises
rather than silently degenerating into EQ_FPR.
"""

from __future__ import annotations

from collections.abc import Mapping

from fabrid.evaluation.record_level import ClientId
from fabrid.schemas.allocation import Allocation, AllocationDecision, AllocationPolicy

_WEIGHT_EQUALITY_TOLERANCE = 1e-12
_BISECTION_ITERATIONS = 100


def _max_constant_budget_share(
    weight: Mapping[ClientId, float], budget: float, alpha_max: float
) -> float:
    """Largest c such that sum_k min(c, alpha_max * w_k) <= budget."""

    def total_cost(c: float) -> float:
        return sum(min(c, alpha_max * w) for w in weight.values())

    upper = alpha_max * max(weight.values())
    if total_cost(upper) <= budget:
        return upper
    low, high = 0.0, upper
    for _ in range(_BISECTION_ITERATIONS):
        mid = (low + high) / 2
        if total_cost(mid) <= budget:
            low = mid
        else:
            high = mid
    return low


def allocate_equal_alert(
    weight: Mapping[ClientId, float], budget: float, alpha_max: float
) -> Allocation:
    if not weight:
        raise ValueError("allocate_equal_alert requires at least one client")
    if budget < 0:
        raise ValueError(f"budget must be non-negative, got {budget}")

    values = list(weight.values())
    if max(values) - min(values) < _WEIGHT_EQUALITY_TOLERANCE:
        raise ValueError(
            "EQ_ALERT is identical to EQ_FPR under equal-client weighting and must not be used "
            "as a separate baseline; this call received (near-)equal weights"
        )

    constant_share = _max_constant_budget_share(weight, budget, alpha_max)
    decisions = {
        client_id: AllocationDecision(
            client_id=client_id, alpha_selected=min(constant_share / w, alpha_max)
        )
        for client_id, w in weight.items()
    }
    return Allocation(policy=AllocationPolicy.EQ_ALERT, decisions=decisions)
