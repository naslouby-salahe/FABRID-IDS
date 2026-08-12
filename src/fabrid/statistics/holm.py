"""Holm-Bonferroni step-down correction for a family of hypothesis tests."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HolmResult:
    original_index: int
    p_value: float
    adjusted_p_value: float
    reject_at_alpha: bool


def holm_correction(p_values: tuple[float, ...], alpha: float) -> tuple[HolmResult, ...]:
    if not p_values:
        raise ValueError("holm_correction requires at least one p-value")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if any(not (0.0 <= p <= 1.0) for p in p_values):
        raise ValueError("all p-values must be in [0, 1]")

    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])

    adjusted: dict[int, float] = {}
    running_max = 0.0
    for rank, original_index in enumerate(order):
        multiplier = m - rank
        candidate = min(1.0, p_values[original_index] * multiplier)
        running_max = max(running_max, candidate)
        adjusted[original_index] = running_max

    still_rejecting = True
    results_by_index: dict[int, HolmResult] = {}
    for rank, original_index in enumerate(order):
        multiplier = m - rank
        threshold = alpha / multiplier
        still_rejecting = still_rejecting and p_values[original_index] <= threshold
        results_by_index[original_index] = HolmResult(
            original_index=original_index,
            p_value=p_values[original_index],
            adjusted_p_value=adjusted[original_index],
            reject_at_alpha=still_rejecting,
        )
    return tuple(results_by_index[i] for i in range(m))
