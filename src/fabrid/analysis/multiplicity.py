from __future__ import annotations

from dataclasses import dataclass

from fabrid.analysis.types import HypothesisIndex
from fabrid.domain.enums import HypothesisDecision
from fabrid.domain.values import Probability


@dataclass(frozen=True, slots=True)
class HolmResult:
    index: HypothesisIndex
    p_value: Probability
    adjusted_p_value: Probability
    decision: HypothesisDecision


def holm_correction(
    p_values: tuple[Probability, ...],
    alpha: Probability,
) -> tuple[HolmResult, ...]:
    if not p_values:
        raise ValueError("Holm correction requires at least one p-value")
    if alpha.value <= 0.0 or alpha.value >= 1.0:
        raise ValueError("Holm alpha must be strictly between zero and one")

    family_size = len(p_values)
    ordered_indices = tuple(
        sorted(range(family_size), key=lambda index: p_values[index].value)
    )
    adjusted_values = [0.0] * family_size
    running_maximum = 0.0
    for rank, index in enumerate(ordered_indices):
        multiplier = family_size - rank
        candidate = min(1.0, p_values[index].value * multiplier)
        running_maximum = max(running_maximum, candidate)
        adjusted_values[index] = running_maximum

    decisions = [HypothesisDecision.RETAIN] * family_size
    still_rejecting = True
    for rank, index in enumerate(ordered_indices):
        threshold = alpha.value / (family_size - rank)
        still_rejecting = still_rejecting and p_values[index].value <= threshold
        decisions[index] = (
            HypothesisDecision.REJECT
            if still_rejecting
            else HypothesisDecision.RETAIN
        )

    return tuple(
        HolmResult(
            index=HypothesisIndex(index),
            p_value=p_values[index],
            adjusted_p_value=Probability(adjusted_values[index]),
            decision=decisions[index],
        )
        for index in range(family_size)
    )
