"""Client detection-utility curves: u_{k,j} = mean over eligible subtypes of TPR_{k,a,j}.

Attacks are averaged within client by subtype first, so a large subtype
cannot dominate the client's utility value merely by contributing more rows.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from fabrid.evaluation.record_level import AttackSubtype, ClientId
from fabrid.schemas.allocation import ClientUtilityCurve


class SubtypeRecallSource(Protocol):
    """Anything that can report a subtype-level true positive rate.

    Lets `client_utility` be reused for both raw and lower-confidence-bound
    (`fabrid.frontier.conservative`) recall sources.
    """

    def true_positive_rate(self) -> float: ...


@dataclass(frozen=True, slots=True)
class SubtypeConfusionCounts:
    true_positive: int
    false_negative: int

    def __post_init__(self) -> None:
        if self.true_positive < 0 or self.false_negative < 0:
            raise ValueError(
                f"confusion counts must be non-negative, got "
                f"true_positive={self.true_positive}, false_negative={self.false_negative}"
            )
        if self.true_positive + self.false_negative == 0:
            raise ValueError("a subtype confusion count must cover at least one attack row")

    def true_positive_rate(self) -> float:
        return self.true_positive / (self.true_positive + self.false_negative)


def client_utility(subtype_counts: Mapping[AttackSubtype, SubtypeRecallSource]) -> float:
    """u_{k,j}: mean subtype TPR for one client at one candidate rate."""
    if not subtype_counts:
        raise ValueError("client_utility requires at least one eligible attack subtype")
    return sum(counts.true_positive_rate() for counts in subtype_counts.values()) / len(
        subtype_counts
    )


def build_utility_curve(
    client_id: ClientId,
    alpha_grid: tuple[float, ...],
    subtype_counts_by_candidate: Sequence[Mapping[AttackSubtype, SubtypeConfusionCounts]],
) -> ClientUtilityCurve:
    """One `client_utility` evaluation per candidate in `alpha_grid`, in order."""
    if len(subtype_counts_by_candidate) != len(alpha_grid):
        raise ValueError(
            f"subtype_counts_by_candidate ({len(subtype_counts_by_candidate)}) must have one "
            f"entry per alpha_grid candidate ({len(alpha_grid)})"
        )
    utility = tuple(client_utility(counts) for counts in subtype_counts_by_candidate)
    return ClientUtilityCurve(client_id=client_id, alpha_grid=alpha_grid, utility=utility)
