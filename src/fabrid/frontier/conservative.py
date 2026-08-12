"""Conservative (lower-confidence-bound) utility curves.

For every subtype recall estimate, computes a one-sided 95% Clopper-Pearson
binomial lower confidence bound, then resolves FABRID using that pessimistic
curve instead of the raw TPR. Tests whether FABRID exploits uncertain
optimistic validation estimates.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from scipy.stats import beta

from fabrid.evaluation.record_level import AttackSubtype, ClientId
from fabrid.frontier.utility import SubtypeConfusionCounts, client_utility
from fabrid.schemas.allocation import ClientUtilityCurve

_DEFAULT_CONFIDENCE = 0.95


@dataclass(frozen=True, slots=True)
class SubtypeConfusionCountsLcb:
    """A `SubtypeConfusionCounts`-shaped view whose rate is the LCB, not the raw TPR."""

    lower_confidence_bound: float

    def true_positive_rate(self) -> float:
        return self.lower_confidence_bound


def one_sided_lower_confidence_bound(
    counts: SubtypeConfusionCounts, confidence: float = _DEFAULT_CONFIDENCE
) -> float:
    """One-sided Clopper-Pearson lower bound on the true positive rate."""
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    successes = counts.true_positive
    trials = counts.true_positive + counts.false_negative
    if successes == 0:
        return 0.0
    if successes == trials:
        return float(beta.ppf(1.0 - confidence, trials, 1))
    return float(beta.ppf(1.0 - confidence, successes, trials - successes + 1))


def conservative_client_utility(
    subtype_counts: Mapping[AttackSubtype, SubtypeConfusionCounts],
    confidence: float = _DEFAULT_CONFIDENCE,
) -> float:
    lcb_counts = {
        subtype: SubtypeConfusionCountsLcb(one_sided_lower_confidence_bound(counts, confidence))
        for subtype, counts in subtype_counts.items()
    }
    if not lcb_counts:
        raise ValueError(
            "conservative_client_utility requires at least one eligible attack subtype"
        )
    return client_utility(lcb_counts)


def build_conservative_utility_curve(
    client_id: ClientId,
    alpha_grid: tuple[float, ...],
    subtype_counts_by_candidate: Sequence[Mapping[AttackSubtype, SubtypeConfusionCounts]],
    confidence: float = _DEFAULT_CONFIDENCE,
) -> ClientUtilityCurve:
    if len(subtype_counts_by_candidate) != len(alpha_grid):
        raise ValueError(
            f"subtype_counts_by_candidate ({len(subtype_counts_by_candidate)}) must have one "
            f"entry per alpha_grid candidate ({len(alpha_grid)})"
        )
    utility = tuple(
        conservative_client_utility(counts, confidence) for counts in subtype_counts_by_candidate
    )
    return ClientUtilityCurve(client_id=client_id, alpha_grid=alpha_grid, utility=utility)
