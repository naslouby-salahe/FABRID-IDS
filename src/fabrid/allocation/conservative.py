from __future__ import annotations

from scipy.stats import beta

from fabrid.allocation.contracts import (
    ClientUtilityCurve,
    ClientUtilityPoint,
)
from fabrid.allocation.frontier import (
    ClientFrontierInputs,
    SubtypeConfusionCounts,
    client_eligibility,
    eligible_subtypes,
)
from fabrid.domain.enums import EligibilityStatus
from fabrid.domain.values import DetectionUtility, Probability, TruePositiveRate
from fabrid.protocol.models import UtilityEligibility


def one_sided_lower_confidence_bound(
    counts: SubtypeConfusionCounts,
    confidence: Probability,
) -> TruePositiveRate:
    if confidence.value <= 0.0 or confidence.value >= 1.0:
        raise ValueError("confidence must be strictly between zero and one")

    successes = counts.true_positive.value
    trials = counts.attack_rows.value
    if successes == 0:
        return TruePositiveRate(0.0)
    if successes == trials:
        return TruePositiveRate(
            float(beta.ppf(1.0 - confidence.value, trials, 1))
        )
    return TruePositiveRate(
        float(
            beta.ppf(
                1.0 - confidence.value,
                successes,
                trials - successes + 1,
            )
        )
    )


def build_conservative_utility_curve(
    inputs: ClientFrontierInputs,
    guardrails: UtilityEligibility,
    confidence: Probability,
) -> ClientUtilityCurve:
    if client_eligibility(inputs, guardrails) is not EligibilityStatus.ELIGIBLE:
        raise ValueError("conservative utility requires an eligible client")

    selection = eligible_subtypes(inputs, guardrails)
    points: list[ClientUtilityPoint] = []
    for candidate in inputs.candidates:
        lower_bounds = tuple(
            one_sided_lower_confidence_bound(subtype.counts, confidence).value
            for subtype in candidate.subtypes
            if selection.contains(subtype.subtype)
        )
        if not lower_bounds:
            raise ValueError("conservative utility requires eligible attack subtypes")
        points.append(
            ClientUtilityPoint(
                target_rate=candidate.target_rate,
                utility=DetectionUtility(sum(lower_bounds) / len(lower_bounds)),
            )
        )

    return ClientUtilityCurve(client_id=inputs.client_id, points=tuple(points))
