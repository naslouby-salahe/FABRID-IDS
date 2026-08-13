from __future__ import annotations

import numpy as np

from fabrid.allocation.conservative import (
    build_conservative_utility_curve,
    one_sided_lower_confidence_bound,
)
from fabrid.allocation.frontier import (
    CandidateConfusions,
    ClientFrontierInputs,
    SubtypeConfusion,
    SubtypeConfusionCounts,
)
from fabrid.domain.enums import FallbackPolicy
from fabrid.domain.identifiers import AttackSubtypeId, ClientId
from fabrid.domain.scores import ScoreVector
from fabrid.domain.values import Probability, RowCount, TargetFalsePositiveRate
from fabrid.protocol.models import UtilityEligibility


def test_one_sided_lower_bound_is_conservative() -> None:
    counts = SubtypeConfusionCounts(
        true_positive=RowCount(80),
        false_negative=RowCount(20),
    )
    bound = one_sided_lower_confidence_bound(counts, Probability(0.95))
    assert 0.0 <= bound.value < counts.true_positive_rate().value


def test_conservative_curve_preserves_target_grid_and_lowers_utility() -> None:
    subtype_a = AttackSubtypeId("bashlite_scan")
    subtype_b = AttackSubtypeId("mirai_ack")
    inputs = ClientFrontierInputs(
        client_id=ClientId("client-a"),
        benign_frontier_scores=ScoreVector(np.asarray((0.1, 0.2, 0.3))),
        candidates=(
            CandidateConfusions(
                target_rate=TargetFalsePositiveRate(0.0),
                subtypes=(
                    SubtypeConfusion(
                        subtype=subtype_a,
                        counts=SubtypeConfusionCounts(RowCount(0), RowCount(50)),
                    ),
                    SubtypeConfusion(
                        subtype=subtype_b,
                        counts=SubtypeConfusionCounts(RowCount(0), RowCount(50)),
                    ),
                ),
            ),
            CandidateConfusions(
                target_rate=TargetFalsePositiveRate(0.01),
                subtypes=(
                    SubtypeConfusion(
                        subtype=subtype_a,
                        counts=SubtypeConfusionCounts(RowCount(40), RowCount(10)),
                    ),
                    SubtypeConfusion(
                        subtype=subtype_b,
                        counts=SubtypeConfusionCounts(RowCount(45), RowCount(5)),
                    ),
                ),
            ),
        ),
    )
    guardrails = UtilityEligibility(
        minimum_attack_validation_rows=RowCount(100),
        minimum_eligible_subtypes=RowCount(2),
        minimum_rows_per_subtype=RowCount(50),
        fallback_policy=FallbackPolicy.EQUAL_FPR_AT_BUDGET,
    )

    curve = build_conservative_utility_curve(
        inputs=inputs,
        guardrails=guardrails,
        confidence=Probability(0.95),
    )

    assert tuple(point.target_rate for point in curve.points) == (
        TargetFalsePositiveRate(0.0),
        TargetFalsePositiveRate(0.01),
    )
    observed_mean = (40 / 50 + 45 / 50) / 2
    assert curve.points[1].utility.value < observed_mean
