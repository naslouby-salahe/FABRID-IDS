from __future__ import annotations

import numpy as np
import pytest

from fabrid.allocation.frontier import (
    AttackSubtypeSelection,
    CandidateConfusions,
    ClientFrontierInputs,
    EligibleClientFrontier,
    FallbackClientFrontier,
    FederationFrontierInputs,
    SubtypeConfusion,
    SubtypeConfusionCounts,
    build_federation_frontier,
    restrict_to_subtypes,
)
from fabrid.domain.identifiers import AttackSubtypeId, ClientId
from fabrid.domain.scores import ScoreVector
from fabrid.domain.values import RowCount, TargetFalsePositiveRate
from fabrid.protocol.specification import PROTOCOL


def _inputs(client_id: ClientId, subtype_rows: tuple[RowCount, RowCount]) -> ClientFrontierInputs:
    scan = AttackSubtypeId("scan")
    udp = AttackSubtypeId("udp")
    candidates = tuple(
        CandidateConfusions(
            target_rate=rate,
            subtypes=(
                SubtypeConfusion(scan, SubtypeConfusionCounts(tp, RowCount(subtype_rows[0].value - tp.value))),
                SubtypeConfusion(udp, SubtypeConfusionCounts(tp, RowCount(subtype_rows[1].value - tp.value))),
            ),
        )
        for rate, tp in (
            (TargetFalsePositiveRate(0.0), RowCount(0)),
            (TargetFalsePositiveRate(0.01), RowCount(50)),
            (TargetFalsePositiveRate(0.02), RowCount(100)),
        )
    )
    return ClientFrontierInputs(
        client_id=client_id,
        benign_frontier_scores=ScoreVector(np.asarray((0.1, 0.2, 0.3, 0.4, 0.5))),
        candidates=candidates,
    )


def test_eligible_frontier_has_curve_and_provisional_thresholds() -> None:
    client = _inputs(ClientId("eligible"), (RowCount(100), RowCount(100)))

    federation = build_federation_frontier(
        FederationFrontierInputs((client,)),
        PROTOCOL.utility_eligibility,
    )

    frontier = federation.clients[0]
    assert isinstance(frontier, EligibleClientFrontier)
    assert tuple(point.utility.value for point in frontier.utility_curve.points) == pytest.approx((0.0, 0.5, 1.0))
    assert len(frontier.provisional_thresholds) == 3


def test_ineligible_frontier_is_explicit_fallback_without_curve() -> None:
    scan = AttackSubtypeId("scan")
    client = ClientFrontierInputs(
        client_id=ClientId("fallback"),
        benign_frontier_scores=ScoreVector(np.asarray((0.1, 0.2))),
        candidates=(
            CandidateConfusions(
                target_rate=TargetFalsePositiveRate(0.01),
                subtypes=(SubtypeConfusion(scan, SubtypeConfusionCounts(RowCount(5), RowCount(0))),),
            ),
        ),
    )

    federation = build_federation_frontier(
        FederationFrontierInputs((client,)),
        PROTOCOL.utility_eligibility,
    )

    assert isinstance(federation.clients[0], FallbackClientFrontier)


def test_restrict_to_subtypes_preserves_benign_frontier_scores() -> None:
    inputs = _inputs(ClientId("client"), (RowCount(100), RowCount(100)))
    selection = AttackSubtypeSelection((AttackSubtypeId("scan"),))

    restricted = restrict_to_subtypes(inputs, selection)

    assert restricted.benign_frontier_scores is inputs.benign_frontier_scores
    assert all(
        tuple(subtype.subtype for subtype in candidate.subtypes) == (AttackSubtypeId("scan"),)
        for candidate in restricted.candidates
    )


def test_empty_federation_frontier_is_rejected() -> None:
    with pytest.raises(ValueError):
        FederationFrontierInputs(())
