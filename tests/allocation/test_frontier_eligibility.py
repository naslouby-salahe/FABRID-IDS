from __future__ import annotations

import numpy as np

from fabrid.allocation.frontier import (
    CandidateConfusions,
    ClientFrontierInputs,
    FederationFrontierInputs,
    SubtypeConfusion,
    SubtypeConfusionCounts,
    build_federation_frontier,
    client_eligibility,
    eligible_subtypes,
)
from fabrid.domain.enums import EligibilityStatus
from fabrid.domain.identifiers import AttackSubtypeId, ClientId
from fabrid.domain.scores import ScoreVector
from fabrid.domain.values import Probability, RowCount, TargetFalsePositiveRate
from fabrid.protocol.specification import PROTOCOL


def _client(
    client_id: ClientId,
    rows: tuple[tuple[AttackSubtypeId, RowCount], ...],
) -> ClientFrontierInputs:
    return ClientFrontierInputs(
        client_id=client_id,
        benign_frontier_scores=ScoreVector(
            np.asarray((0.1, 0.2, 0.3), dtype=np.float64)
        ),
        candidates=(
            CandidateConfusions(
                target_rate=TargetFalsePositiveRate(0.01),
                subtypes=tuple(
                    SubtypeConfusion(
                        subtype=subtype,
                        counts=SubtypeConfusionCounts(
                            true_positive=RowCount(row_count.value),
                            false_negative=RowCount(0),
                        ),
                    )
                    for subtype, row_count in rows
                ),
            ),
        ),
    )


def test_client_requires_total_rows_and_two_eligible_subtypes() -> None:
    scan = AttackSubtypeId("scan")
    udp = AttackSubtypeId("udp")
    eligible = _client(
        ClientId("eligible"),
        ((scan, RowCount(100)), (udp, RowCount(100))),
    )
    too_few_rows = _client(
        ClientId("too-few-rows"),
        ((scan, RowCount(60)), (udp, RowCount(60))),
    )
    one_subtype = _client(ClientId("one-subtype"), ((scan, RowCount(300)),))

    guardrails = PROTOCOL.utility_eligibility
    assert client_eligibility(eligible, guardrails) is EligibilityStatus.ELIGIBLE
    assert client_eligibility(too_few_rows, guardrails) is EligibilityStatus.FALLBACK
    assert client_eligibility(one_subtype, guardrails) is EligibilityStatus.FALLBACK


def test_subtype_below_row_floor_is_excluded_without_invalidating_others() -> None:
    scan = AttackSubtypeId("scan")
    udp = AttackSubtypeId("udp")
    tcp = AttackSubtypeId("tcp")
    client = _client(
        ClientId("mixed-subtypes"),
        ((scan, RowCount(100)), (udp, RowCount(49)), (tcp, RowCount(100))),
    )

    selection = eligible_subtypes(client, PROTOCOL.utility_eligibility)

    assert selection.contains(scan)
    assert not selection.contains(udp)
    assert selection.contains(tcp)
    assert (
        client_eligibility(client, PROTOCOL.utility_eligibility)
        is EligibilityStatus.ELIGIBLE
    )


def test_federation_frontier_reports_typed_fallback_rate() -> None:
    scan = AttackSubtypeId("scan")
    udp = AttackSubtypeId("udp")
    eligible = _client(
        ClientId("eligible"),
        ((scan, RowCount(100)), (udp, RowCount(100))),
    )
    fallback = _client(ClientId("fallback"), ((scan, RowCount(300)),))

    frontier = build_federation_frontier(
        FederationFrontierInputs((eligible, fallback)),
        PROTOCOL.utility_eligibility,
    )

    assert frontier.fallback_rate == Probability(0.5)
