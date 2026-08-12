from __future__ import annotations

import numpy as np
import pytest

from fabrid.config.protocol import UtilityEligibilityGuardrails
from fabrid.evaluation.record_level import AttackSubtype, ClientId
from fabrid.frontier.builder import ClientFrontierInputs, build_federation_frontier
from fabrid.frontier.utility import SubtypeConfusionCounts

_GRID = (0.0, 0.01, 0.02)
_GUARDRAILS = UtilityEligibilityGuardrails(
    min_attack_validation_rows=100, min_eligible_subtypes=2, min_rows_per_eligible_subtype=40
)


def _eligible_client_inputs() -> ClientFrontierInputs:
    return ClientFrontierInputs(
        benign_frontier_scores=np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
        subtype_validation_row_counts={AttackSubtype("scan"): 60, AttackSubtype("udp"): 60},
        subtype_confusion_by_candidate=[
            {
                AttackSubtype("scan"): SubtypeConfusionCounts(0, 10),
                AttackSubtype("udp"): SubtypeConfusionCounts(0, 10),
            },
            {
                AttackSubtype("scan"): SubtypeConfusionCounts(5, 5),
                AttackSubtype("udp"): SubtypeConfusionCounts(5, 5),
            },
            {
                AttackSubtype("scan"): SubtypeConfusionCounts(10, 0),
                AttackSubtype("udp"): SubtypeConfusionCounts(10, 0),
            },
        ],
    )


def _ineligible_client_inputs() -> ClientFrontierInputs:
    return ClientFrontierInputs(
        benign_frontier_scores=np.array([0.1, 0.2]),
        subtype_validation_row_counts={AttackSubtype("scan"): 5},
        subtype_confusion_by_candidate=[
            {AttackSubtype("scan"): SubtypeConfusionCounts(0, 5)},
            {AttackSubtype("scan"): SubtypeConfusionCounts(2, 3)},
            {AttackSubtype("scan"): SubtypeConfusionCounts(5, 0)},
        ],
    )


def test_eligible_client_gets_utility_curve_and_thresholds() -> None:
    federation = build_federation_frontier(
        {ClientId("1"): _eligible_client_inputs()}, _GRID, _GUARDRAILS
    )
    frontier = federation.client_frontiers[ClientId("1")]
    assert frontier.eligible
    assert frontier.utility_curve is not None
    assert frontier.utility_curve.utility == pytest.approx((0.0, 0.5, 1.0))
    assert frontier.provisional_thresholds is not None
    assert len(frontier.provisional_thresholds) == len(_GRID)


def test_ineligible_client_has_no_curve() -> None:
    federation = build_federation_frontier(
        {ClientId("1"): _ineligible_client_inputs()}, _GRID, _GUARDRAILS
    )
    frontier = federation.client_frontiers[ClientId("1")]
    assert not frontier.eligible
    assert frontier.utility_curve is None
    assert frontier.provisional_thresholds is None


def test_fallback_rate_and_eligible_ids_mixed_population() -> None:
    federation = build_federation_frontier(
        {
            ClientId("eligible"): _eligible_client_inputs(),
            ClientId("fallback"): _ineligible_client_inputs(),
        },
        _GRID,
        _GUARDRAILS,
    )
    assert federation.fallback_rate == pytest.approx(0.5)
    assert federation.eligible_client_ids() == (ClientId("eligible"),)
    assert set(federation.utility_curves().keys()) == {ClientId("eligible")}


def test_empty_clients_rejected() -> None:
    with pytest.raises(ValueError):
        build_federation_frontier({}, _GRID, _GUARDRAILS)


def test_candidate_count_mismatch_rejected() -> None:
    inputs = ClientFrontierInputs(
        benign_frontier_scores=np.array([0.1]),
        subtype_validation_row_counts={AttackSubtype("scan"): 60, AttackSubtype("udp"): 60},
        subtype_confusion_by_candidate=[{AttackSubtype("scan"): SubtypeConfusionCounts(5, 5)}],
    )
    with pytest.raises(ValueError):
        build_federation_frontier({ClientId("1"): inputs}, _GRID, _GUARDRAILS)
