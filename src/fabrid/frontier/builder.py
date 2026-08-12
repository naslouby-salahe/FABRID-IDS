"""Federation frontier construction: eligibility gating, utility curves, provisional thresholds.

For each client, either:
- eligible: build its `ClientUtilityCurve` over the full candidate grid and a
  provisional threshold per candidate from `BENIGN_FRONTIER` scores, or
- ineligible: fall back to `alpha_k = min(B_FP, alpha_max)`, contributing no
  utility curve and reserving its budget share before eligible clients are
  optimized.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from fabrid.calibration.order_statistic import Threshold, calibrate_threshold
from fabrid.config.protocol import UtilityEligibilityGuardrails
from fabrid.data.eligibility import fallback_rate, is_client_eligible
from fabrid.evaluation.record_level import AttackSubtype, ClientId
from fabrid.frontier.utility import SubtypeConfusionCounts, build_utility_curve
from fabrid.schemas.allocation import ClientUtilityCurve


@dataclass(frozen=True, slots=True)
class ClientFrontierInputs:
    benign_frontier_scores: np.ndarray
    subtype_validation_row_counts: Mapping[AttackSubtype, int]
    subtype_confusion_by_candidate: Sequence[Mapping[AttackSubtype, SubtypeConfusionCounts]]


@dataclass(frozen=True, slots=True)
class ClientFrontier:
    client_id: ClientId
    eligible: bool
    utility_curve: ClientUtilityCurve | None
    provisional_thresholds: tuple[Threshold, ...] | None


@dataclass(frozen=True, slots=True)
class FederationFrontier:
    client_frontiers: Mapping[ClientId, ClientFrontier]
    fallback_rate: float

    def eligible_client_ids(self) -> tuple[ClientId, ...]:
        return tuple(
            client_id for client_id, frontier in self.client_frontiers.items() if frontier.eligible
        )

    def utility_curves(self) -> dict[ClientId, ClientUtilityCurve]:
        return {
            client_id: frontier.utility_curve
            for client_id, frontier in self.client_frontiers.items()
            if frontier.eligible and frontier.utility_curve is not None
        }


def _build_client_frontier(
    client_id: ClientId,
    inputs: ClientFrontierInputs,
    alpha_grid: tuple[float, ...],
    guardrails: UtilityEligibilityGuardrails,
) -> ClientFrontier:
    if len(inputs.subtype_confusion_by_candidate) != len(alpha_grid):
        raise ValueError(
            f"client {client_id}: subtype_confusion_by_candidate length "
            f"({len(inputs.subtype_confusion_by_candidate)}) must match alpha_grid length "
            f"({len(alpha_grid)})"
        )

    if not is_client_eligible(inputs.subtype_validation_row_counts, guardrails):
        return ClientFrontier(
            client_id=client_id, eligible=False, utility_curve=None, provisional_thresholds=None
        )

    utility_curve = build_utility_curve(
        client_id, alpha_grid, inputs.subtype_confusion_by_candidate
    )
    provisional_thresholds = tuple(
        calibrate_threshold(inputs.benign_frontier_scores, alpha) for alpha in alpha_grid
    )
    return ClientFrontier(
        client_id=client_id,
        eligible=True,
        utility_curve=utility_curve,
        provisional_thresholds=provisional_thresholds,
    )


def build_federation_frontier(
    client_inputs: Mapping[ClientId, ClientFrontierInputs],
    alpha_grid: tuple[float, ...],
    guardrails: UtilityEligibilityGuardrails,
) -> FederationFrontier:
    if not client_inputs:
        raise ValueError("build_federation_frontier requires at least one client")

    client_frontiers = {
        client_id: _build_client_frontier(client_id, inputs, alpha_grid, guardrails)
        for client_id, inputs in client_inputs.items()
    }
    eligible_count = sum(1 for frontier in client_frontiers.values() if frontier.eligible)
    return FederationFrontier(
        client_frontiers=client_frontiers,
        fallback_rate=fallback_rate(eligible_count, len(client_frontiers)),
    )
