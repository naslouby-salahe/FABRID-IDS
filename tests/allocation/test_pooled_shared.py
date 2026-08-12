from __future__ import annotations

import numpy as np
import pytest

from fabrid.allocation.pooled_shared import (
    ClientValidationData,
    pooled_candidate_thresholds,
    select_pooled_shared_threshold,
)
from fabrid.evaluation.record_level import AttackSubtype, ClientId


def test_selects_threshold_maximizing_recall_within_budget() -> None:
    client_data = {
        ClientId("1"): ClientValidationData(
            benign_frontier_scores=np.array([0.1, 0.2, 0.3, 0.9]),
            attack_scores_by_subtype={AttackSubtype("scan"): np.array([0.5, 0.6, 0.95])},
        ),
    }
    weight = {ClientId("1"): 1.0}
    candidates = pooled_candidate_thresholds(client_data)
    # budget allows exactly one benign false alert (score 0.9) -> threshold 0.3 is feasible
    # (fpr = 1/4 = 0.25) and catches all three attacks, strictly better than the stricter
    # feasible-under-budget alternatives (0.9 or +inf, both recall < 1.0).
    result = select_pooled_shared_threshold(
        client_data, weight, budget=0.25, candidate_thresholds=candidates
    )
    assert result.threshold == pytest.approx(0.3)
    assert result.macro_recall == pytest.approx(1.0)
    assert result.weighted_fpr == pytest.approx(0.25)


def test_infeasible_budget_falls_back_to_zero_alerts() -> None:
    client_data = {
        ClientId("1"): ClientValidationData(
            benign_frontier_scores=np.array([0.1, 0.2]),
            attack_scores_by_subtype={AttackSubtype("scan"): np.array([0.9])},
        ),
    }
    weight = {ClientId("1"): 1.0}
    result = select_pooled_shared_threshold(
        client_data, weight, budget=0.0, candidate_thresholds=(np.inf,)
    )
    assert result.threshold == np.inf
    assert result.weighted_fpr == 0.0


def test_mismatched_client_sets_rejected() -> None:
    client_data = {
        ClientId("1"): ClientValidationData(
            benign_frontier_scores=np.array([0.1]),
            attack_scores_by_subtype={AttackSubtype("scan"): np.array([0.5])},
        ),
    }
    with pytest.raises(ValueError):
        select_pooled_shared_threshold(
            client_data, {ClientId("2"): 1.0}, budget=0.1, candidate_thresholds=(0.1,)
        )


def test_empty_candidates_rejected() -> None:
    client_data = {
        ClientId("1"): ClientValidationData(
            benign_frontier_scores=np.array([0.1]),
            attack_scores_by_subtype={AttackSubtype("scan"): np.array([0.5])},
        ),
    }
    with pytest.raises(ValueError):
        select_pooled_shared_threshold(
            client_data, {ClientId("1"): 1.0}, budget=0.1, candidate_thresholds=()
        )
