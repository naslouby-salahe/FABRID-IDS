from __future__ import annotations

import numpy as np
import pytest

from fabrid.allocation.problem import (
    AllocationProblem,
    ClientFrontierInputs,
    EligibilityStatus,
    FederationFrontierInputs,
    FrontierScoreArtifacts,
    build_allocation_problem,
    build_client_frontier_inputs,
    client_eligibility,
    equal_client_weights,
)
from fabrid.config import LOCAL_TARGET_RATE_CAP, AnomalyScore, AttackSubtypeId

from .synthetic_federation import (
    ALPHA_GRID,
    eligibility_config,
    inputs_for,
    synthetic_client_scores,
    synthetic_population,
    synthetic_problem,
)


def test_build_client_frontier_inputs_candidates_follow_alpha_grid() -> None:
    inputs = inputs_for(("a",))
    client = inputs.clients[0]
    assert len(client.candidates) == len(ALPHA_GRID)
    for candidate, alpha in zip(client.candidates, ALPHA_GRID, strict=True):
        assert candidate.target_rate == pytest.approx(alpha)


def test_build_client_frontier_inputs_thresholds_match_reference_scan() -> None:
    from fabrid.detector.calibration import calibrate_threshold, sorted_order_statistic_threshold

    inputs = inputs_for(("a",))
    client = inputs.clients[0]
    assert client.benign_frontier_scores.size > 0
    sorted_benign = np.sort(client.benign_frontier_scores)
    subtype_scores = _reference_subtype_scores(client)
    for candidate in client.candidates:
        reference_threshold = calibrate_threshold(
            client.benign_frontier_scores, candidate.target_rate
        )
        assert (
            sorted_order_statistic_threshold(sorted_benign, candidate.target_rate)
            == reference_threshold
        )
        for confusion in candidate.subtypes:
            scores = subtype_scores[confusion.subtype]
            expected_tp = int(np.count_nonzero(scores > reference_threshold))
            assert confusion.counts.true_positive == expected_tp
            assert confusion.counts.false_negative == int(scores.size) - expected_tp


def _reference_subtype_scores(
    client: ClientFrontierInputs,
) -> dict[AttackSubtypeId, np.ndarray]:
    from collections import defaultdict

    grouped: defaultdict[AttackSubtypeId, list[AnomalyScore]] = defaultdict(list)
    _, validation = synthetic_client_scores(
        client.client_id,
        seed=1,
        attack_rows_per_subtype=300,
        subtypes=("bashlite_scan", "mirai_scan"),
    )
    for record in validation.records:
        if record.attack_subtype is not None:
            grouped[record.attack_subtype].append(record.score)
    return {subtype: np.asarray(scores, dtype=np.float64) for subtype, scores in grouped.items()}


def test_client_eligibility_requires_enough_subtypes() -> None:
    inputs = inputs_for(("a",), subtypes=("bashlite_scan",))
    assert client_eligibility(inputs.clients[0], eligibility_config()) is (
        EligibilityStatus.FALLBACK
    )


def test_client_eligibility_falls_back_on_sparse_validation() -> None:
    inputs = inputs_for(("a",), attack_rows_per_subtype=10)
    assert client_eligibility(inputs.clients[0], eligibility_config()) is (
        EligibilityStatus.FALLBACK
    )


def test_client_eligibility_passes_with_adequate_data() -> None:
    inputs = inputs_for(("a",))
    assert client_eligibility(inputs.clients[0], eligibility_config()) is (
        EligibilityStatus.ELIGIBLE
    )


def test_build_allocation_problem_covers_fallback_clients() -> None:
    problem = synthetic_problem(("a", "b"), 0.01)
    assert isinstance(problem, AllocationProblem)
    assert problem.budget == pytest.approx(0.01)
    assert problem.maximum_target_rate == pytest.approx(LOCAL_TARGET_RATE_CAP)
    assert problem.remaining_budget >= 0.0
    assert problem.frontier.fallback_rate == 0.0
    assert problem.fallback.target_for("a") is None


def test_with_eligible_curves_replaces_only_utility_values() -> None:
    problem = synthetic_problem(("a", "b"), 0.01)
    curves = problem.require_eligible_curves()
    replaced = problem.with_eligible_curves(curves)
    assert replaced.require_eligible_curves().clients == curves.clients
    assert replaced.budget == problem.budget
    assert replaced.remaining_budget == problem.remaining_budget


def test_allocation_problem_rejects_client_mismatch() -> None:
    inputs = inputs_for(("a",))
    population = synthetic_population(("a", "b"))
    weights = equal_client_weights(population)
    eligibility = eligibility_config()
    with pytest.raises(ValueError):
        build_allocation_problem(
            inputs,
            weights,
            budget=0.01,
            eligibility=eligibility,
            maximum_target_rate=LOCAL_TARGET_RATE_CAP,
        )


def test_fallback_targets_cover_eligibility_fallback_clients() -> None:
    frontier_a, validation_a = synthetic_client_scores("a", seed=6)
    frontier_b, validation_b = synthetic_client_scores("b", seed=7, attack_rows_per_subtype=5)
    inputs = FederationFrontierInputs(
        clients=(
            build_client_frontier_inputs(
                FrontierScoreArtifacts(benign_frontier=frontier_a, attack_validation=validation_a),
                ALPHA_GRID,
                calibration_row_count=frontier_a.row_count,
            ),
            build_client_frontier_inputs(
                FrontierScoreArtifacts(benign_frontier=frontier_b, attack_validation=validation_b),
                ALPHA_GRID,
                calibration_row_count=frontier_b.row_count,
            ),
        )
    )
    population = synthetic_population(("a", "b"))
    weights = equal_client_weights(population)
    problem = build_allocation_problem(
        inputs,
        weights,
        budget=0.01,
        eligibility=eligibility_config(),
        maximum_target_rate=LOCAL_TARGET_RATE_CAP,
    )
    assert problem.frontier.fallback_rate == pytest.approx(0.5)
    assert problem.fallback.target_for("a") is None
    assert problem.fallback.target_for("b") == pytest.approx(0.01)
    assert problem.remaining_budget == pytest.approx(0.01 - 0.5 * 0.01)
