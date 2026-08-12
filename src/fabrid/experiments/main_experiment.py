"""Main-experiment execution: one seed x one budget across the deployable policies.

Reuses frozen scores only (via `scoring.frontier_inputs`); never retrains or
rescoring. Solver-invalid coordinates are excluded rather than retried with
loosened tolerances, per protocol.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from fabrid.allocation.equal_fpr import allocate_equal_fpr
from fabrid.allocation.fabrid_macro import allocate_fabrid_macro
from fabrid.allocation.fabrid_minimax import allocate_fabrid_minimax
from fabrid.allocation.greedy import allocate_greedy
from fabrid.calibration.final_calibration import calibrate_final_thresholds
from fabrid.config.protocol import SolverSettings, UtilityEligibilityGuardrails
from fabrid.evaluation.record_level import (
    AttackSubtype,
    ClientId,
    TruePositiveRate,
    client_macro_recall,
    federation_macro_recall,
    worst_client_recall,
)
from fabrid.frontier.builder import build_federation_frontier
from fabrid.optimization.milp import SolverInvalidError
from fabrid.schemas.allocation import Allocation, AllocationPolicy
from fabrid.schemas.score_artifact import ScoreArtifact
from fabrid.scoring.frontier_inputs import (
    attack_test_scores_by_subtype,
    benign_final_cal_scores,
    build_client_frontier_inputs,
)


@dataclass(frozen=True, slots=True)
class SeedBudgetResult:
    seed: int
    budget: float
    macro_recall_by_policy: dict[AllocationPolicy, float] = field(
        default_factory=dict[AllocationPolicy, float]
    )
    worst_client_recall_by_policy: dict[AllocationPolicy, float] = field(
        default_factory=dict[AllocationPolicy, float]
    )
    excluded_policies: dict[AllocationPolicy, str] = field(
        default_factory=dict[AllocationPolicy, str]
    )


def _client_recall(
    threshold_value: float, subtype_scores: Mapping[AttackSubtype, np.ndarray]
) -> float:
    recall_by_subtype = {
        subtype: TruePositiveRate(float((scores > threshold_value).mean()))
        for subtype, scores in subtype_scores.items()
    }
    return client_macro_recall(recall_by_subtype)


def _evaluate_allocation(
    allocation: Allocation, artifacts: Mapping[ClientId, ScoreArtifact]
) -> tuple[float, float]:
    final_cal_scores = {c: benign_final_cal_scores(artifacts[c]) for c in allocation.decisions}
    thresholds = calibrate_final_thresholds(allocation, final_cal_scores)

    client_recall: dict[ClientId, float] = {}
    for client_id, result in thresholds.items():
        subtype_scores = attack_test_scores_by_subtype(artifacts[client_id])
        if not subtype_scores:
            continue
        client_recall[client_id] = _client_recall(result.threshold.value, subtype_scores)

    if not client_recall:
        raise ValueError("no client in this allocation has any attack-test rows")
    return federation_macro_recall(client_recall), worst_client_recall(client_recall)


def run_seed_at_budget(
    artifacts: Mapping[ClientId, ScoreArtifact],
    alpha_grid: tuple[float, ...],
    guardrails: UtilityEligibilityGuardrails,
    budget: float,
    alpha_max: float,
    solver_settings: SolverSettings,
    seed: int,
) -> SeedBudgetResult:
    result = SeedBudgetResult(seed=seed, budget=budget)

    client_inputs = {
        client_id: build_client_frontier_inputs(artifact, alpha_grid)
        for client_id, artifact in artifacts.items()
    }
    federation = build_federation_frontier(client_inputs, alpha_grid, guardrails)
    eligible_ids = federation.eligible_client_ids()
    utility_curves = federation.utility_curves()
    weight = dict.fromkeys(artifacts, 1.0 / len(artifacts))
    eligible_weight = {c: weight[c] for c in eligible_ids}

    eq_fpr = allocate_equal_fpr(list(artifacts.keys()), budget, alpha_max)
    (
        result.macro_recall_by_policy[AllocationPolicy.EQ_FPR],
        result.worst_client_recall_by_policy[AllocationPolicy.EQ_FPR],
    ) = _evaluate_allocation(eq_fpr, artifacts)

    if eligible_ids:
        greedy = allocate_greedy(utility_curves, eligible_weight, budget, alpha_max)
        (
            result.macro_recall_by_policy[AllocationPolicy.GREEDY],
            result.worst_client_recall_by_policy[AllocationPolicy.GREEDY],
        ) = _evaluate_allocation(greedy, artifacts)

        for policy, allocate in (
            (AllocationPolicy.FABRID_MACRO, allocate_fabrid_macro),
            (AllocationPolicy.FABRID_MINIMAX, allocate_fabrid_minimax),
        ):
            try:
                allocation = allocate(utility_curves, eligible_weight, budget, solver_settings)
            except SolverInvalidError as error:
                result.excluded_policies[policy] = str(error)
                continue
            (
                result.macro_recall_by_policy[policy],
                result.worst_client_recall_by_policy[policy],
            ) = _evaluate_allocation(allocation, artifacts)

    return result
