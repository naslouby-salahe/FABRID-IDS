"""Attack-subtype-disjoint generalization runs (roadmap section 58, Phase 14): resolve every
allocation policy using only one fold rotation's validation-fold attack subtypes, then evaluate
MacroRecall/WorstClientRecall/BUR against the held-out test-fold subtypes only. This measures
whether a policy generalizes to attack subtypes it never saw during allocation, not just to
unseen rows of subtypes it already saw.
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
from fabrid.config.attack_folds import (
    AttackFoldsConfig,
    BotnetFamily,
    BotnetFamilyDirection,
    FoldRotation,
)
from fabrid.config.protocol import SolverSettings, UtilityEligibilityGuardrails
from fabrid.evaluation.record_level import (
    AttackSubtype,
    ClientId,
    ClientWeight,
    FalsePositiveRate,
    TruePositiveRate,
    budget_usage_ratio,
    client_macro_recall,
    federation_fpr,
    federation_macro_recall,
    worst_client_recall,
)
from fabrid.frontier.builder import build_federation_frontier, restrict_to_subtypes
from fabrid.optimization.milp import SolverInvalidError
from fabrid.schemas.allocation import Allocation, AllocationPolicy
from fabrid.schemas.score_artifact import ScoreArtifact
from fabrid.scoring.frontier_inputs import (
    attack_test_scores_by_subtype,
    benign_final_cal_scores,
    benign_test_scores,
    build_client_frontier_inputs,
)


@dataclass(frozen=True, slots=True)
class RotationResult:
    macro_recall_by_policy: dict[AllocationPolicy, float] = field(
        default_factory=dict[AllocationPolicy, float]
    )
    worst_client_recall_by_policy: dict[AllocationPolicy, float] = field(
        default_factory=dict[AllocationPolicy, float]
    )
    bur_by_policy: dict[AllocationPolicy, float] = field(
        default_factory=dict[AllocationPolicy, float]
    )
    excluded_policies: dict[AllocationPolicy, str] = field(
        default_factory=dict[AllocationPolicy, str]
    )


def _evaluate_allocation_on_held_out_subtypes(
    allocation: Allocation,
    artifacts: Mapping[ClientId, ScoreArtifact],
    test_subtypes: frozenset[AttackSubtype],
    budget: float,
) -> tuple[float, float, float]:
    final_cal_scores = {c: benign_final_cal_scores(artifacts[c]) for c in allocation.decisions}
    thresholds = calibrate_final_thresholds(allocation, final_cal_scores)

    client_recall: dict[ClientId, float] = {}
    client_fpr: dict[ClientId, FalsePositiveRate] = {}
    for client_id, calibration in thresholds.items():
        threshold_value = calibration.threshold.value
        held_out_subtype_scores = {
            subtype: scores
            for subtype, scores in attack_test_scores_by_subtype(artifacts[client_id]).items()
            if subtype in test_subtypes
        }
        if not held_out_subtype_scores:
            continue
        recall_by_subtype = {
            subtype: TruePositiveRate(float((scores > threshold_value).mean()))
            for subtype, scores in held_out_subtype_scores.items()
        }
        client_recall[client_id] = client_macro_recall(recall_by_subtype)

        benign_scores = benign_test_scores(artifacts[client_id])
        false_positive = int(np.sum(benign_scores > threshold_value))
        client_fpr[client_id] = FalsePositiveRate(
            false_positive / benign_scores.shape[0] if benign_scores.shape[0] else 0.0
        )

    if not client_recall:
        raise ValueError("no client has any test row for this rotation's held-out attack subtypes")

    weight = {c: ClientWeight(1.0 / len(client_fpr)) for c in client_fpr}
    fed_fpr = federation_fpr(client_fpr, weight)
    bur = budget_usage_ratio(fed_fpr, budget)
    return federation_macro_recall(client_recall), worst_client_recall(client_recall), bur


def _run_policies_restricted_to_subtypes(
    artifacts: Mapping[ClientId, ScoreArtifact],
    alpha_grid: tuple[float, ...],
    guardrails: UtilityEligibilityGuardrails,
    validation_subtypes: frozenset[AttackSubtype],
    test_subtypes: frozenset[AttackSubtype],
    budget: float,
    alpha_max: float,
    solver_settings: SolverSettings,
) -> RotationResult:
    """Shared core for both attack-subtype-disjoint rotations and botnet-family-disjoint
    directions: resolve every deployable policy using only `validation_subtypes` for utility/
    eligibility, then evaluate MacroRecall/WorstClientRecall/BUR on `test_subtypes`.
    """
    client_inputs = {
        client_id: restrict_to_subtypes(
            build_client_frontier_inputs(artifact, alpha_grid), validation_subtypes
        )
        for client_id, artifact in artifacts.items()
    }
    federation = build_federation_frontier(client_inputs, alpha_grid, guardrails)
    eligible_ids = federation.eligible_client_ids()
    utility_curves = federation.utility_curves()
    weight = dict.fromkeys(artifacts, 1.0 / len(artifacts))
    eligible_weight = {c: weight[c] for c in eligible_ids}

    result = RotationResult()

    eq_fpr = allocate_equal_fpr(list(artifacts.keys()), budget, alpha_max)
    (
        result.macro_recall_by_policy[AllocationPolicy.EQ_FPR],
        result.worst_client_recall_by_policy[AllocationPolicy.EQ_FPR],
        result.bur_by_policy[AllocationPolicy.EQ_FPR],
    ) = _evaluate_allocation_on_held_out_subtypes(eq_fpr, artifacts, test_subtypes, budget)

    if eligible_ids:
        greedy = allocate_greedy(utility_curves, eligible_weight, budget, alpha_max)
        (
            result.macro_recall_by_policy[AllocationPolicy.GREEDY],
            result.worst_client_recall_by_policy[AllocationPolicy.GREEDY],
            result.bur_by_policy[AllocationPolicy.GREEDY],
        ) = _evaluate_allocation_on_held_out_subtypes(greedy, artifacts, test_subtypes, budget)

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
                result.bur_by_policy[policy],
            ) = _evaluate_allocation_on_held_out_subtypes(
                allocation, artifacts, test_subtypes, budget
            )

    return result


def run_attack_subtype_disjoint_rotation(
    artifacts: Mapping[ClientId, ScoreArtifact],
    alpha_grid: tuple[float, ...],
    guardrails: UtilityEligibilityGuardrails,
    fold_config: AttackFoldsConfig,
    rotation: FoldRotation,
    budget: float,
    alpha_max: float,
    solver_settings: SolverSettings,
) -> RotationResult:
    """Resolve every deployable policy using only `rotation`'s validation-fold subtypes, then
    evaluate MacroRecall/WorstClientRecall/BUR on `rotation`'s test-fold subtypes.
    """
    return _run_policies_restricted_to_subtypes(
        artifacts,
        alpha_grid,
        guardrails,
        frozenset(fold_config.validation_subtypes(rotation)),
        frozenset(fold_config.test_subtypes(rotation)),
        budget,
        alpha_max,
        solver_settings,
    )


def run_botnet_family_disjoint_direction(
    artifacts: Mapping[ClientId, ScoreArtifact],
    alpha_grid: tuple[float, ...],
    guardrails: UtilityEligibilityGuardrails,
    family_subtypes: Mapping[BotnetFamily, tuple[AttackSubtype, ...]],
    direction: BotnetFamilyDirection,
    budget: float,
    alpha_max: float,
    solver_settings: SolverSettings,
) -> RotationResult:
    """Resolve every deployable policy using only `direction.validation_family`'s subtypes,
    then evaluate MacroRecall/WorstClientRecall/BUR on `direction.test_family`'s subtypes.

    `artifacts` must already be restricted by the caller to the roadmap's 7 dual-family
    clients (both Mirai and BASHLITE present); this function does not filter clients by
    family availability, only attack subtypes by family membership.
    """
    return _run_policies_restricted_to_subtypes(
        artifacts,
        alpha_grid,
        guardrails,
        frozenset(family_subtypes[direction.validation_family]),
        frozenset(family_subtypes[direction.test_family]),
        budget,
        alpha_max,
        solver_settings,
    )
