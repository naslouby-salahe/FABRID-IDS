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
    ClientWeight,
    FalsePositiveRate,
    TruePositiveRate,
    budget_usage_ratio,
    client_macro_recall,
    federation_fpr,
    federation_macro_recall,
    worst_client_recall,
)
from fabrid.frontier.builder import build_federation_frontier
from fabrid.frontier.conservative import build_conservative_utility_curve
from fabrid.optimization.milp import SolverInvalidError
from fabrid.schemas.allocation import Allocation, AllocationPolicy
from fabrid.schemas.result import ResultRow, SolverStatus, WeightMode
from fabrid.schemas.score_artifact import ScoreArtifact
from fabrid.scoring.frontier_inputs import (
    attack_test_scores_by_subtype,
    benign_final_cal_scores,
    benign_test_scores,
    build_client_frontier_inputs,
)


@dataclass(frozen=True, slots=True)
class SeedBudgetResult:
    seed: int
    budget: float
    fallback_rate: float
    macro_recall_by_policy: dict[AllocationPolicy, float] = field(
        default_factory=dict[AllocationPolicy, float]
    )
    worst_client_recall_by_policy: dict[AllocationPolicy, float] = field(
        default_factory=dict[AllocationPolicy, float]
    )
    fpr_by_policy: dict[AllocationPolicy, float] = field(
        default_factory=dict[AllocationPolicy, float]
    )
    bur_by_policy: dict[AllocationPolicy, float | None] = field(
        default_factory=dict[AllocationPolicy, float | None]
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
    allocation: Allocation, artifacts: Mapping[ClientId, ScoreArtifact], budget: float
) -> tuple[float, float, float, float | None]:
    """Returns (MacroRecall, WorstClientRecall, federation FPR, BUR) for `allocation`, evaluated
    on real ATTACK_TEST/BENIGN_TEST scores. FPR/BUR are computed over every client in the
    allocation (not just those with attack-test rows), using equal client weighting, matching
    `federation_fpr`'s primary-weighting convention (roadmap section 51). BUR is mathematically
    undefined at `budget=0` (division by zero) and is reported as `None` (not 0 or infinity) in
    that case, matching the zero-budget protocol test case (T13).
    """
    final_cal_scores = {c: benign_final_cal_scores(artifacts[c]) for c in allocation.decisions}
    thresholds = calibrate_final_thresholds(allocation, final_cal_scores)

    client_recall: dict[ClientId, float] = {}
    client_fpr: dict[ClientId, FalsePositiveRate] = {}
    for client_id, result in thresholds.items():
        threshold_value = result.threshold.value
        benign_scores = benign_test_scores(artifacts[client_id])
        false_positive = int(np.sum(benign_scores > threshold_value))
        client_fpr[client_id] = FalsePositiveRate(
            false_positive / benign_scores.shape[0] if benign_scores.shape[0] else 0.0
        )

        subtype_scores = attack_test_scores_by_subtype(artifacts[client_id])
        if not subtype_scores:
            continue
        client_recall[client_id] = _client_recall(threshold_value, subtype_scores)

    if not client_recall:
        raise ValueError("no client in this allocation has any attack-test rows")

    weight = {c: ClientWeight(1.0 / len(client_fpr)) for c in client_fpr}
    fed_fpr = federation_fpr(client_fpr, weight)
    bur = budget_usage_ratio(fed_fpr, budget) if budget > 0 else None
    return federation_macro_recall(client_recall), worst_client_recall(client_recall), fed_fpr, bur


@dataclass(frozen=True, slots=True)
class ResultRowProvenance:
    """Identifiers that are constant across every row of one seed x budget x policy cell."""

    experiment_id: str
    dataset_id: str
    budget_id: str
    weight_mode: WeightMode
    model_sha256: str
    split_sha256: str
    feature_sha256: str
    protocol_sha256: str
    git_commit: str
    solver_status: SolverStatus
    solver_objective: float | None = None
    solver_gap: float | None = None
    solver_runtime_ms: float | None = None


def build_result_rows(
    allocation: Allocation,
    artifacts: Mapping[ClientId, ScoreArtifact],
    weight: Mapping[ClientId, float],
    seed: int,
    budget: float,
    provenance: ResultRowProvenance,
) -> list[ResultRow]:
    """One row per (client, attack_subtype) covered by `allocation`, per the roadmap's primary
    result schema. `fp`/`tn`/`fpr` are the client's benign-test confusion counts, repeated across
    that client's subtype rows (there is one benign evaluation per client, not per subtype).
    """
    final_cal_scores = {c: benign_final_cal_scores(artifacts[c]) for c in allocation.decisions}
    thresholds = calibrate_final_thresholds(allocation, final_cal_scores)

    rows: list[ResultRow] = []
    for client_id, calibration in thresholds.items():
        artifact = artifacts[client_id]
        threshold_value = calibration.threshold.value

        benign_scores = benign_test_scores(artifact)
        false_positive = int(np.sum(benign_scores > threshold_value))
        true_negative = benign_scores.shape[0] - false_positive
        fpr = false_positive / benign_scores.shape[0] if benign_scores.shape[0] else 0.0

        subtype_scores = attack_test_scores_by_subtype(artifact)
        subtype_tprs = {
            subtype: TruePositiveRate(float((scores > threshold_value).mean()))
            for subtype, scores in subtype_scores.items()
        }
        macro_recall = client_macro_recall(subtype_tprs) if subtype_tprs else 0.0

        for subtype, scores in subtype_scores.items():
            true_positive = int(np.sum(scores > threshold_value))
            false_negative = scores.shape[0] - true_positive
            rows.append(
                ResultRow(
                    experiment_id=provenance.experiment_id,
                    dataset_id=provenance.dataset_id,
                    seed=seed,
                    budget_id=provenance.budget_id,
                    budget_value=budget,
                    weight_mode=provenance.weight_mode,
                    policy=allocation.policy,
                    client_id=client_id,
                    alpha_selected=calibration.alpha_selected,
                    threshold=threshold_value,
                    calibration_n=calibration.calibration_n,
                    nominal_weight=weight[client_id],
                    realized_weight=weight[client_id],
                    n_benign_test=benign_scores.shape[0],
                    n_attack_test=scores.shape[0],
                    attack_subtype=subtype,
                    true_positive=true_positive,
                    false_negative=false_negative,
                    false_positive=false_positive,
                    true_negative=true_negative,
                    fpr=fpr,
                    tpr=subtype_tprs[subtype].value,
                    macro_attack_recall=macro_recall,
                    false_alert_count=false_positive,
                    solver_status=provenance.solver_status,
                    solver_objective=provenance.solver_objective,
                    solver_gap=provenance.solver_gap,
                    solver_runtime_ms=provenance.solver_runtime_ms,
                    model_sha256=provenance.model_sha256,
                    score_sha256=artifact.sha256(),
                    split_sha256=provenance.split_sha256,
                    feature_sha256=provenance.feature_sha256,
                    protocol_sha256=provenance.protocol_sha256,
                    git_commit=provenance.git_commit,
                )
            )
    return rows


def run_seed_at_budget(
    artifacts: Mapping[ClientId, ScoreArtifact],
    alpha_grid: tuple[float, ...],
    guardrails: UtilityEligibilityGuardrails,
    budget: float,
    alpha_max: float,
    solver_settings: SolverSettings,
    seed: int,
) -> SeedBudgetResult:
    client_inputs = {
        client_id: build_client_frontier_inputs(artifact, alpha_grid)
        for client_id, artifact in artifacts.items()
    }
    federation = build_federation_frontier(client_inputs, alpha_grid, guardrails)
    result = SeedBudgetResult(seed=seed, budget=budget, fallback_rate=federation.fallback_rate)
    eligible_ids = federation.eligible_client_ids()
    utility_curves = federation.utility_curves()
    weight = dict.fromkeys(artifacts, 1.0 / len(artifacts))
    eligible_weight = {c: weight[c] for c in eligible_ids}

    eq_fpr = allocate_equal_fpr(list(artifacts.keys()), budget, alpha_max)
    (
        result.macro_recall_by_policy[AllocationPolicy.EQ_FPR],
        result.worst_client_recall_by_policy[AllocationPolicy.EQ_FPR],
        result.fpr_by_policy[AllocationPolicy.EQ_FPR],
        result.bur_by_policy[AllocationPolicy.EQ_FPR],
    ) = _evaluate_allocation(eq_fpr, artifacts, budget)

    if eligible_ids:
        greedy = allocate_greedy(utility_curves, eligible_weight, budget, alpha_max)
        (
            result.macro_recall_by_policy[AllocationPolicy.GREEDY],
            result.worst_client_recall_by_policy[AllocationPolicy.GREEDY],
            result.fpr_by_policy[AllocationPolicy.GREEDY],
            result.bur_by_policy[AllocationPolicy.GREEDY],
        ) = _evaluate_allocation(greedy, artifacts, budget)

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
                result.fpr_by_policy[policy],
                result.bur_by_policy[policy],
            ) = _evaluate_allocation(allocation, artifacts, budget)

    return result


def run_conservative_minimax_at_budget(
    artifacts: Mapping[ClientId, ScoreArtifact],
    alpha_grid: tuple[float, ...],
    guardrails: UtilityEligibilityGuardrails,
    budget: float,
    solver_settings: SolverSettings,
    confidence: float = 0.95,
) -> tuple[float, float, float, float | None] | None:
    """D005 (roadmap section 63): resolve FABRID_MINIMAX using the one-sided LCB utility curve
    instead of raw validation utility. Returns (MacroRecall, WorstClientRecall, federation FPR,
    BUR) evaluated on the same real ATTACK_TEST data as `run_seed_at_budget`, or None if
    solver-invalid or no client is eligible — same exclusion discipline as the raw-utility path.
    """
    client_inputs = {
        client_id: build_client_frontier_inputs(artifact, alpha_grid)
        for client_id, artifact in artifacts.items()
    }
    federation = build_federation_frontier(client_inputs, alpha_grid, guardrails)
    eligible_ids = federation.eligible_client_ids()
    if not eligible_ids:
        return None

    weight = dict.fromkeys(artifacts, 1.0 / len(artifacts))
    eligible_weight = {c: weight[c] for c in eligible_ids}
    conservative_curves = {
        client_id: build_conservative_utility_curve(
            client_id,
            alpha_grid,
            client_inputs[client_id].subtype_confusion_by_candidate,
            confidence,
        )
        for client_id in eligible_ids
    }

    try:
        allocation = allocate_fabrid_minimax(
            conservative_curves, eligible_weight, budget, solver_settings
        )
    except SolverInvalidError:
        return None

    return _evaluate_allocation(allocation, artifacts, budget)
