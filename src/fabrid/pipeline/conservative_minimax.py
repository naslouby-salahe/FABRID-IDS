from __future__ import annotations

from fabrid.allocation.conservative import build_conservative_utility_curve
from fabrid.allocation.contracts import Allocation, ClientUtilityCurves
from fabrid.allocation.fabrid_minimax import allocate_fabrid_minimax
from fabrid.allocation.frontier import client_eligibility
from fabrid.allocation.solver import SolverInvalidError
from fabrid.domain.coordinates import ExperimentCoordinate
from fabrid.domain.enums import (
    AllocationPolicy,
    DatasetId,
    EligibilityStatus,
    ExperimentId,
    ExperimentVariantId,
    SolverStatus,
    WeightMode,
)
from fabrid.domain.identifiers import CampaignId, FailureReason
from fabrid.domain.values import DetectorSeed
from fabrid.evaluation.evaluator import EvaluationProvenance
from fabrid.evaluation.results import ExcludedPolicyEvaluation, SeedBudgetEvaluation
from fabrid.pipeline.allocation import (
    CompletedPolicyRun,
    ExcludedPolicyRun,
    LoadedSeedScores,
    SeedBudgetRun,
    build_allocation_problem,
    evaluate_policy,
    merge_full_allocation,
)
from fabrid.protocol.models import BudgetLevel, FabridProtocol


def run_conservative_minimax_seed_budget(
    campaign_id: CampaignId,
    detector_seed: DetectorSeed,
    budget_level: BudgetLevel,
    scores: LoadedSeedScores,
    protocol: FabridProtocol,
    provenance: EvaluationProvenance,
) -> SeedBudgetRun:
    coordinate = ExperimentCoordinate(
        campaign_id=campaign_id,
        experiment_id=ExperimentId.CONSERVATIVE_MINIMAX,
        variant_id=ExperimentVariantId.CONSERVATIVE_MINIMAX,
        dataset_id=DatasetId.NBAIOT,
        detector_seed=detector_seed,
        budget_id=budget_level.budget_id,
        budget=budget_level.value,
        weight_mode=WeightMode.EQUAL_CLIENT,
    )
    problem = build_allocation_problem(
        scores=scores,
        protocol=protocol,
        budget=budget_level.value,
    )
    eligible_population = problem.frontier.eligible_population()
    if eligible_population is None:
        excluded = ExcludedPolicyEvaluation(
            policy=AllocationPolicy.FABRID_MINIMAX_CONSERVATIVE,
            status=SolverStatus.NOT_APPLICABLE,
            reason=FailureReason("no client is eligible for conservative Minimax sensitivity"),
        )
        run = ExcludedPolicyRun(excluded)
        return SeedBudgetRun(
            evaluation=SeedBudgetEvaluation(
                experiment=coordinate,
                policies=(excluded,),
            ),
            records=(),
            policy_runs=(run,),
        )

    conservative_curves = ClientUtilityCurves(
        tuple(
            build_conservative_utility_curve(
                inputs=inputs,
                guardrails=protocol.utility_eligibility,
                confidence=protocol.conservative_utility_confidence,
            )
            for inputs in problem.inputs.clients
            if client_eligibility(inputs, protocol.utility_eligibility)
            is EligibilityStatus.ELIGIBLE
        )
    )
    try:
        optimized = allocate_fabrid_minimax(
            utility_curves=conservative_curves,
            weights=problem.weights.subset(eligible_population),
            remaining_budget=problem.remaining_budget,
            settings=protocol.solver,
        )
    except SolverInvalidError as error:
        excluded = ExcludedPolicyEvaluation(
            policy=AllocationPolicy.FABRID_MINIMAX_CONSERVATIVE,
            status=SolverStatus.SOLVER_INVALID,
            reason=error.reason,
        )
        run = ExcludedPolicyRun(excluded)
        return SeedBudgetRun(
            evaluation=SeedBudgetEvaluation(
                experiment=coordinate,
                policies=(excluded,),
            ),
            records=(),
            policy_runs=(run,),
        )

    conservative_eligible = Allocation(
        policy=AllocationPolicy.FABRID_MINIMAX_CONSERVATIVE,
        decisions=optimized.allocation.decisions,
    )
    full_allocation = merge_full_allocation(
        policy=AllocationPolicy.FABRID_MINIMAX_CONSERVATIVE,
        population=scores.population,
        fallback=problem.fallback,
        eligible=conservative_eligible,
    )
    completed: CompletedPolicyRun = evaluate_policy(
        coordinate=coordinate,
        allocation=full_allocation,
        scores=scores,
        weights=problem.weights,
        solver=optimized.solver,
        provenance=provenance,
        fallback_rate=problem.frontier.fallback_rate,
    )
    return SeedBudgetRun(
        evaluation=SeedBudgetEvaluation(
            experiment=coordinate,
            policies=(completed.result.summary,),
        ),
        records=completed.result.records,
        policy_runs=(completed,),
    )
