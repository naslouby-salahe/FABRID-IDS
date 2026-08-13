from __future__ import annotations

import numpy as np

from fabrid.allocation.diagnostics.test_oracle import (
    OracleAccessToken,
    OracleAuthorization,
    allocate_test_oracle,
)
from fabrid.allocation.frontier import FederationFrontierInputs
from fabrid.allocation.frontier_inputs import (
    AttackSubtypeScores,
    FrontierScorePopulation,
    build_client_frontier_inputs,
    frontier_score_population,
)
from fabrid.allocation.problem import build_allocation_problem, merge_full_allocation
from fabrid.allocation.solver import not_applicable_solver_evidence
from fabrid.domain.coordinates import ExperimentCoordinate
from fabrid.domain.enums import (
    AllocationPolicy,
    DatasetId,
    ExperimentId,
    ExperimentVariantId,
    WeightMode,
)
from fabrid.domain.identifiers import CampaignId
from fabrid.domain.scores import ScoreVector
from fabrid.domain.values import DetectorSeed
from fabrid.evaluation.evaluator import EvaluationProvenance
from fabrid.pipeline.allocation import LoadedSeedScores, evaluate_policy, equal_client_weights
from fabrid.pipeline.diagnostics import DiagnosticPolicyEvidence
from fabrid.protocol.models import BudgetLevel, FabridProtocol


def _test_population(scores: LoadedSeedScores, client_index: int) -> FrontierScorePopulation:
    client = scores.clients[client_index]
    attack_records = client.evaluation.attack_test.records
    subtypes = tuple(
        sorted(
            {
                record.attack_subtype
                for record in attack_records
                if record.attack_subtype is not None
            },
            key=lambda subtype: subtype.value,
        )
    )
    attack_test = tuple(
        AttackSubtypeScores(
            subtype=subtype,
            scores=ScoreVector(
                np.fromiter(
                    (
                        record.score.value
                        for record in attack_records
                        if record.attack_subtype == subtype
                    ),
                    dtype=np.float64,
                )
            ),
        )
        for subtype in subtypes
    )
    base = frontier_score_population(client.frontier)
    return FrontierScorePopulation(
        client_id=client.client_id,
        benign_frontier=base.benign_frontier,
        attack_validation=attack_test,
    )


def run_test_oracle_diagnostic(
    campaign_id: CampaignId,
    detector_seed: DetectorSeed,
    budget: BudgetLevel,
    scores: LoadedSeedScores,
    protocol: FabridProtocol,
    provenance: EvaluationProvenance,
) -> DiagnosticPolicyEvidence:
    inputs = FederationFrontierInputs(
        tuple(
            build_client_frontier_inputs(_test_population(scores, index), protocol.alpha_grid)
            for index in range(len(scores.clients))
        )
    )
    weights = equal_client_weights(scores.population)
    problem = build_allocation_problem(
        inputs=inputs,
        weights=weights,
        budget=budget.value,
        eligibility=protocol.utility_eligibility,
        maximum_target_rate=protocol.alpha_grid.maximum,
    )
    eligible_population = problem.frontier.eligible_population()
    eligible_curves = problem.frontier.eligible_curves()
    solver = not_applicable_solver_evidence()
    eligible_allocation = None
    if eligible_population is not None and eligible_curves is not None:
        optimized = allocate_test_oracle(
            token=OracleAccessToken(OracleAuthorization.ACKNOWLEDGED_NON_DEPLOYABLE),
            test_utility_curves=eligible_curves,
            weights=problem.weights.subset(eligible_population),
            remaining_budget=problem.remaining_budget,
            settings=protocol.solver,
        )
        eligible_allocation = optimized.allocation
        solver = optimized.solver
    allocation = merge_full_allocation(
        AllocationPolicy.TEST_ORACLE,
        scores.population,
        problem.fallback,
        eligible_allocation,
    )
    coordinate = ExperimentCoordinate(
        campaign_id=campaign_id,
        experiment_id=ExperimentId.MATCHED_BUDGET,
        variant_id=ExperimentVariantId.PRIMARY,
        dataset_id=DatasetId.NBAIOT,
        detector_seed=detector_seed,
        budget_id=budget.budget_id,
        budget=budget.value,
        weight_mode=WeightMode.EQUAL_CLIENT,
    )
    evaluated = evaluate_policy(
        coordinate=coordinate,
        allocation=allocation,
        scores=scores,
        weights=problem.weights,
        solver=solver,
        provenance=provenance,
        fallback_rate=problem.frontier.fallback_rate,
    )
    summary = evaluated.result.summary
    return DiagnosticPolicyEvidence(
        detector_seed=detector_seed,
        budget=budget,
        policy=AllocationPolicy.TEST_ORACLE,
        macro_recall=summary.macro_recall,
        worst_client_recall=summary.worst_client_recall,
        federation_fpr=summary.federation_fpr,
        budget_usage=summary.budget_usage,
    )
