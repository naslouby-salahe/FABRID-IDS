from __future__ import annotations

from dataclasses import dataclass

from fabrid.allocation.contracts import Allocation
from fabrid.allocation.frontier import FederationFrontierInputs
from fabrid.allocation.frontier_inputs import (
    FrontierScorePopulation,
    build_client_frontier_inputs,
    frontier_score_population,
)
from fabrid.allocation.policies.fabrid_macro import allocate_fabrid_macro
from fabrid.allocation.policies.fabrid_minimax import allocate_fabrid_minimax
from fabrid.allocation.problem import build_allocation_problem, merge_full_allocation
from fabrid.allocation.solver import SolverInvalidError
from fabrid.allocation.weights import equal_client_weights
from fabrid.analysis.frontier_resampling import bootstrap_frontier_populations
from fabrid.analysis.stability import (
    AllocationStabilityAnalysis,
    allocation_sensitivity_seeds,
    summarize_allocations,
)
from fabrid.domain.coordinates import ExperimentCoordinate
from fabrid.domain.enums import (
    AllocationPolicy,
    DatasetId,
    EvidenceAvailability,
    ExperimentId,
    ExperimentVariantId,
    WeightMode,
)
from fabrid.domain.identifiers import CampaignId, FailureReason
from fabrid.domain.values import AnalysisSeed, DetectorSeed, RowCount
from fabrid.pipeline.allocation import LoadedSeedScores
from fabrid.protocol.models import BudgetLevel, FabridProtocol

_STABILITY_POLICIES = (
    AllocationPolicy.FABRID_MACRO,
    AllocationPolicy.FABRID_MINIMAX,
)


@dataclass(frozen=True, slots=True)
class AvailablePolicyStability:
    policy: AllocationPolicy
    availability: EvidenceAvailability
    expected_replicates: RowCount
    completed_replicates: RowCount
    solver_invalid_replicates: RowCount
    analysis: AllocationStabilityAnalysis

    def __post_init__(self) -> None:
        if self.availability is not EvidenceAvailability.AVAILABLE:
            raise ValueError("available stability must use AVAILABLE evidence state")
        if self.completed_replicates != self.expected_replicates:
            raise ValueError("available stability requires every preregistered replicate")
        if self.solver_invalid_replicates.value != 0:
            raise ValueError("available stability cannot contain solver-invalid replicates")


@dataclass(frozen=True, slots=True)
class UnavailablePolicyStability:
    policy: AllocationPolicy
    availability: EvidenceAvailability
    expected_replicates: RowCount
    completed_replicates: RowCount
    solver_invalid_replicates: RowCount
    reason: FailureReason

    def __post_init__(self) -> None:
        if self.availability is not EvidenceAvailability.INSUFFICIENT_EVIDENCE:
            raise ValueError(
                "unavailable stability must use INSUFFICIENT_EVIDENCE state"
            )


PolicyStability = AvailablePolicyStability | UnavailablePolicyStability


@dataclass(frozen=True, slots=True)
class SeedBudgetAllocationStability:
    experiment: ExperimentCoordinate
    policies: tuple[PolicyStability, ...]

    def __post_init__(self) -> None:
        policy_ids = tuple(policy.policy for policy in self.policies)
        if policy_ids != _STABILITY_POLICIES:
            raise ValueError("allocation stability must contain both FABRID policies in order")


@dataclass(frozen=True, slots=True)
class CampaignAllocationStability:
    cells: tuple[SeedBudgetAllocationStability, ...]


@dataclass(frozen=True, slots=True)
class _PolicyReplicates:
    policy: AllocationPolicy
    allocations: tuple[Allocation, ...]
    solver_invalid_replicates: RowCount


def _base_populations(
    scores: LoadedSeedScores,
) -> tuple[FrontierScorePopulation, ...]:
    return tuple(frontier_score_population(client.frontier) for client in scores.clients)


def _allocate_policy(
    policy: AllocationPolicy,
    populations: tuple[FrontierScorePopulation, ...],
    scores: LoadedSeedScores,
    budget_level: BudgetLevel,
    protocol: FabridProtocol,
) -> Allocation:
    inputs = FederationFrontierInputs(
        tuple(
            build_client_frontier_inputs(population, protocol.alpha_grid)
            for population in populations
        )
    )
    weights = equal_client_weights(scores.population)
    problem = build_allocation_problem(
        inputs=inputs,
        weights=weights,
        budget=budget_level.value,
        eligibility=protocol.utility_eligibility,
        maximum_target_rate=protocol.alpha_grid.maximum,
    )
    eligible_population = problem.frontier.eligible_population()
    eligible_curves = problem.frontier.eligible_curves()
    if eligible_population is None or eligible_curves is None:
        return merge_full_allocation(
            policy=policy,
            population=scores.population,
            fallback=problem.fallback,
            eligible=None,
        )

    eligible_weights = problem.weights.subset(eligible_population)
    optimized = (
        allocate_fabrid_macro(
            eligible_curves,
            eligible_weights,
            problem.remaining_budget,
            protocol.solver,
        )
        if policy is AllocationPolicy.FABRID_MACRO
        else allocate_fabrid_minimax(
            eligible_curves,
            eligible_weights,
            problem.remaining_budget,
            protocol.solver,
        )
    )
    return merge_full_allocation(
        policy=policy,
        population=scores.population,
        fallback=problem.fallback,
        eligible=optimized.allocation,
    )


def _summarize_policy(
    replicates: _PolicyReplicates,
    expected_replicates: RowCount,
) -> PolicyStability:
    completed_count = RowCount(len(replicates.allocations))
    if completed_count == expected_replicates:
        return AvailablePolicyStability(
            policy=replicates.policy,
            availability=EvidenceAvailability.AVAILABLE,
            expected_replicates=expected_replicates,
            completed_replicates=completed_count,
            solver_invalid_replicates=replicates.solver_invalid_replicates,
            analysis=summarize_allocations(replicates.allocations),
        )
    return UnavailablePolicyStability(
        policy=replicates.policy,
        availability=EvidenceAvailability.INSUFFICIENT_EVIDENCE,
        expected_replicates=expected_replicates,
        completed_replicates=completed_count,
        solver_invalid_replicates=replicates.solver_invalid_replicates,
        reason=FailureReason(
            "one or more preregistered bootstrap allocation solves were invalid"
        ),
    )


def run_seed_budget_allocation_stability(
    campaign_id: CampaignId,
    detector_seed: DetectorSeed,
    budget_level: BudgetLevel,
    scores: LoadedSeedScores,
    protocol: FabridProtocol,
) -> SeedBudgetAllocationStability:
    coordinate = ExperimentCoordinate(
        campaign_id=campaign_id,
        experiment_id=ExperimentId.MATCHED_BUDGET,
        variant_id=ExperimentVariantId.PRIMARY,
        dataset_id=DatasetId.NBAIOT,
        detector_seed=detector_seed,
        budget_id=budget_level.budget_id,
        budget=budget_level.value,
        weight_mode=WeightMode.EQUAL_CLIENT,
    )
    base = _base_populations(scores)
    replicate_seeds = allocation_sensitivity_seeds(
        protocol.allocation_sensitivity_replicates,
        AnalysisSeed(detector_seed.value),
    )
    macro_allocations: list[Allocation] = []
    minimax_allocations: list[Allocation] = []
    macro_invalid = RowCount(0)
    minimax_invalid = RowCount(0)

    for replicate_seed in replicate_seeds:
        resampled = bootstrap_frontier_populations(base, replicate_seed)
        try:
            macro_allocations.append(
                _allocate_policy(
                    policy=AllocationPolicy.FABRID_MACRO,
                    populations=resampled,
                    scores=scores,
                    budget_level=budget_level,
                    protocol=protocol,
                )
            )
        except SolverInvalidError:
            macro_invalid = RowCount(macro_invalid.value + 1)

        try:
            minimax_allocations.append(
                _allocate_policy(
                    policy=AllocationPolicy.FABRID_MINIMAX,
                    populations=resampled,
                    scores=scores,
                    budget_level=budget_level,
                    protocol=protocol,
                )
            )
        except SolverInvalidError:
            minimax_invalid = RowCount(minimax_invalid.value + 1)

    macro = _PolicyReplicates(
        policy=AllocationPolicy.FABRID_MACRO,
        allocations=tuple(macro_allocations),
        solver_invalid_replicates=macro_invalid,
    )
    minimax = _PolicyReplicates(
        policy=AllocationPolicy.FABRID_MINIMAX,
        allocations=tuple(minimax_allocations),
        solver_invalid_replicates=minimax_invalid,
    )
    return SeedBudgetAllocationStability(
        experiment=coordinate,
        policies=(
            _summarize_policy(macro, protocol.allocation_sensitivity_replicates),
            _summarize_policy(minimax, protocol.allocation_sensitivity_replicates),
        ),
    )
