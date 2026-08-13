from __future__ import annotations

from dataclasses import dataclass

from fabrid.artifacts.allocation_store import AllocationArtifact, persist_allocation
from fabrid.artifacts.evaluation_store import persist_seed_budget_evaluation
from fabrid.artifacts.json_store import StoredJsonArtifact
from fabrid.artifacts.layout import ArtifactLayout
from fabrid.artifacts.result_store import StoredResultArtifact, write_result_records
from fabrid.domain.coordinates import AllocationCoordinate, ExperimentCoordinate
from fabrid.domain.enums import AllocationPolicy
from fabrid.pipeline.allocation import CompletedPolicyRun, SeedBudgetRun


@dataclass(frozen=True, slots=True)
class StoredPolicyAllocation:
    policy: AllocationPolicy
    artifact: StoredJsonArtifact


@dataclass(frozen=True, slots=True)
class MaterializedSeedBudget:
    coordinate: ExperimentCoordinate
    result_table: StoredResultArtifact
    evaluation_summary: StoredJsonArtifact
    allocations: tuple[StoredPolicyAllocation, ...]


def materialize_seed_budget(
    run: SeedBudgetRun,
    layout: ArtifactLayout,
) -> MaterializedSeedBudget:
    allocation_artifacts: list[StoredPolicyAllocation] = []
    for policy_run in run.policy_runs:
        if not isinstance(policy_run, CompletedPolicyRun):
            continue
        coordinate = AllocationCoordinate(
            experiment=run.evaluation.experiment,
            policy=policy_run.allocation.policy,
        )
        allocation_artifacts.append(
            StoredPolicyAllocation(
                policy=policy_run.allocation.policy,
                artifact=persist_allocation(
                    AllocationArtifact(
                        coordinate=coordinate,
                        allocation=policy_run.allocation,
                        solver=policy_run.solver,
                    ),
                    layout,
                ),
            )
        )

    return MaterializedSeedBudget(
        coordinate=run.evaluation.experiment,
        result_table=write_result_records(
            run.records,
            layout.result_path(run.evaluation.experiment),
        ),
        evaluation_summary=persist_seed_budget_evaluation(
            run.evaluation,
            layout,
        ),
        allocations=tuple(allocation_artifacts),
    )
