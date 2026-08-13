from __future__ import annotations

from dataclasses import dataclass

from fabrid.artifacts.allocation_store import AllocationArtifact, persist_allocation
from fabrid.artifacts.dataset_store import (
    StoredDatasetManifests,
    persist_dataset_manifests,
)
from fabrid.artifacts.evaluation_store import persist_seed_budget_evaluation
from fabrid.artifacts.json_store import StoredJsonArtifact
from fabrid.artifacts.protocol_store import persist_protocol_snapshot
from fabrid.artifacts.result_store import StoredResultArtifact, write_result_records
from fabrid.datasets.nbaiot.specification import NBAIOT_PRIMARY_POPULATION
from fabrid.domain.coordinates import AllocationCoordinate, ExperimentCoordinate
from fabrid.domain.enums import (
    AllocationPolicy,
    DatasetId,
    ExperimentId,
    ExperimentVariantId,
)
from fabrid.domain.identifiers import CampaignId
from fabrid.evaluation.results import SeedBudgetEvaluation
from fabrid.pipeline.allocation import (
    CompletedPolicyRun,
    SeedBudgetRun,
    load_seed_scores,
    run_seed_budget,
)
from fabrid.pipeline.context import PipelinePaths
from fabrid.pipeline.provenance import build_evaluation_provenance, resolve_git_commit
from fabrid.pipeline.scoring import materialize_detector_scores
from fabrid.pipeline.training import prepare_nbaiot_federation, train_detector_seed
from fabrid.protocol.models import FabridProtocol
from fabrid.protocol.specification import PROTOCOL


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


@dataclass(frozen=True, slots=True)
class MatchedBudgetCampaign:
    campaign_id: CampaignId
    protocol_snapshot: StoredJsonArtifact
    dataset_manifests: StoredDatasetManifests
    evaluations: tuple[SeedBudgetEvaluation, ...]
    artifacts: tuple[MaterializedSeedBudget, ...]


def _materialize_seed_budget(
    run: SeedBudgetRun,
    paths: PipelinePaths,
) -> MaterializedSeedBudget:
    layout = paths.artifacts
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


def run_matched_budget_campaign(
    campaign_id: CampaignId,
    paths: PipelinePaths,
    protocol: FabridProtocol = PROTOCOL,
) -> MatchedBudgetCampaign:
    git_commit = resolve_git_commit()
    layout = paths.artifacts
    protocol_snapshot = persist_protocol_snapshot(campaign_id, protocol, layout)
    prepared = prepare_nbaiot_federation(paths, protocol)
    dataset_manifests = persist_dataset_manifests(
        campaign_id=campaign_id,
        dataset_id=DatasetId.NBAIOT,
        feature_manifest=prepared.feature_manifest,
        split_manifest=prepared.split_manifest,
        layout=layout,
    )

    evaluations: list[SeedBudgetEvaluation] = []
    artifacts: list[MaterializedSeedBudget] = []
    for detector_seed in protocol.detector.seeds:
        trained = train_detector_seed(
            campaign_id=campaign_id,
            detector_seed=detector_seed,
            prepared=prepared,
            protocol=protocol,
            paths=paths,
        )
        stored_scores = materialize_detector_scores(campaign_id, trained, paths)
        provenance = build_evaluation_provenance(
            trained=trained,
            scores=stored_scores,
            dataset_manifests=dataset_manifests,
            protocol_snapshot=protocol_snapshot,
            git_commit=git_commit,
        )
        loaded_scores = load_seed_scores(
            campaign_id=campaign_id,
            detector_seed=detector_seed,
            population=NBAIOT_PRIMARY_POPULATION,
            paths=paths,
        )

        for budget_level in protocol.budgets:
            run = run_seed_budget(
                campaign_id=campaign_id,
                experiment_id=ExperimentId.MATCHED_BUDGET,
                variant_id=ExperimentVariantId.PRIMARY,
                detector_seed=detector_seed,
                budget_level=budget_level,
                scores=loaded_scores,
                protocol=protocol,
                provenance=provenance,
            )
            evaluations.append(run.evaluation)
            artifacts.append(_materialize_seed_budget(run, paths))

    return MatchedBudgetCampaign(
        campaign_id=campaign_id,
        protocol_snapshot=protocol_snapshot,
        dataset_manifests=dataset_manifests,
        evaluations=tuple(evaluations),
        artifacts=tuple(artifacts),
    )
