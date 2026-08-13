from __future__ import annotations

from dataclasses import dataclass

from fabrid.artifacts.dataset_store import (
    StoredDatasetManifests,
    persist_dataset_manifests,
)
from fabrid.artifacts.json_store import StoredJsonArtifact
from fabrid.artifacts.protocol_store import persist_protocol_snapshot
from fabrid.datasets.nbaiot.specification import NBAIOT_PRIMARY_POPULATION
from fabrid.domain.enums import (
    DatasetId,
    ExperimentId,
    ExperimentVariantId,
)
from fabrid.domain.identifiers import CampaignId
from fabrid.evaluation.results import SeedBudgetEvaluation
from fabrid.pipeline.allocation import load_seed_scores, run_seed_budget
from fabrid.pipeline.context import PipelinePaths
from fabrid.pipeline.materialization import (
    MaterializedSeedBudget,
    materialize_seed_budget,
)
from fabrid.pipeline.provenance import build_evaluation_provenance, resolve_git_commit
from fabrid.pipeline.scoring import materialize_detector_scores
from fabrid.pipeline.training import prepare_nbaiot_federation, train_detector_seed
from fabrid.protocol.models import FabridProtocol
from fabrid.protocol.specification import PROTOCOL


@dataclass(frozen=True, slots=True)
class MatchedBudgetCampaign:
    campaign_id: CampaignId
    protocol_snapshot: StoredJsonArtifact
    dataset_manifests: StoredDatasetManifests
    evaluations: tuple[SeedBudgetEvaluation, ...]
    artifacts: tuple[MaterializedSeedBudget, ...]


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
            artifacts.append(materialize_seed_budget(run, layout))

    return MatchedBudgetCampaign(
        campaign_id=campaign_id,
        protocol_snapshot=protocol_snapshot,
        dataset_manifests=dataset_manifests,
        evaluations=tuple(evaluations),
        artifacts=tuple(artifacts),
    )
