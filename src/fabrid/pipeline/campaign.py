from __future__ import annotations

from dataclasses import dataclass

from fabrid.analysis.gates import analyze_practical_gates
from fabrid.analysis.persistence import (
    StoredPracticalGates,
    StoredPrimaryInference,
    persist_practical_gates,
    persist_primary_inference,
)
from fabrid.analysis.primary import analyze_primary_inference
from fabrid.artifacts.dataset_store import (
    StoredDatasetManifests,
    persist_dataset_manifests,
)
from fabrid.artifacts.json_store import StoredJsonArtifact
from fabrid.artifacts.protocol_store import persist_protocol_snapshot
from fabrid.datasets.nbaiot.specification import NBAIOT_PRIMARY_POPULATION
from fabrid.domain.enums import DatasetId, ExperimentId, ExperimentVariantId
from fabrid.domain.identifiers import CampaignId
from fabrid.evaluation.results import SeedBudgetEvaluation
from fabrid.pipeline.allocation import load_seed_scores, run_seed_budget
from fabrid.pipeline.conservative_minimax import run_conservative_minimax_seed_budget
from fabrid.pipeline.context import PipelinePaths
from fabrid.pipeline.generalization import (
    run_attack_subtype_generalization_seed,
    run_botnet_family_generalization_seed,
)
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
class ExperimentExecution:
    evaluations: tuple[SeedBudgetEvaluation, ...]
    artifacts: tuple[MaterializedSeedBudget, ...]


@dataclass(frozen=True, slots=True)
class FabridCampaign:
    campaign_id: CampaignId
    protocol_snapshot: StoredJsonArtifact
    dataset_manifests: StoredDatasetManifests
    matched_budget: ExperimentExecution
    conservative_minimax: ExperimentExecution
    attack_subtype_disjoint: ExperimentExecution
    botnet_family_disjoint: ExperimentExecution
    primary_inference: StoredPrimaryInference
    practical_gates: StoredPracticalGates


def run_fabrid_campaign(
    campaign_id: CampaignId,
    paths: PipelinePaths,
    protocol: FabridProtocol = PROTOCOL,
) -> FabridCampaign:
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

    primary_evaluations: list[SeedBudgetEvaluation] = []
    primary_artifacts: list[MaterializedSeedBudget] = []
    conservative_evaluations: list[SeedBudgetEvaluation] = []
    conservative_artifacts: list[MaterializedSeedBudget] = []
    subtype_evaluations: list[SeedBudgetEvaluation] = []
    subtype_artifacts: list[MaterializedSeedBudget] = []
    family_evaluations: list[SeedBudgetEvaluation] = []
    family_artifacts: list[MaterializedSeedBudget] = []

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
            primary_run = run_seed_budget(
                campaign_id=campaign_id,
                experiment_id=ExperimentId.MATCHED_BUDGET,
                variant_id=ExperimentVariantId.PRIMARY,
                detector_seed=detector_seed,
                budget_level=budget_level,
                scores=loaded_scores,
                protocol=protocol,
                provenance=provenance,
            )
            primary_evaluations.append(primary_run.evaluation)
            primary_artifacts.append(materialize_seed_budget(primary_run, layout))

            conservative_run = run_conservative_minimax_seed_budget(
                campaign_id=campaign_id,
                detector_seed=detector_seed,
                budget_level=budget_level,
                scores=loaded_scores,
                protocol=protocol,
                provenance=provenance,
            )
            conservative_evaluations.append(conservative_run.evaluation)
            conservative_artifacts.append(
                materialize_seed_budget(conservative_run, layout)
            )

        subtype_execution = run_attack_subtype_generalization_seed(
            campaign_id=campaign_id,
            detector_seed=detector_seed,
            scores=loaded_scores,
            provenance=provenance,
            protocol=protocol,
            layout=layout,
        )
        subtype_evaluations.extend(subtype_execution.evaluations)
        subtype_artifacts.extend(subtype_execution.artifacts)

        family_execution = run_botnet_family_generalization_seed(
            campaign_id=campaign_id,
            detector_seed=detector_seed,
            scores=loaded_scores,
            provenance=provenance,
            protocol=protocol,
            layout=layout,
        )
        family_evaluations.extend(family_execution.evaluations)
        family_artifacts.extend(family_execution.artifacts)

    matched_budget = ExperimentExecution(
        evaluations=tuple(primary_evaluations),
        artifacts=tuple(primary_artifacts),
    )
    primary_inference = persist_primary_inference(
        campaign_id=campaign_id,
        inference=analyze_primary_inference(matched_budget.evaluations, protocol),
        layout=layout,
    )
    practical_gates = persist_practical_gates(
        campaign_id=campaign_id,
        analysis=analyze_practical_gates(
            matched_budget.evaluations,
            primary_inference.inference,
            protocol,
        ),
        layout=layout,
    )

    return FabridCampaign(
        campaign_id=campaign_id,
        protocol_snapshot=protocol_snapshot,
        dataset_manifests=dataset_manifests,
        matched_budget=matched_budget,
        conservative_minimax=ExperimentExecution(
            evaluations=tuple(conservative_evaluations),
            artifacts=tuple(conservative_artifacts),
        ),
        attack_subtype_disjoint=ExperimentExecution(
            evaluations=tuple(subtype_evaluations),
            artifacts=tuple(subtype_artifacts),
        ),
        botnet_family_disjoint=ExperimentExecution(
            evaluations=tuple(family_evaluations),
            artifacts=tuple(family_artifacts),
        ),
        primary_inference=primary_inference,
        practical_gates=practical_gates,
    )
