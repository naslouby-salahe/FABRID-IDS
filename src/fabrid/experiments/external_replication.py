from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from pydantic import TypeAdapter, ValidationError

from fabrid.allocation.problem import FrontierScoreArtifacts, equal_client_weights
from fabrid.artifacts.json import digest_text, protocol_config_digest, read_typed_json
from fabrid.artifacts.paths import (
    ArtifactPaths,
    DetectorCoordinate,
    ExperimentCoordinate,
)
from fabrid.config import (
    EXTERNAL_SOURCE_FILE,
    AllocationPolicy,
    ApplicationConfig,
    ArtifactDigest,
    AttackSplit,
    AttackSubtypeId,
    BenignSplit,
    DatasetId,
    DetectorSeed,
    ExperimentId,
    ExperimentVariantId,
    ExternalReplicationConfig,
    FabridConfig,
    Label,
    PositiveInt,
    Probability,
    ReplicationEvidenceLevel,
    SourceRowIndex,
    WeightMode,
)
from fabrid.datasets.cic_iot_diad import (
    ExternalFederation,
    assess_external_evidence,
    device_row_census,
    prepare_external_federation,
)
from fabrid.datasets.registry import (
    ClientPopulation,
    DeviceDataset,
    DeviceSplitPlan,
    FeatureMatrix,
    plan_device_splits,
    resolve_raw_dataset_root,
)
from fabrid.detector.autoencoder import Autoencoder, AutoencoderArchitecture
from fabrid.detector.scoring import (
    ScoreCoordinate,
    ScorePartitionArtifact,
    ScoreRecord,
    build_score_partition,
    score_feature_matrix,
)
from fabrid.detector.training import (
    CheckpointMetadata,
    ClientScaler,
    ClientTrainingData,
    DetectorInputSpace,
    FeatureScaler,
    FederatedScalers,
    FederatedTrainingData,
    FederatedTrainingSettings,
    fit_feature_scaler,
    load_detector_checkpoint,
    save_detector_checkpoint,
    train_federated_autoencoder,
)
from fabrid.errors import DatasetError
from fabrid.evaluation.metrics import CompletedPolicyEvaluation
from fabrid.experiments.matched_budget import (
    ClientEvaluationArtifacts,
    EvaluationProvenance,
    LoadedClientScores,
    LoadedSeedScores,
    SeedBudgetRun,
    oracle_loaded,
    persist_seed_budget,
    run_seed_budget,
)
from fabrid.validation.datasets import SplitManifest, build_split_manifest
from fabrid.validation.reproducibility import report_progress, resolve_git_commit

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ExternalTrainedSeed:
    seed: DetectorSeed
    model: Autoencoder
    scalers: FederatedScalers
    metadata: CheckpointMetadata


def _rows(matrix: FeatureMatrix, start: SourceRowIndex, end: SourceRowIndex) -> FeatureMatrix:
    return FeatureMatrix(matrix.values[start:end])


def require_trained_feature_width(features: FeatureMatrix, trained: ExternalTrainedSeed) -> None:
    expected = trained.model.architecture.feature_count
    if features.feature_count != expected:
        raise DatasetError(
            f"external feature width {features.feature_count} does not match "
            f"trained detector feature width {expected}; "
            "physical-device replication cannot apply trained weights to a "
            "different packet feature space"
        )


def _external_architecture(
    federation: ExternalFederation,
    external: ExternalReplicationConfig,
) -> AutoencoderArchitecture:
    return AutoencoderArchitecture(
        feature_count=len(federation.kept_columns),
        hidden_layers=external.detector.hidden_layers,
    )


def _external_training_settings(
    external: ExternalReplicationConfig,
) -> FederatedTrainingSettings:
    return FederatedTrainingSettings(
        learning_rate=external.detector.learning_rate,
        local_epochs=external.detector.local_epochs,
        rounds=external.detector.rounds,
        batch_size=external.detector.batch_size,
    )


def _training_digest(
    federation: ExternalFederation,
    split_plans: tuple[DeviceSplitPlan, ...],
) -> ArtifactDigest:
    return digest_text(
        (
            *tuple(federation.kept_columns),
            *(
                f"{device.client_id}:{plan.benign.train_end}"
                for device, plan in zip(federation.devices, split_plans, strict=True)
            ),
        )
    )


def _external_checkpoint_matches(
    saved: CheckpointMetadata,
    external: ExternalReplicationConfig,
    seed: DetectorSeed,
    architecture: AutoencoderArchitecture,
    settings: FederatedTrainingSettings,
    paths: ArtifactPaths,
    coordinate: DetectorCoordinate,
    expected_training_digest: ArtifactDigest,
) -> bool:
    digest_path = paths.checkpoint_training_digest_path(coordinate)
    return (
        saved.dataset_id == external.dataset_id
        and saved.detector_seed == seed
        and saved.feature_count == architecture.feature_count
        and saved.hidden_layers == architecture.hidden_layers
        and saved.input_space is DetectorInputSpace.PER_CLIENT_ZSCORE
        and saved.training == settings
        and digest_path.exists()
        and digest_path.read_text().strip() == expected_training_digest
    )


def load_or_train_external_seed(
    federation: ExternalFederation,
    split_plans: tuple[DeviceSplitPlan, ...],
    paths: ArtifactPaths,
    external: ExternalReplicationConfig,
    seed: DetectorSeed,
) -> ExternalTrainedSeed:
    if len(federation.devices) != len(split_plans):
        raise ValueError("external federation and split plans must be aligned")
    coordinate = DetectorCoordinate(dataset_id=external.dataset_id, detector_seed=seed)
    architecture = _external_architecture(federation, external)
    settings = _external_training_settings(external)
    expected_training_digest = _training_digest(federation, split_plans)
    metadata_path = paths.checkpoint_metadata_path(coordinate)
    if metadata_path.exists():
        try:
            saved = read_typed_json(metadata_path, TypeAdapter(CheckpointMetadata))
        except (ValidationError, ValueError):
            logger.info("[REBUILD] external checkpoint seed=%d (invalid metadata)", seed)
        else:
            if _external_checkpoint_matches(
                saved,
                external,
                seed,
                architecture,
                settings,
                paths,
                coordinate,
                expected_training_digest,
            ):
                model, scalers, metadata = load_detector_checkpoint(paths, coordinate)
                logger.info("[REUSE] external checkpoint seed=%d", seed)
                return ExternalTrainedSeed(
                    seed=seed, model=model, scalers=scalers, metadata=metadata
                )
            logger.info("[REBUILD] external checkpoint seed=%d (protocol mismatch)", seed)
    unscaled_clients = tuple(
        ClientTrainingData(
            client_id=device.client_id,
            features=_rows(device.benign, 0, plan.benign.train_end),
        )
        for device, plan in zip(federation.devices, split_plans, strict=True)
    )
    scalers = FederatedScalers(
        clients=tuple(
            ClientScaler(client_id=client.client_id, scaler=fit_feature_scaler(client.features))
            for client in unscaled_clients
        )
    )
    scaled_clients = tuple(
        ClientTrainingData(
            client_id=client.client_id,
            features=scalers.for_client(client.client_id).transform(client.features),
        )
        for client in unscaled_clients
    )
    model = train_federated_autoencoder(
        FederatedTrainingData(clients=scaled_clients),
        architecture,
        settings,
        seed,
        torch.device("cpu"),
    )
    metadata = save_detector_checkpoint(
        paths, coordinate, model, scalers, settings, expected_training_digest
    )
    paths.checkpoint_training_digest_path(coordinate).write_text(expected_training_digest + "\n")
    logger.info("[BUILD] external checkpoint seed=%d", seed)
    return ExternalTrainedSeed(seed=seed, model=model, scalers=scalers, metadata=metadata)


def _scored_partition(
    coordinate: ScoreCoordinate,
    split: BenignSplit | AttackSplit,
    features: FeatureMatrix,
    scaler: FeatureScaler,
    trained: ExternalTrainedSeed,
    device: torch.device,
    subtype: AttackSubtypeId | None,
    label: Label,
    batch_size: PositiveInt,
) -> ScorePartitionArtifact:
    require_trained_feature_width(features, trained)
    scaled = scaler.transform(features)
    scores = score_feature_matrix(trained.model, scaled, device, batch_size)
    return build_score_partition(coordinate, split, scores, label, EXTERNAL_SOURCE_FILE, subtype)


def external_split_plans(
    federation: ExternalFederation,
    external: ExternalReplicationConfig,
) -> tuple[DeviceSplitPlan, ...]:
    return tuple(
        plan_device_splits(device, external.benign_splits, external.attack_split)
        for device in federation.devices
    )


def devices_with_benign_train(
    devices: tuple[DeviceDataset, ...],
    external: ExternalReplicationConfig,
) -> tuple[DeviceDataset, ...]:
    return tuple(
        device
        for device in devices
        if plan_device_splits(
            device, external.benign_splits, external.attack_split
        ).benign.train_end
        > 0
    )


def external_loaded_scores(
    federation: ExternalFederation,
    plans: tuple[DeviceSplitPlan, ...],
    seed: DetectorSeed,
    trained: ExternalTrainedSeed,
    dataset_id: DatasetId,
    batch_size: PositiveInt,
    paths: ArtifactPaths,
) -> LoadedSeedScores:
    device = torch.device("cpu")
    clients: list[LoadedClientScores] = []
    total_devices = len(federation.devices)
    for device_index, device_dataset in enumerate(federation.devices):
        client_id = device_dataset.client_id
        report_progress(
            logger,
            paths,
            f"external scoring seed {seed}",
            device_index + 1,
            total_devices,
            detail=client_id,
        )
        benign_matrix = device_dataset.benign
        plan = plans[device_index]
        benign_plan = plan.benign
        if benign_plan.train_end == 0:
            continue
        require_trained_feature_width(benign_matrix, trained)
        scaler = trained.scalers.for_client(client_id)
        coordinate = ScoreCoordinate(dataset_id=dataset_id, detector_seed=seed, client_id=client_id)
        frontier = _scored_partition(
            coordinate,
            BenignSplit.FRONTIER,
            _rows(benign_matrix, benign_plan.train_end, benign_plan.frontier_end),
            scaler,
            trained,
            device,
            None,
            Label.BENIGN,
            batch_size,
        )
        final_cal = _scored_partition(
            coordinate,
            BenignSplit.FINAL_CAL,
            _rows(benign_matrix, benign_plan.frontier_end, benign_plan.final_cal_end),
            scaler,
            trained,
            device,
            None,
            Label.BENIGN,
            batch_size,
        )
        benign_test = _scored_partition(
            coordinate,
            BenignSplit.TEST,
            _rows(benign_matrix, benign_plan.final_cal_end, benign_matrix.row_count),
            scaler,
            trained,
            device,
            None,
            Label.BENIGN,
            batch_size,
        )
        validation_records: list[ScoreRecord] = []
        test_records: list[ScoreRecord] = []
        for block in device_dataset.attacks:
            boundary = plan.attack_boundary(block.subtype)
            validation_artifact = _scored_partition(
                coordinate,
                AttackSplit.VALIDATION,
                _rows(block.features, 0, boundary.validation_end),
                scaler,
                trained,
                device,
                block.subtype,
                Label.ATTACK,
                batch_size,
            )
            test_artifact = _scored_partition(
                coordinate,
                AttackSplit.TEST,
                _rows(block.features, boundary.validation_end, block.features.row_count),
                scaler,
                trained,
                device,
                block.subtype,
                Label.ATTACK,
                batch_size,
            )
            validation_records.extend(validation_artifact.records)
            test_records.extend(test_artifact.records)
        attack_validation = ScorePartitionArtifact(
            coordinate=coordinate,
            split=AttackSplit.VALIDATION,
            records=tuple(validation_records),
        )
        attack_test = ScorePartitionArtifact(
            coordinate=coordinate,
            split=AttackSplit.TEST,
            records=tuple(test_records),
        )
        clients.append(
            LoadedClientScores(
                client_id=client_id,
                frontier=FrontierScoreArtifacts(
                    benign_frontier=frontier, attack_validation=attack_validation
                ),
                evaluation=ClientEvaluationArtifacts(
                    client_id=client_id,
                    final_calibration=final_cal,
                    benign_test=benign_test,
                    attack_test=attack_test,
                ),
            )
        )
    return LoadedSeedScores(clients=tuple(clients))


def _external_provenance(
    federation: ExternalFederation,
    protocol: FabridConfig,
    trained: ExternalTrainedSeed,
    score_digest: ArtifactDigest,
    split_manifest: SplitManifest,
) -> EvaluationProvenance:
    return EvaluationProvenance(
        model_sha256=trained.metadata.model_digest,
        score_sha256=score_digest,
        split_sha256=split_manifest.digest,
        feature_sha256=digest_text(tuple(federation.kept_columns)),
        protocol_sha256=protocol_config_digest(protocol),
        git_commit=resolve_git_commit(),
    )


@dataclass(frozen=True, slots=True)
class _PreparedExternalRun:
    federation: ExternalFederation
    protocol: FabridConfig
    split_plans: tuple[DeviceSplitPlan, ...]
    split_manifest: SplitManifest
    dataset_id: DatasetId


def _prepare_confirmatory_external_run(
    application: ApplicationConfig,
    paths: ArtifactPaths,
) -> _PreparedExternalRun | None:
    external = application.external_replication
    cic_root = resolve_raw_dataset_root(
        paths.raw_data_root, external.dataset_id, application.datasets
    )
    census = device_row_census(
        cic_root,
        application.datasets.cic_iot_diad,
    )
    eligibility = external.eligibility
    assessment = assess_external_evidence(
        census,
        eligibility.minimum_benign_rows,
        eligibility.minimum_attack_rows,
        eligibility.minimum_eligible_clients,
    )
    if assessment.evidence_level is not ReplicationEvidenceLevel.CONFIRMATORY:
        logger.info(
            "[GATE] external evidence %s; replication skipped",
            assessment.evidence_level.value,
        )
        return None
    logger.info(
        "[GATE] external evidence CONFIRMATORY (%d eligible devices)",
        len(assessment.eligible_devices),
    )
    federation = prepare_external_federation(
        cic_root,
        application.datasets.cic_iot_diad,
        external.feature_parse_threshold,
        assessment.eligible_devices,
    )
    scorable = devices_with_benign_train(federation.devices, external)
    if not scorable:
        raise DatasetError(
            "no eligible external devices have a non-empty benign training split",
            path=cic_root,
        )
    federation = ExternalFederation(
        devices=scorable,
        kept_columns=federation.kept_columns,
        dropped_rows=federation.dropped_rows,
    )
    split_plans = external_split_plans(federation, external)
    dataset_id = external.dataset_id
    return _PreparedExternalRun(
        federation=federation,
        protocol=external.protocol_for(application.protocol),
        split_plans=split_plans,
        split_manifest=build_split_manifest(
            dataset_id,
            ClientPopulation(tuple(device.client_id for device in federation.devices)),
            split_plans,
        ),
        dataset_id=dataset_id,
    )


def _warn_if_fallback_exceeds_limit(
    runs: tuple[SeedBudgetRun, ...], maximum_fallback_fraction: Probability
) -> None:
    fallback_rates = tuple(
        policy.fallback_rate
        for run in runs
        for policy in (run.evaluation.policy(AllocationPolicy.FABRID_MACRO),)
        if isinstance(policy, CompletedPolicyEvaluation)
    )
    if not fallback_rates:
        return
    worst_fallback = max(fallback_rates)
    if worst_fallback > maximum_fallback_fraction:
        logger.warning(
            "[GATE] executed fallback rate %.3f exceeds the %.3f limit; "
            "external replication is supportive, not confirmatory",
            worst_fallback,
            maximum_fallback_fraction,
        )


def run_external_campaign(
    application: ApplicationConfig,
    paths: ArtifactPaths,
) -> tuple[SeedBudgetRun, ...]:
    prepared = _prepare_confirmatory_external_run(application, paths)
    if prepared is None:
        return ()
    external = application.external_replication
    trained_seeds = tuple(
        load_or_train_external_seed(
            prepared.federation,
            prepared.split_plans,
            paths,
            external,
            seed,
        )
        for seed in external.detector.seeds
    )
    trained_by_seed = {item.seed: item for item in trained_seeds}
    runs: list[SeedBudgetRun] = []
    total_cells = len(prepared.protocol.seeds) * len(external.budgets)
    cell_index = 0
    for seed in prepared.protocol.seeds:
        trained = trained_by_seed[seed]
        loaded = external_loaded_scores(
            prepared.federation,
            prepared.split_plans,
            seed,
            trained,
            prepared.dataset_id,
            external.detector.score_batch_size,
            paths,
        )
        provenance = _external_provenance(
            prepared.federation,
            prepared.protocol,
            trained,
            loaded.digest(),
            prepared.split_manifest,
        )
        oracle_scores = oracle_loaded(loaded)
        for budget_level in external.budgets:
            cell_index += 1
            report_progress(
                logger,
                paths,
                "external replication cells",
                cell_index,
                total_cells,
                detail=f"seed {seed} budget {budget_level.budget_id.value}",
            )
            run = run_seed_budget(
                ExperimentCoordinate(
                    experiment_id=ExperimentId.EXTERNAL_REPLICATION,
                    variant_id=ExperimentVariantId.EXTERNAL_PRIMARY,
                    dataset_id=prepared.dataset_id,
                    detector_seed=seed,
                    budget_id=budget_level.budget_id,
                    budget=budget_level.value,
                    weight_mode=WeightMode.EQUAL_CLIENT,
                ),
                loaded,
                oracle_scores,
                prepared.protocol,
                provenance,
                equal_client_weights(loaded.population),
            )
            persist_seed_budget(paths, run)
            runs.append(run)
    completed = tuple(runs)
    _warn_if_fallback_exceeds_limit(completed, external.eligibility.maximum_fallback_fraction)
    logger.info("[SAVE] external replication runs: %d", len(completed))
    return completed
