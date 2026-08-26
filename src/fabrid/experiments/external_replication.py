from __future__ import annotations

import json
import logging
import os
import pickle
import shutil
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import cast

import numpy as np
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
    BudgetId,
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
    WorkerCount,
)
from fabrid.datasets.cic_iot_diad import (
    ExternalEvidenceAssessment,
    ExternalFederation,
    assess_external_evidence,
    device_row_census,
    prepare_external_federation,
)
from fabrid.datasets.registry import (
    AttackFeatureBlock,
    ClientPopulation,
    DeviceDataset,
    DeviceSplitPlan,
    FeatureMatrix,
    plan_device_splits,
    resolve_raw_dataset_root,
)
from fabrid.detector.autoencoder import Autoencoder, AutoencoderArchitecture, resolve_device
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
    persist_seed_budget,
    run_seed_budget,
)
from fabrid.validation.completion import CellState, verify_experiment
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
        resolve_device(),
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
    device = resolve_device()
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


@dataclass(frozen=True, slots=True)
class _ExternalPreparationInputs:
    cic_root: Path
    assessment: ExternalEvidenceAssessment


@dataclass(frozen=True, slots=True)
class _PendingExternalBudgets:
    entries: tuple[tuple[DetectorSeed, frozenset[BudgetId]], ...]

    def budget_ids_for(self, seed: DetectorSeed) -> frozenset[BudgetId]:
        for saved_seed, budget_ids in self.entries:
            if saved_seed == seed:
                return budget_ids
        raise KeyError(seed)


_spawn_prepared_external_run: _PreparedExternalRun | None = None


_EXTERNAL_FEDERATION_CACHE_VERSION = "external-federation-v1"
_CACHE_LOCK_TIMEOUT_SECONDS = 60 * 60
_CACHE_LOCK_POLL_SECONDS = 0.5


def _external_federation_cache_paths(paths: ArtifactPaths) -> tuple[Path, Path]:
    directory = paths.preprocessing_dir(DatasetId.CIC_IOT_DIAD) / "external_federation_cache"
    return directory / "federation.pkl", directory / "federation.lock"


def _external_federation_mmap_directory(cache_path: Path, cache_key: ArtifactDigest) -> Path:
    return cache_path.parent / f"federation_mmap_v3_{cache_key[:16]}"


def load_mmap_external_federation_cache(
    cache_path: Path,
    expected_key: ArtifactDigest,
) -> ExternalFederation | None:
    directory = _external_federation_mmap_directory(cache_path, expected_key)
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["cache_key"] != expected_key:
            return None
        if manifest["finite_validated"] is not True:
            return None
        devices = tuple(
            DeviceDataset(
                client_id=item["client_id"],
                benign_source_file=item["benign_source_file"],
                benign=FeatureMatrix.from_cached_values(
                    np.load(directory / item["benign_features"], mmap_mode="r")
                ),
                attacks=tuple(
                    AttackFeatureBlock(
                        subtype=block["subtype"],
                        source_file=block["source_file"],
                        features=FeatureMatrix.from_cached_values(
                            np.load(directory / block["features"], mmap_mode="r")
                        ),
                    )
                    for block in item["attacks"]
                ),
            )
            for item in manifest["devices"]
        )
        federation = ExternalFederation(
            devices=devices,
            kept_columns=tuple(manifest["kept_columns"]),
            dropped_rows=manifest["dropped_rows"],
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        logger.warning("[CACHE] external memory-mapped federation cache is unreadable")
        return None
    logger.info("[CACHE] reused memory-mapped external federation (%d devices)", len(devices))
    return federation


def save_mmap_external_federation_cache(
    cache_path: Path,
    cache_key: ArtifactDigest,
    federation: ExternalFederation,
) -> None:
    target = _external_federation_mmap_directory(cache_path, cache_key)
    if target.exists():
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".external_federation_", dir=cache_path.parent))
    try:
        serialized_devices = []
        for device_index, device in enumerate(federation.devices):
            benign_name = f"device_{device_index}_benign.npy"
            np.save(staging / benign_name, device.benign.values, allow_pickle=False)
            serialized_attacks = []
            for attack_index, block in enumerate(device.attacks):
                attack_name = f"device_{device_index}_attack_{attack_index}.npy"
                np.save(staging / attack_name, block.features.values, allow_pickle=False)
                serialized_attacks.append(
                    {
                        "subtype": block.subtype,
                        "source_file": block.source_file,
                        "features": attack_name,
                    }
                )
            serialized_devices.append(
                {
                    "client_id": device.client_id,
                    "benign_source_file": device.benign_source_file,
                    "benign_features": benign_name,
                    "attacks": serialized_attacks,
                }
            )
        manifest = {
            "cache_key": cache_key,
            "finite_validated": True,
            "kept_columns": list(federation.kept_columns),
            "dropped_rows": federation.dropped_rows,
            "devices": serialized_devices,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True), encoding="utf-8"
        )
        try:
            os.replace(staging, target)
        except FileExistsError:
            shutil.rmtree(staging, ignore_errors=True)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    logger.info(
        "[CACHE] saved memory-mapped external federation (%d devices)", len(federation.devices)
    )


def _external_federation_cache_key(
    application: ApplicationConfig,
    external: ExternalReplicationConfig,
    cic_root: Path,
    assessment: ExternalEvidenceAssessment,
) -> ArtifactDigest:
    return digest_text(
        (
            _EXTERNAL_FEDERATION_CACHE_VERSION,
            str(cic_root.resolve()),
            application.datasets.cic_iot_diad.model_dump_json(),
            str(external.feature_parse_threshold),
            ",".join(assessment.eligible_devices),
        )
    )


def load_external_federation_cache(
    cache_path: Path,
    expected_key: ArtifactDigest,
) -> ExternalFederation | None:
    if not cache_path.is_file():
        return None
    try:
        with cache_path.open("rb") as handle:
            cache_key, federation = pickle.load(handle)
    except (EOFError, OSError, pickle.UnpicklingError, ValueError):
        logger.warning("[CACHE] external federation cache is unreadable; rebuilding")
        return None
    if cache_key != expected_key or not isinstance(federation, ExternalFederation):
        logger.info("[CACHE] external federation cache key changed; rebuilding")
        return None
    logger.info("[CACHE] reused external federation (%d devices)", len(federation.devices))
    return federation


def save_external_federation_cache(
    cache_path: Path,
    cache_key: ArtifactDigest,
    federation: ExternalFederation,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_suffix(".tmp")
    with temporary_path.open("wb") as handle:
        pickle.dump((cache_key, federation), handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, cache_path)
    logger.info("[CACHE] saved external federation (%d devices)", len(federation.devices))


def _load_or_prepare_cached_external_federation(
    application: ApplicationConfig,
    paths: ArtifactPaths,
    external: ExternalReplicationConfig,
    cic_root: Path,
    assessment: ExternalEvidenceAssessment,
) -> ExternalFederation:
    cache_path, lock_path = _external_federation_cache_paths(paths)
    cache_key = _external_federation_cache_key(application, external, cic_root, assessment)
    mapped = load_mmap_external_federation_cache(cache_path, cache_key)
    if mapped is not None:
        return mapped
    cached = load_external_federation_cache(cache_path, cache_key)
    if cached is not None:
        save_mmap_external_federation_cache(cache_path, cache_key, cached)
        mapped = load_mmap_external_federation_cache(cache_path, cache_key)
        return mapped if mapped is not None else cached

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    owns_lock = False
    while not owns_lock:
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            mapped = load_mmap_external_federation_cache(cache_path, cache_key)
            if mapped is not None:
                return mapped
            cached = load_external_federation_cache(cache_path, cache_key)
            if cached is not None:
                save_mmap_external_federation_cache(cache_path, cache_key, cached)
                mapped = load_mmap_external_federation_cache(cache_path, cache_key)
                return mapped if mapped is not None else cached
            if time.monotonic() - start > _CACHE_LOCK_TIMEOUT_SECONDS:
                raise TimeoutError(
                    "timed out waiting for external federation cache build"
                ) from None
            time.sleep(_CACHE_LOCK_POLL_SECONDS)
        else:
            os.close(descriptor)
            owns_lock = True
    try:
        mapped = load_mmap_external_federation_cache(cache_path, cache_key)
        if mapped is not None:
            return mapped
        cached = load_external_federation_cache(cache_path, cache_key)
        if cached is not None:
            save_mmap_external_federation_cache(cache_path, cache_key, cached)
            mapped = load_mmap_external_federation_cache(cache_path, cache_key)
            return mapped if mapped is not None else cached
        federation = prepare_external_federation(
            cic_root,
            application.datasets.cic_iot_diad,
            external.feature_parse_threshold,
            assessment.eligible_devices,
        )
        save_external_federation_cache(cache_path, cache_key, federation)
        save_mmap_external_federation_cache(cache_path, cache_key, federation)
        mapped = load_mmap_external_federation_cache(cache_path, cache_key)
        return mapped if mapped is not None else federation
    finally:
        lock_path.unlink(missing_ok=True)


def _external_preparation_inputs(
    application: ApplicationConfig,
    paths: ArtifactPaths,
) -> _ExternalPreparationInputs:
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
    return _ExternalPreparationInputs(cic_root=cic_root, assessment=assessment)


def _prepare_confirmatory_external_run(
    application: ApplicationConfig,
    paths: ArtifactPaths,
    inputs: _ExternalPreparationInputs | None = None,
) -> _PreparedExternalRun | None:
    external = application.external_replication
    inputs = inputs or _external_preparation_inputs(application, paths)
    assessment = inputs.assessment
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
    federation = _load_or_prepare_cached_external_federation(
        application,
        paths,
        external,
        inputs.cic_root,
        assessment,
    )
    scorable = devices_with_benign_train(federation.devices, external)
    if not scorable:
        raise DatasetError(
            "no eligible external devices have a non-empty benign training split",
            path=inputs.cic_root,
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


def _run_external_seed(
    prepared: _PreparedExternalRun,
    paths: ArtifactPaths,
    external: ExternalReplicationConfig,
    seed: DetectorSeed,
    pending_budget_ids: frozenset[BudgetId],
) -> tuple[SeedBudgetRun, ...]:
    trained = load_or_train_external_seed(
        prepared.federation,
        prepared.split_plans,
        paths,
        external,
        seed,
    )
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
    runs: list[SeedBudgetRun] = []
    for budget_level in external.budgets:
        if budget_level.budget_id not in pending_budget_ids:
            continue
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
            prepared.protocol,
            provenance,
            equal_client_weights(loaded.population),
        )
        persist_seed_budget(paths, run)
        runs.append(run)
    return tuple(runs)


def _initialize_external_spawn_worker(
    application: ApplicationConfig,
    paths: ArtifactPaths,
    inputs: _ExternalPreparationInputs,
) -> None:
    global _spawn_prepared_external_run
    _spawn_prepared_external_run = _prepare_confirmatory_external_run(application, paths, inputs)
    if _spawn_prepared_external_run is None:
        raise RuntimeError("external spawn worker has no eligible federation")


def _run_external_seed_from_spawned_prepared(
    paths: ArtifactPaths,
    external: ExternalReplicationConfig,
    seed: DetectorSeed,
    pending_budget_ids: frozenset[BudgetId],
) -> tuple[SeedBudgetRun, ...]:
    if _spawn_prepared_external_run is None:
        raise RuntimeError("external spawn worker started without prepared federation")
    return _run_external_seed(
        _spawn_prepared_external_run,
        paths,
        external,
        seed,
        pending_budget_ids,
    )


def _pending_external_budgets_by_seed(
    application: ApplicationConfig,
    paths: ArtifactPaths,
) -> _PendingExternalBudgets:
    verification = verify_experiment(application, paths, ExperimentId.EXTERNAL_REPLICATION)
    return _PendingExternalBudgets(
        entries=tuple(
            (
                seed,
                frozenset(
                    cell.coordinate.budget_id
                    for cell in verification.cells
                    if cell.coordinate.detector_seed == seed and cell.state is not CellState.VALID
                ),
            )
            for seed in application.external_replication.detector.seeds
            if any(
                cell.coordinate.detector_seed == seed and cell.state is not CellState.VALID
                for cell in verification.cells
            )
        )
    )


def run_external_campaign(
    application: ApplicationConfig,
    paths: ArtifactPaths,
    workers: WorkerCount = 1,
) -> tuple[SeedBudgetRun, ...]:
    if workers < 1:
        raise ValueError("external replication workers must be at least one")
    external = application.external_replication
    pending_by_seed = _pending_external_budgets_by_seed(application, paths)
    if not pending_by_seed.entries:
        logger.info("[REUSE] all external replication cells are already verified")
        return ()
    pending_seeds = tuple(seed for seed, _ in pending_by_seed.entries)
    if workers == 1 or len(pending_seeds) == 1:
        prepared = _prepare_confirmatory_external_run(application, paths)
        if prepared is None:
            return ()
        completed_by_seed = {
            seed: _run_external_seed(
                prepared, paths, external, seed, pending_by_seed.budget_ids_for(seed)
            )
            for seed in pending_seeds
        }
    else:
        completed_by_seed = cast(dict[DetectorSeed, tuple[SeedBudgetRun, ...]], {})
        inputs = _external_preparation_inputs(application, paths)
        if inputs.assessment.evidence_level is not ReplicationEvidenceLevel.CONFIRMATORY:
            logger.info(
                "[GATE] external evidence %s; replication skipped",
                inputs.assessment.evidence_level.value,
            )
            return ()

        external_workers = min(workers, len(pending_seeds), 5)
        logger.info("[RUN] external CUDA seed workers=%d", external_workers)
        with ProcessPoolExecutor(
            max_workers=external_workers,
            mp_context=get_context("spawn"),
            initializer=_initialize_external_spawn_worker,
            initargs=(application, paths, inputs),
        ) as executor:
            futures = {
                executor.submit(
                    _run_external_seed_from_spawned_prepared,
                    paths,
                    external,
                    seed,
                    pending_by_seed.budget_ids_for(seed),
                ): seed
                for seed in pending_seeds
            }
            for future in as_completed(futures):
                completed_by_seed[futures[future]] = future.result()
    completed = tuple(run for seed in pending_seeds for run in completed_by_seed[seed])
    _warn_if_fallback_exceeds_limit(completed, external.eligibility.maximum_fallback_fraction)
    logger.info("[SAVE] external replication runs: %d", len(completed))
    return completed
