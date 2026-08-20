from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from fabrid.artifacts.json import digest_file, digest_text, read_typed_json, write_typed_json
from fabrid.artifacts.parquet import write_parquet_frame
from fabrid.artifacts.paths import ArtifactPaths, DetectorCoordinate, ScoreCoordinate
from fabrid.config import (
    PREPROCESSED_SOURCE_FILE,
    ApplicationConfig,
    ArtifactDigest,
    AttackSplit,
    AttackSubtypeId,
    BenignSplit,
    ClientId,
    ColumnName,
    DatasetId,
    DetectorSeed,
    EnvironmentText,
    FabridConfig,
    FeatureCount,
    Identifier,
    Label,
    PreprocessedColumn,
    PreprocessingStatus,
    ProtocolVersion,
    RowCount,
)
from fabrid.datasets.nbaiot import read_device_directory
from fabrid.datasets.registry import (
    AttackSplitBoundary,
    BenignSplitBoundaries,
    ClientPopulation,
    DeviceSplitPlan,
    FeatureMatrix,
    build_dataset_registry,
    plan_device_splits,
)
from fabrid.detector.autoencoder import Autoencoder, AutoencoderArchitecture, resolve_device
from fabrid.detector.scoring import (
    ScorePartitionArtifact,
    ScoreRecord,
    build_score_partition,
    persist_score_partition,
    score_feature_matrix,
)
from fabrid.detector.training import (
    CheckpointMetadata,
    ClientScaler,
    ClientTrainingData,
    DetectorInputSpace,
    FederatedScalers,
    FederatedTrainingData,
    FederatedTrainingSettings,
    fit_feature_scaler,
    load_detector_checkpoint,
    save_detector_checkpoint,
    train_federated_autoencoder,
)
from fabrid.errors import DatasetError
from fabrid.execution.application import ApplicationContext
from fabrid.experiments.matched_budget import LoadedSeedScores, load_seed_scores
from fabrid.validation.datasets import (
    FeatureManifest,
    PreprocessingProvenance,
    SplitManifest,
    build_split_manifest,
    preprocessing_relevant_config,
)
from fabrid.validation.reproducibility import now_iso, report_progress

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PreparedFederation:
    population: ClientPopulation
    feature_manifest: FeatureManifest
    split_manifest: SplitManifest
    provenance: PreprocessingProvenance
    raw_digest: ArtifactDigest
    config_digest: ArtifactDigest


@dataclass(frozen=True, slots=True)
class TrainedDetectorSeed:
    coordinate: DetectorCoordinate
    model: Autoencoder
    scalers: FederatedScalers
    metadata: CheckpointMetadata


class ScoreProvenance(BaseModel):
    model_config = ConfigDict(frozen=True, revalidate_instances="never")
    checkpoint_digest: ArtifactDigest
    preprocessing_digest: ArtifactDigest
    scoring_config_digest: ArtifactDigest


_SCORE_PROVENANCE_ADAPTER = TypeAdapter(ScoreProvenance)


@dataclass(frozen=True, slots=True)
class StoredSeedScores:
    seed: DetectorSeed
    loaded: LoadedSeedScores
    score_sha256: ArtifactDigest


def _config_digest(application: ApplicationConfig) -> ArtifactDigest:
    relevant = preprocessing_relevant_config(application)
    return digest_text((relevant.model_dump_json(),))


def _scoring_config_digest(config: FabridConfig) -> ArtifactDigest:
    return digest_text(
        (
            str(config.detector.score_batch_size),
            str(config.scoring.ensemble_score_feature),
        )
    )


def _raw_digest(context: ApplicationContext, dataset_id: DatasetId) -> ArtifactDigest:
    from fabrid.validation.datasets import raw_dataset_directory_digest

    return raw_dataset_directory_digest(context.raw_dataset_root(dataset_id))


def _feature_columns(feature_count: FeatureCount, prefix: Identifier) -> tuple[ColumnName, ...]:
    return tuple(f"{prefix}{index}" for index in range(feature_count))


def _repeated_text(token: EnvironmentText, row_count: RowCount) -> np.ndarray:
    return np.repeat(np.asarray(token, dtype=object), row_count)


def _benign_split_values(boundaries: BenignSplitBoundaries) -> np.ndarray:
    counts = boundaries.counts()
    return np.concatenate(
        (
            _repeated_text(BenignSplit.TRAIN.value, counts.train),
            _repeated_text(BenignSplit.FRONTIER.value, counts.frontier),
            _repeated_text(BenignSplit.FINAL_CAL.value, counts.final_cal),
            _repeated_text(BenignSplit.TEST.value, counts.test),
        )
    )


def _attack_split_values(boundary: AttackSplitBoundary) -> np.ndarray:
    counts = boundary.counts()
    return np.concatenate(
        (
            _repeated_text(AttackSplit.VALIDATION.value, counts.validation),
            _repeated_text(AttackSplit.TEST.value, counts.test),
        )
    )


def _build_preprocessing_frame(
    context: ApplicationContext,
    population: ClientPopulation,
) -> tuple[pl.DataFrame, tuple[DeviceSplitPlan, ...], FeatureManifest]:
    application = context.config
    config = application.protocol
    dataset_root = context.raw_dataset_root(config.dataset_id)
    client_id_parts: list[np.ndarray] = []
    source_file_parts: list[np.ndarray] = []
    source_row_parts: list[np.ndarray] = []
    split_parts: list[np.ndarray] = []
    subtype_values: list[AttackSubtypeId | None] = []
    feature_parts: list[np.ndarray] = []
    split_plans: list[DeviceSplitPlan] = []
    for client_id in population.clients:
        device = read_device_directory(
            client_id, dataset_root / client_id, application.datasets.nbaiot
        )
        plan = plan_device_splits(device, config.benign_splits, config.attack_split)
        split_plans.append(plan)
        client_row_count = device.benign.row_count + sum(
            block.features.row_count for block in device.attacks
        )
        client_matrix = np.empty((client_row_count, device.benign.feature_count), dtype=np.float64)
        row_offset = 0
        benign_rows = device.benign.row_count
        client_id_parts.append(_repeated_text(client_id, benign_rows))
        source_file_parts.append(_repeated_text(device.benign_source_file, benign_rows))
        source_row_parts.append(np.arange(benign_rows, dtype=np.int64))
        split_parts.append(_benign_split_values(plan.benign))
        subtype_values.extend([None] * benign_rows)
        client_matrix[row_offset : row_offset + benign_rows] = device.benign.values
        row_offset += benign_rows
        for block in device.attacks:
            attack_rows = block.features.row_count
            client_id_parts.append(_repeated_text(client_id, attack_rows))
            source_file_parts.append(_repeated_text(block.source_file, attack_rows))
            source_row_parts.append(np.arange(attack_rows, dtype=np.int64))
            split_parts.append(_attack_split_values(plan.attack_boundary(block.subtype)))
            subtype_values.extend([block.subtype] * attack_rows)
            client_matrix[row_offset : row_offset + attack_rows] = block.features.values
            row_offset += attack_rows
        feature_parts.append(client_matrix)
    feature_matrix = np.concatenate(feature_parts, axis=0)
    del feature_parts
    feature_names = _feature_columns(
        feature_matrix.shape[1], application.preprocessing.feature_column_prefix
    )
    frame = pl.DataFrame(
        [
            pl.Series(
                PreprocessedColumn.CLIENT_ID, np.concatenate(client_id_parts), dtype=pl.String
            ),
            pl.Series(
                PreprocessedColumn.SOURCE_FILE, np.concatenate(source_file_parts), dtype=pl.String
            ),
            pl.Series(
                PreprocessedColumn.SOURCE_ROW, np.concatenate(source_row_parts), dtype=pl.Int64
            ),
            pl.Series(
                PreprocessedColumn.TIMESTAMP,
                np.full(feature_matrix.shape[0], None, dtype=object),
                dtype=pl.Float64,
            ),
            pl.Series(PreprocessedColumn.SPLIT, np.concatenate(split_parts), dtype=pl.String),
            pl.Series(PreprocessedColumn.ATTACK_SUBTYPE, subtype_values),
            *[
                pl.Series(name, feature_matrix[:, index])
                for index, name in enumerate(feature_names)
            ],
        ]
    )
    del feature_matrix
    manifest = FeatureManifest(dataset_id=config.dataset_id, columns=feature_names)
    return frame, tuple(split_plans), manifest


def resolve_preprocessing(
    paths: ArtifactPaths,
    dataset_id: DatasetId,
    raw_digest: ArtifactDigest,
    config_digest: ArtifactDigest,
    protocol_version: ProtocolVersion,
    overwrite: bool,
) -> PreprocessingStatus:
    provenance_path = paths.preprocessing_provenance_path(dataset_id)
    if overwrite:
        logger.info("[INVALIDATE] preprocessing for %s (explicit overwrite)", dataset_id.value)
        return PreprocessingStatus.REBUILD
    if not provenance_path.exists():
        return PreprocessingStatus.BUILD
    try:
        provenance = read_typed_json(provenance_path, TypeAdapter(PreprocessingProvenance))
    except (OSError, ValueError, ValidationError):
        logger.info("[INVALIDATE] preprocessing for %s (unreadable provenance)", dataset_id.value)
        return PreprocessingStatus.REBUILD
    if provenance.raw_digest != raw_digest:
        logger.info("[INVALIDATE] preprocessing for %s (raw data changed)", dataset_id.value)
        return PreprocessingStatus.REBUILD
    if provenance.preprocessing_config_digest != config_digest:
        logger.info("[INVALIDATE] preprocessing for %s (configuration changed)", dataset_id.value)
        return PreprocessingStatus.REBUILD
    if provenance.schema_version != protocol_version:
        logger.info(
            "[INVALIDATE] preprocessing for %s (protocol version changed)", dataset_id.value
        )
        return PreprocessingStatus.REBUILD
    if not paths.preprocessing_features_path(dataset_id).exists():
        return PreprocessingStatus.BUILD
    if not paths.preprocessing_split_manifest_path(dataset_id).exists():
        return PreprocessingStatus.BUILD
    return PreprocessingStatus.REUSE


def build_nbaiot_preprocessing(
    context: ApplicationContext,
    population: ClientPopulation,
    raw_digest: ArtifactDigest,
    config_digest: ArtifactDigest,
) -> PreparedFederation:
    application = context.config
    paths = context.paths
    config = application.protocol
    dataset_id = config.dataset_id
    frame, split_plans, manifest = _build_preprocessing_frame(context, population)
    split_manifest = build_split_manifest(dataset_id, population, split_plans)
    paths.preprocessing_dir(dataset_id).mkdir(parents=True, exist_ok=True)
    write_parquet_frame(paths.preprocessing_features_path(dataset_id), frame)
    write_typed_json(
        manifest, TypeAdapter(FeatureManifest), paths.preprocessing_manifest_path(dataset_id)
    )
    paths.preprocessing_manifest_digest_path(dataset_id).write_text(manifest.digest + "\n")
    write_typed_json(
        split_manifest,
        TypeAdapter(SplitManifest),
        paths.preprocessing_split_manifest_path(dataset_id),
    )
    features_digest = digest_file(paths.preprocessing_features_path(dataset_id))
    provenance = PreprocessingProvenance(
        dataset_id=dataset_id,
        schema_version=application.protocol.protocol_version,
        raw_digest=raw_digest,
        preprocessing_config_digest=config_digest,
        feature_manifest_digest=manifest.digest,
        split_manifest_digest=split_manifest.digest,
        features_digest=features_digest,
        built_at=now_iso(),
    )
    write_typed_json(
        provenance,
        TypeAdapter(PreprocessingProvenance),
        paths.preprocessing_provenance_path(dataset_id),
    )
    logger.info("[BUILD] preprocessed %s (%d rows)", dataset_id.value, frame.height)
    return PreparedFederation(
        population=population,
        feature_manifest=manifest,
        split_manifest=split_manifest,
        provenance=provenance,
        raw_digest=raw_digest,
        config_digest=config_digest,
    )


def _partition_frame(
    paths: ArtifactPaths,
    dataset_id: DatasetId,
    client_id: ClientId,
    split: BenignSplit | AttackSplit,
    feature_columns: tuple[ColumnName, ...],
    *,
    attack_subtype: AttackSubtypeId | None = None,
) -> pl.DataFrame:
    scan = (
        pl.scan_parquet(paths.preprocessing_features_path(dataset_id))
        .filter(pl.col(PreprocessedColumn.CLIENT_ID) == client_id)
        .filter(pl.col(PreprocessedColumn.SPLIT) == split.value)
    )
    if attack_subtype is not None:
        scan = scan.filter(pl.col(PreprocessedColumn.ATTACK_SUBTYPE) == attack_subtype)
    return scan.select(list(feature_columns)).collect()


def load_preprocessed_partition(
    paths: ArtifactPaths,
    dataset_id: DatasetId,
    client_id: ClientId,
    split: BenignSplit,
    feature_columns: tuple[ColumnName, ...],
) -> FeatureMatrix:
    rows = _partition_frame(paths, dataset_id, client_id, split, feature_columns)
    return FeatureMatrix(np.asarray(rows.to_numpy(), dtype=np.float64))


def load_preprocessed_attack_partition(
    paths: ArtifactPaths,
    dataset_id: DatasetId,
    client_id: ClientId,
    subtype: AttackSubtypeId,
    split: AttackSplit,
    feature_columns: tuple[ColumnName, ...],
) -> FeatureMatrix:
    rows = _partition_frame(
        paths,
        dataset_id,
        client_id,
        split,
        feature_columns,
        attack_subtype=subtype,
    )
    return FeatureMatrix(np.asarray(rows.to_numpy(), dtype=np.float64))


def prepare_federation(
    context: ApplicationContext,
    *,
    overwrite: bool = False,
) -> PreparedFederation:
    application = context.config
    paths = context.paths
    config = application.protocol
    dataset_id = config.dataset_id
    spec = build_dataset_registry(application).spec(dataset_id)
    if spec.population is None:
        raise DatasetError(f"no registered population for {dataset_id.value}")
    raw_digest = _raw_digest(context, dataset_id)
    config_digest = _config_digest(application)
    status = resolve_preprocessing(
        paths, dataset_id, raw_digest, config_digest, config.protocol_version, overwrite
    )
    if status is PreprocessingStatus.REUSE:
        provenance = read_typed_json(
            paths.preprocessing_provenance_path(dataset_id), TypeAdapter(PreprocessingProvenance)
        )
        manifest = read_typed_json(
            paths.preprocessing_manifest_path(dataset_id), TypeAdapter(FeatureManifest)
        )
        split_manifest = read_typed_json(
            paths.preprocessing_split_manifest_path(dataset_id), TypeAdapter(SplitManifest)
        )
        if (
            manifest.digest != provenance.feature_manifest_digest
            or split_manifest.digest != provenance.split_manifest_digest
        ):
            logger.info(
                "[INVALIDATE] preprocessing for %s (inconsistent provenance)", dataset_id.value
            )
            return build_nbaiot_preprocessing(context, spec.population, raw_digest, config_digest)
        logger.info("[REUSE] preprocessed %s", dataset_id.value)
        return PreparedFederation(
            population=spec.population,
            feature_manifest=manifest,
            split_manifest=split_manifest,
            provenance=provenance,
            raw_digest=raw_digest,
            config_digest=config_digest,
        )
    return build_nbaiot_preprocessing(context, spec.population, raw_digest, config_digest)


def _architecture(manifest: FeatureManifest, config: FabridConfig) -> AutoencoderArchitecture:
    return AutoencoderArchitecture(
        feature_count=len(manifest.columns), hidden_layers=config.detector.hidden_layers
    )


def _training_settings(config: FabridConfig) -> FederatedTrainingSettings:
    return FederatedTrainingSettings(
        learning_rate=config.detector.learning_rate,
        local_epochs=config.detector.local_epochs,
        rounds=config.detector.rounds,
        batch_size=config.detector.batch_size,
    )


def checkpoint_matches_config(
    saved: CheckpointMetadata,
    prepared: PreparedFederation,
    seed: DetectorSeed,
    architecture: AutoencoderArchitecture,
    training: FederatedTrainingSettings,
) -> bool:
    architecture_matches = (
        saved.dataset_id == prepared.provenance.dataset_id
        and saved.detector_seed == seed
        and saved.feature_count == len(prepared.feature_manifest.columns)
        and saved.hidden_layers == architecture.hidden_layers
        and saved.input_space is DetectorInputSpace.PER_CLIENT_ZSCORE
        and saved.training == training
    )
    return (
        architecture_matches and saved.preprocessing_digest == prepared.provenance.combined_digest
    )


def load_or_train_seed(
    prepared: PreparedFederation,
    paths: ArtifactPaths,
    config: FabridConfig,
    seed: DetectorSeed,
) -> TrainedDetectorSeed:
    coordinate = DetectorCoordinate(dataset_id=prepared.provenance.dataset_id, detector_seed=seed)
    metadata_path = paths.checkpoint_metadata_path(coordinate)
    if metadata_path.exists():
        try:
            saved = read_typed_json(metadata_path, TypeAdapter(CheckpointMetadata))
        except (ValidationError, ValueError):
            logger.info("[REBUILD] checkpoint seed=%d (invalid checkpoint metadata)", seed)
        else:
            if checkpoint_matches_config(
                saved,
                prepared,
                seed,
                _architecture(prepared.feature_manifest, config),
                _training_settings(config),
            ):
                model, scalers, metadata = load_detector_checkpoint(paths, coordinate)
                logger.info("[REUSE] checkpoint seed=%d", seed)
                return TrainedDetectorSeed(
                    coordinate=coordinate, model=model, scalers=scalers, metadata=metadata
                )
            logger.info(
                "[REBUILD] checkpoint seed=%d (protocol or preprocessing provenance mismatch)",
                seed,
            )
    unscaled_clients = tuple(
        ClientTrainingData(
            client_id=client_id,
            features=load_preprocessed_partition(
                paths,
                prepared.provenance.dataset_id,
                client_id,
                BenignSplit.TRAIN,
                prepared.feature_manifest.columns,
            ),
        )
        for client_id in prepared.population.clients
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
        _architecture(prepared.feature_manifest, config),
        _training_settings(config),
        seed,
        resolve_device(),
    )
    metadata = save_detector_checkpoint(
        paths,
        coordinate,
        model,
        scalers,
        _training_settings(config),
        prepared.provenance.combined_digest,
    )
    logger.info("[BUILD] checkpoint seed=%d", seed)
    return TrainedDetectorSeed(
        coordinate=coordinate, model=model, scalers=scalers, metadata=metadata
    )


def _seed_score_files_exist(
    paths: ArtifactPaths,
    dataset_id: DatasetId,
    seed: DetectorSeed,
    population: ClientPopulation,
) -> bool:
    for client_id in population.clients:
        coordinate = ScoreCoordinate(dataset_id=dataset_id, detector_seed=seed, client_id=client_id)
        for split in (
            BenignSplit.FRONTIER,
            BenignSplit.FINAL_CAL,
            BenignSplit.TEST,
            AttackSplit.VALIDATION,
            AttackSplit.TEST,
        ):
            if not paths.score_path(coordinate, split).exists():
                return False
    return True


def ensemble_scores(
    reconstruction_scores: np.ndarray,
    feature_values: np.ndarray,
    calibration_scores: np.ndarray,
    calibration_feature: np.ndarray,
) -> np.ndarray:
    def _percentiles(values: np.ndarray, pool: np.ndarray) -> np.ndarray:
        ordered = np.sort(pool)
        return np.searchsorted(ordered, values, side="right") / ordered.size

    return np.maximum(
        _percentiles(reconstruction_scores, calibration_scores),
        _percentiles(feature_values, calibration_feature),
    )


def _client_feature_values(
    paths: ArtifactPaths,
    dataset_id: DatasetId,
    client_id: ClientId,
    column: ColumnName,
) -> pl.DataFrame:
    frame = (
        pl.scan_parquet(paths.preprocessing_features_path(dataset_id))
        .filter(pl.col(PreprocessedColumn.CLIENT_ID) == client_id)
        .select(
            PreprocessedColumn.SPLIT,
            PreprocessedColumn.ATTACK_SUBTYPE,
            PreprocessedColumn.SOURCE_ROW,
            column,
        )
        .collect()
    )
    if column not in frame.columns:
        raise ValueError(f"ensemble score feature {column!r} missing from preprocessed features")
    return frame


def _split_feature_values(
    frame: pl.DataFrame, split: BenignSplit, column: ColumnName
) -> np.ndarray:
    return frame.filter(pl.col(PreprocessedColumn.SPLIT) == split.value)[column].to_numpy()


def _attack_feature_values(
    frame: pl.DataFrame,
    split: AttackSplit,
    subtype: AttackSubtypeId,
    column: ColumnName,
) -> np.ndarray:
    return frame.filter(
        (pl.col(PreprocessedColumn.SPLIT) == split.value)
        & (pl.col(PreprocessedColumn.ATTACK_SUBTYPE) == subtype)
    )[column].to_numpy()


def _materialize_seed_scores(
    prepared: PreparedFederation,
    trained: TrainedDetectorSeed,
    paths: ArtifactPaths,
    config: FabridConfig,
) -> None:
    dataset_id = prepared.provenance.dataset_id
    seed = trained.coordinate.detector_seed
    device = resolve_device()
    feature_columns = prepared.feature_manifest.columns
    total_clients = len(prepared.population.clients)
    for client_index, client_id in enumerate(prepared.population.clients, start=1):
        report_progress(
            logger,
            paths,
            f"scoring {dataset_id.value} seed {seed}",
            client_index,
            total_clients,
            detail=client_id,
        )
        coordinate = ScoreCoordinate(dataset_id=dataset_id, detector_seed=seed, client_id=client_id)
        scaler = trained.scalers.for_client(client_id)
        ensemble_feature = config.scoring.ensemble_score_feature
        frontier_features = load_preprocessed_partition(
            paths, dataset_id, client_id, BenignSplit.FRONTIER, feature_columns
        )
        final_cal_features = load_preprocessed_partition(
            paths, dataset_id, client_id, BenignSplit.FINAL_CAL, feature_columns
        )
        test_features = load_preprocessed_partition(
            paths, dataset_id, client_id, BenignSplit.TEST, feature_columns
        )
        frontier_scores = score_feature_matrix(
            trained.model,
            scaler.transform(frontier_features),
            device,
            config.detector.score_batch_size,
        )
        final_cal_scores = score_feature_matrix(
            trained.model,
            scaler.transform(final_cal_features),
            device,
            config.detector.score_batch_size,
        )
        test_scores = score_feature_matrix(
            trained.model, scaler.transform(test_features), device, config.detector.score_batch_size
        )
        frontier_feature: np.ndarray | None = None
        final_cal_feature: np.ndarray | None = None
        test_feature: np.ndarray | None = None
        calibration_scores: np.ndarray | None = None
        calibration_feature: np.ndarray | None = None
        feature_rows: pl.DataFrame | None = None
        if ensemble_feature is not None:
            feature_rows = _client_feature_values(paths, dataset_id, client_id, ensemble_feature)
            frontier_feature = _split_feature_values(
                feature_rows, BenignSplit.FRONTIER, ensemble_feature
            )
            final_cal_feature = _split_feature_values(
                feature_rows, BenignSplit.FINAL_CAL, ensemble_feature
            )
            test_feature = _split_feature_values(feature_rows, BenignSplit.TEST, ensemble_feature)
            calibration_scores = np.concatenate([frontier_scores, final_cal_scores])
            calibration_feature = np.concatenate([frontier_feature, final_cal_feature])
        benign_partitions: tuple[tuple[BenignSplit, np.ndarray, np.ndarray | None], ...] = (
            (BenignSplit.FRONTIER, frontier_scores, frontier_feature),
            (BenignSplit.FINAL_CAL, final_cal_scores, final_cal_feature),
            (BenignSplit.TEST, test_scores, test_feature),
        )
        for split, scores, feature in benign_partitions:
            if (
                calibration_scores is not None
                and calibration_feature is not None
                and feature is not None
            ):
                scores = ensemble_scores(scores, feature, calibration_scores, calibration_feature)
            persist_score_partition(
                paths.score_path(coordinate, split),
                build_score_partition(
                    coordinate, split, scores, Label.BENIGN, PREPROCESSED_SOURCE_FILE
                ),
            )
        for split in (AttackSplit.VALIDATION, AttackSplit.TEST):
            partition_records: list[ScoreRecord] = []
            for subtype in _attack_subtypes(paths, prepared, client_id):
                features = load_preprocessed_attack_partition(
                    paths, dataset_id, client_id, subtype, split, feature_columns
                )
                scores = score_feature_matrix(
                    trained.model,
                    scaler.transform(features),
                    device,
                    config.detector.score_batch_size,
                )
                if (
                    calibration_scores is not None
                    and calibration_feature is not None
                    and feature_rows is not None
                    and ensemble_feature is not None
                ):
                    scores = ensemble_scores(
                        scores,
                        _attack_feature_values(feature_rows, split, subtype, ensemble_feature),
                        calibration_scores,
                        calibration_feature,
                    )
                partition_records.extend(
                    build_score_partition(
                        coordinate,
                        split,
                        scores,
                        Label.ATTACK,
                        PREPROCESSED_SOURCE_FILE,
                        subtype,
                    ).records
                )
            persist_score_partition(
                paths.score_path(coordinate, split),
                ScorePartitionArtifact(
                    coordinate=coordinate,
                    split=split,
                    records=tuple(partition_records),
                ),
            )


def load_or_materialize_scores(
    prepared: PreparedFederation,
    trained: TrainedDetectorSeed,
    paths: ArtifactPaths,
    config: FabridConfig,
) -> StoredSeedScores:
    dataset_id = prepared.provenance.dataset_id
    seed = trained.coordinate.detector_seed
    population = prepared.population
    expected_provenance = ScoreProvenance(
        checkpoint_digest=trained.metadata.combined_digest,
        preprocessing_digest=prepared.provenance.combined_digest,
        scoring_config_digest=_scoring_config_digest(config),
    )
    if _seed_score_files_exist(paths, dataset_id, seed, population) and scores_reuse_valid(
        paths, dataset_id, seed, expected_provenance
    ):
        logger.info("[REUSE] frozen scores seed=%d", seed)
    else:
        if _seed_score_files_exist(paths, dataset_id, seed, population):
            logger.info("[REBUILD] frozen scores seed=%d (score provenance mismatch)", seed)
        logger.info("[GENERATE] frozen scores seed=%d", seed)
        _materialize_seed_scores(prepared, trained, paths, config)
        write_typed_json(
            expected_provenance,
            _SCORE_PROVENANCE_ADAPTER,
            paths.score_provenance_path(dataset_id, seed),
        )
        logger.info("[SAVE] frozen scores seed=%d", seed)
    loaded = load_seed_scores(paths, dataset_id, seed, population)
    return StoredSeedScores(
        seed=seed,
        loaded=loaded,
        score_sha256=loaded.digest(),
    )


def scores_reuse_valid(
    paths: ArtifactPaths,
    dataset_id: DatasetId,
    seed: DetectorSeed,
    expected: ScoreProvenance,
) -> bool:
    provenance_path = paths.score_provenance_path(dataset_id, seed)
    if not provenance_path.exists():
        return False
    try:
        stored = read_typed_json(provenance_path, TypeAdapter(ScoreProvenance))
    except (OSError, ValueError, ValidationError):
        return False
    return (
        stored.checkpoint_digest == expected.checkpoint_digest
        and stored.preprocessing_digest == expected.preprocessing_digest
        and stored.scoring_config_digest == expected.scoring_config_digest
    )


def _attack_subtypes(
    paths: ArtifactPaths,
    prepared: PreparedFederation,
    client_id: ClientId,
) -> tuple[AttackSubtypeId, ...]:
    subtypes = (
        pl.scan_parquet(paths.preprocessing_features_path(prepared.provenance.dataset_id))
        .filter(pl.col(PreprocessedColumn.CLIENT_ID) == client_id)
        .select(PreprocessedColumn.ATTACK_SUBTYPE)
        .drop_nulls()
        .unique()
        .collect()
        .to_series()
        .to_list()
    )
    return tuple(sorted(str(subtype) for subtype in subtypes))
