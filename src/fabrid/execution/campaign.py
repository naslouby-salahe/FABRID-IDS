from __future__ import annotations

import logging
import sys
import time
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import cast

import polars as pl
from pydantic import TypeAdapter

from fabrid.allocation.problem import ClientRowCount, build_allocation_problem, equal_client_weights
from fabrid.artifacts.json import protocol_config_digest, write_typed_json
from fabrid.artifacts.parquet import write_parquet_models
from fabrid.artifacts.paths import ArtifactPaths, ExperimentCoordinate
from fabrid.config import (
    AnalysisArtifactId,
    BenignSplit,
    CampaignAnalysisId,
    DatasetId,
    DetectorSeed,
    EnvironmentText,
    ExperimentId,
    ExperimentVariantId,
    FabridConfig,
    PreprocessedColumn,
    WeightMode,
    WorkerCount,
)
from fabrid.datasets.registry import ClientPopulation
from fabrid.errors import ExecutionError
from fabrid.evaluation.gates import PracticalGateAnalysis, analyze_practical_gates
from fabrid.evaluation.inference import PrimaryInference, analyze_primary_inference
from fabrid.evaluation.metrics import ClientResultRecord, SeedBudgetEvaluation
from fabrid.execution.application import ApplicationContext
from fabrid.execution.prerequisites import (
    PreparedFederation,
    StoredSeedScores,
    TrainedDetectorSeed,
    load_or_materialize_scores,
    load_or_train_seed,
    prepare_federation,
)
from fabrid.execution.resources import (
    configure_campaign_logging,
    peak_gpu_mb,
    peak_rss_mb,
    report_resources,
)
from fabrid.execution.status import campaign_evidence_currency
from fabrid.experiments.generalization import (
    run_attack_subtype_generalization_seed,
    run_botnet_family_generalization_seed,
)
from fabrid.experiments.matched_budget import (
    EvaluationProvenance,
    LoadedSeedScores,
    PayloadMeasurement,
    build_frontier_inputs,
    measure_client_payload,
    persist_seed_budget,
    run_seed_budget,
)
from fabrid.experiments.sensitivity import (
    measure_seed_utility_heterogeneity,
    run_allocation_stability_seed_budget,
    run_conservative_utility_seed_budget,
    run_weight_sensitivity_seed,
)
from fabrid.reporting.build import (
    build_campaign_report,
    load_evaluations,
    package_results,
)
from fabrid.validation.completion import (
    CellState,
    ExperimentState,
    experiment_contract,
    verify_experiment,
)
from fabrid.validation.datasets import DatasetChecksum
from fabrid.validation.preflight import require_static_preflight
from fabrid.validation.reproducibility import (
    ReproducibilityMetadata,
    capture_reproducibility_metadata,
    now_iso,
    resolve_git_commit,
)


@dataclass(frozen=True, slots=True)
class FabridCampaign:
    matched_budget: tuple[SeedBudgetEvaluation, ...]
    records: tuple[ClientResultRecord, ...]
    primary_inference: PrimaryInference | None
    practical_gates: PracticalGateAnalysis | None
    report_path: Path | None
    results_bundle_path: Path | None


def evaluation_provenance(
    prepared: PreparedFederation,
    stored: StoredSeedScores,
    trained: TrainedDetectorSeed,
    protocol: FabridConfig,
) -> EvaluationProvenance:
    return EvaluationProvenance(
        model_sha256=trained.metadata.model_digest,
        score_sha256=stored.score_sha256,
        split_sha256=prepared.split_manifest.digest,
        feature_sha256=prepared.feature_manifest.digest,
        protocol_sha256=protocol_config_digest(protocol),
        git_commit=resolve_git_commit(),
    )


def _benign_row_counts(
    paths: ArtifactPaths,
    dataset_id: DatasetId,
    population: ClientPopulation,
) -> tuple[ClientRowCount, ...]:
    benign_splits = tuple(split.value for split in BenignSplit)
    rows = (
        pl.scan_parquet(paths.preprocessing_features_path(dataset_id))
        .filter(pl.col(PreprocessedColumn.SPLIT).is_in(benign_splits))
        .group_by(PreprocessedColumn.CLIENT_ID)
        .len()
        .collect()
    )
    counted = tuple(
        ClientRowCount(client_id=row[PreprocessedColumn.CLIENT_ID], row_count=int(row["len"]))
        for row in rows.iter_rows(named=True)
    )
    return tuple(
        next(item for item in counted if item.client_id == client_id)
        for client_id in population.clients
    )


def _measure_seed_payload(
    prepared: PreparedFederation,
    loaded: LoadedSeedScores,
    config: FabridConfig,
) -> tuple[PayloadMeasurement, ...]:
    inputs = build_frontier_inputs(loaded, config.alpha_grid)
    problem = build_allocation_problem(
        inputs,
        equal_client_weights(prepared.population),
        config.budgets[0].value,
        config.utility_eligibility,
        config.maximum_target_rate,
    )
    curves = problem.frontier.eligible_curves()
    if curves is None:
        return ()
    return tuple(measure_client_payload(curve, config.payload_sizing) for curve in curves.clients)


def _run_matched_budget_seed(
    context: ApplicationContext,
    stored: StoredSeedScores,
    provenance: EvaluationProvenance,
    seed: DetectorSeed,
    dataset_id: DatasetId,
    logger: logging.Logger,
) -> tuple[tuple[SeedBudgetEvaluation, ...], tuple[ClientResultRecord, ...]]:
    config = context.config.protocol
    evaluations: list[SeedBudgetEvaluation] = []
    records: list[ClientResultRecord] = []
    for budget_level in config.budgets:
        logger.info(
            "[RUN] matched-budget / %s %s",
            budget_level.budget_id.value,
            budget_level.value,
        )
        run = run_seed_budget(
            ExperimentCoordinate(
                experiment_id=ExperimentId.MATCHED_BUDGET,
                variant_id=ExperimentVariantId.PRIMARY,
                dataset_id=dataset_id,
                detector_seed=seed,
                budget_id=budget_level.budget_id,
                budget=budget_level.value,
                weight_mode=WeightMode.EQUAL_CLIENT,
            ),
            stored.loaded,
            config,
            provenance,
            equal_client_weights(stored.loaded.population),
        )
        persist_seed_budget(context.paths, run)
        evaluations.append(run.evaluation)
        records.extend(run.records)
    return tuple(evaluations), tuple(records)


def _run_attack_subtype_if_enabled(
    context: ApplicationContext,
    stored: StoredSeedScores,
    provenance: EvaluationProvenance,
    seed: DetectorSeed,
    dataset_id: DatasetId,
    logger: logging.Logger,
) -> None:
    if not context.config.experiments.attack_subtype_disjoint:
        return
    for experiment_run in run_attack_subtype_generalization_seed(
        seed,
        stored.loaded,
        context.config.protocol,
        provenance,
        dataset_id,
    ):
        persist_seed_budget(context.paths, experiment_run)
    logger.info("[RUN] attack-subtype-disjoint generalization seed=%d", seed)


def _run_botnet_family_if_enabled(
    context: ApplicationContext,
    stored: StoredSeedScores,
    provenance: EvaluationProvenance,
    seed: DetectorSeed,
    dataset_id: DatasetId,
    logger: logging.Logger,
) -> None:
    if not context.config.experiments.botnet_family_disjoint:
        return
    for experiment_run in run_botnet_family_generalization_seed(
        seed,
        stored.loaded,
        context.config.protocol,
        provenance,
        dataset_id,
        ClientPopulation(context.config.datasets.nbaiot.dual_botnet_devices()),
    ):
        persist_seed_budget(context.paths, experiment_run)
    logger.info("[RUN] botnet-family-disjoint transfer seed=%d", seed)


def _run_stability_and_conservative_if_enabled(
    context: ApplicationContext,
    stored: StoredSeedScores,
    provenance: EvaluationProvenance,
    seed: DetectorSeed,
    dataset_id: DatasetId,
    *,
    stability: bool,
    conservative: bool,
    logger: logging.Logger,
) -> None:
    if not (stability or conservative):
        return
    config = context.config.protocol
    for budget_level in config.budgets:
        if stability:
            run_allocation_stability_seed_budget(
                seed,
                budget_level,
                stored.loaded,
                config,
                context.paths,
                dataset_id,
                worker_limit=1,
            )
        if conservative:
            for experiment_run in run_conservative_utility_seed_budget(
                seed,
                budget_level,
                stored.loaded,
                config,
                provenance,
                context.paths,
                dataset_id,
            ):
                persist_seed_budget(context.paths, experiment_run)
    logger.info("[RUN] allocation stability and conservative utility seed=%d", seed)


def _run_weight_sensitivity_if_enabled(
    context: ApplicationContext,
    prepared: PreparedFederation,
    stored: StoredSeedScores,
    provenance: EvaluationProvenance,
    seed: DetectorSeed,
    dataset_id: DatasetId,
    logger: logging.Logger,
) -> None:
    if not context.config.experiments.weight_sensitivity:
        return
    for experiment_run in run_weight_sensitivity_seed(
        seed,
        stored.loaded,
        context.config.protocol,
        provenance,
        context.paths,
        dataset_id,
        _benign_row_counts(context.paths, dataset_id, prepared.population),
    ):
        persist_seed_budget(context.paths, experiment_run)
    logger.info("[RUN] weight sensitivity seed=%d", seed)


def _persist_heterogeneity_if_enabled(
    context: ApplicationContext,
    stored: StoredSeedScores,
    seed: DetectorSeed,
    dataset_id: DatasetId,
    logger: logging.Logger,
) -> None:
    if not context.config.experiments.utility_heterogeneity:
        return
    config = context.config.protocol
    heterogeneity_rows = measure_seed_utility_heterogeneity(seed, stored.loaded, config)
    if not heterogeneity_rows:
        return
    coordinate = ExperimentCoordinate(
        experiment_id=ExperimentId.UTILITY_HETEROGENEITY,
        variant_id=ExperimentVariantId.PRIMARY,
        dataset_id=dataset_id,
        detector_seed=seed,
        budget_id=config.budgets[0].budget_id,
        budget=config.budgets[0].value,
        weight_mode=WeightMode.EQUAL_CLIENT,
    )
    write_parquet_models(
        context.paths.analysis_path(coordinate, AnalysisArtifactId.UTILITY_HETEROGENEITY),
        heterogeneity_rows,
    )
    logger.info(
        "[SAVE] utility heterogeneity seed=%d (%d candidates)",
        seed,
        len(heterogeneity_rows),
    )


def _run_enabled_seed_experiments(
    context: ApplicationContext,
    prepared: PreparedFederation,
    stored: StoredSeedScores,
    provenance: EvaluationProvenance,
    seed: DetectorSeed,
    pending: frozenset[ExperimentId],
    logger: logging.Logger,
) -> tuple[tuple[SeedBudgetEvaluation, ...], tuple[ClientResultRecord, ...]]:
    dataset_id = prepared.provenance.dataset_id
    evaluations: list[SeedBudgetEvaluation] = []
    records: list[ClientResultRecord] = []
    if ExperimentId.MATCHED_BUDGET in pending:
        seed_evaluations, seed_records = _run_matched_budget_seed(
            context, stored, provenance, seed, dataset_id, logger
        )
        evaluations.extend(seed_evaluations)
        records.extend(seed_records)
    if ExperimentId.ATTACK_SUBTYPE_DISJOINT in pending:
        _run_attack_subtype_if_enabled(context, stored, provenance, seed, dataset_id, logger)
    if ExperimentId.BOTNET_FAMILY_DISJOINT in pending:
        _run_botnet_family_if_enabled(context, stored, provenance, seed, dataset_id, logger)
    if ExperimentId.ALLOCATION_STABILITY in pending or ExperimentId.CONSERVATIVE_UTILITY in pending:
        _run_stability_and_conservative_if_enabled(
            context,
            stored,
            provenance,
            seed,
            dataset_id,
            stability=ExperimentId.ALLOCATION_STABILITY in pending,
            conservative=ExperimentId.CONSERVATIVE_UTILITY in pending,
            logger=logger,
        )
    if ExperimentId.WEIGHT_SENSITIVITY in pending:
        _run_weight_sensitivity_if_enabled(
            context, prepared, stored, provenance, seed, dataset_id, logger
        )
    if ExperimentId.UTILITY_HETEROGENEITY in pending:
        _persist_heterogeneity_if_enabled(context, stored, seed, dataset_id, logger)
    return tuple(evaluations), tuple(records)


def _prepare_seed_worker(
    context: ApplicationContext,
    prepared: PreparedFederation,
    seed: DetectorSeed,
) -> DetectorSeed:
    logger = configure_campaign_logging(context.paths)
    logger.info("[PREPARE] seed %d/%d", seed, len(context.config.protocol.seeds))
    trained = load_or_train_seed(prepared, context.paths, context.config.protocol, seed)
    load_or_materialize_scores(prepared, trained, context.paths, context.config.protocol)
    logger.info("[READY] seed %d", seed)
    return seed


def _run_seed_evaluation_worker(
    context: ApplicationContext,
    prepared: PreparedFederation,
    pending: frozenset[ExperimentId],
    seed: DetectorSeed,
) -> tuple[tuple[SeedBudgetEvaluation, ...], tuple[ClientResultRecord, ...]]:
    logger = configure_campaign_logging(context.paths)
    logger.info("[EVALUATE] seed %d/%d", seed, len(context.config.protocol.seeds))
    trained = load_or_train_seed(prepared, context.paths, context.config.protocol, seed)
    stored = load_or_materialize_scores(prepared, trained, context.paths, context.config.protocol)
    provenance = evaluation_provenance(prepared, stored, trained, context.config.protocol)
    evaluations, records = _run_enabled_seed_experiments(
        context, prepared, stored, provenance, seed, pending, logger
    )
    _persist_seed_payload(context, prepared, stored, seed, logger)
    return evaluations, records


def _seed_evidence_is_complete(
    context: ApplicationContext,
    pending: frozenset[ExperimentId],
    seed: DetectorSeed,
) -> bool:
    for experiment_id in pending:
        contract = experiment_contract(context.config, experiment_id)
        cells = tuple(
            cell
            for cell in verify_experiment(context.config, context.paths, experiment_id).cells
            if cell.coordinate.detector_seed == seed
        )
        if any(cell.state is not CellState.VALID for cell in cells):
            return False
        if not cells and contract.cells:
            return False
    return True


def primary_seed_experiments(
    context: ApplicationContext,
    pending: frozenset[ExperimentId],
) -> frozenset[ExperimentId]:
    primary_dataset_id = context.config.protocol.dataset_id
    return frozenset(
        experiment_id
        for experiment_id in pending
        if any(
            cell.coordinate.dataset_id == primary_dataset_id
            for cell in experiment_contract(context.config, experiment_id).cells
        )
    )


def _persist_seed_payload(
    context: ApplicationContext,
    prepared: PreparedFederation,
    stored: StoredSeedScores,
    seed: DetectorSeed,
    logger: logging.Logger,
) -> None:
    config = context.config.protocol
    payload_rows = _measure_seed_payload(prepared, stored.loaded, config)
    if not payload_rows:
        return
    dataset_id = prepared.provenance.dataset_id
    coordinate = context.paths.matched_budget_coordinate(dataset_id, seed, config.budgets[0])
    write_parquet_models(
        context.paths.analysis_path(coordinate, AnalysisArtifactId.PAYLOAD_MEASUREMENT),
        payload_rows,
    )
    logger.info("[SAVE] payload measurement seed=%d (%d clients)", seed, len(payload_rows))


def _run_verified_external_replication(
    context: ApplicationContext,
    logger: logging.Logger,
    workers: WorkerCount = 1,
) -> None:
    if not context.config.experiments.external_replication:
        return
    verification = verify_experiment(
        context.config, context.paths, ExperimentId.EXTERNAL_REPLICATION
    )
    if verification.state is ExperimentState.PASSED:
        logger.info("[SKIP] external replication already passed (verified evidence)")
        return
    from fabrid.experiments.external_replication import run_external_campaign

    external = context.config.external_replication
    logger.info(
        "[RUN] external replication starting: %d seeds x %d budgets = %d cells (%d devices, %s)",
        len(external.detector.seeds),
        len(external.budgets),
        len(external.detector.seeds) * len(external.budgets),
        verification.expected_cells // max(len(external.budgets), 1),
        external.dataset_id.value,
    )
    started = time.monotonic()
    run_external_campaign(context.config, context.paths, workers=workers)
    logger.info("[RUN] external replication completed in %.0f s", time.monotonic() - started)


def _run_verified_event_level(context: ApplicationContext, logger: logging.Logger) -> None:
    if not context.config.experiments.event_level:
        return
    verification = verify_experiment(context.config, context.paths, ExperimentId.EVENT_LEVEL)
    if verification.state is ExperimentState.PASSED:
        logger.info("[SKIP] event-level already passed (verified evidence)")
        return
    from fabrid.experiments.event_level import prepare_and_run_event_level

    event = context.config.event_level
    logger.info(
        "[RUN] event-level starting: %d seeds, %s, %d budgets, %d event-budget rates",
        len(event.detector.seeds),
        event.dataset_id.value,
        len(event.budgets),
        len(event.event_gate.budgets_per_client_hour),
    )
    started = time.monotonic()
    prepare_and_run_event_level(context)
    logger.info("[RUN] event-level completed in %.0f s", time.monotonic() - started)


def _require_campaign_completion(context: ApplicationContext, logger: logging.Logger) -> None:
    ok, issues = campaign_evidence_currency(context.config, context.paths)
    if ok:
        logger.info("[VERIFY] campaign evidence and results bundle passed")
        return
    raise ExecutionError(f"campaign completion verification failed: {' '.join(issues)}")


def _finalize_campaign(
    context: ApplicationContext,
    prepared: PreparedFederation,
    evaluations: tuple[SeedBudgetEvaluation, ...],
    records: tuple[ClientResultRecord, ...],
    dataset_id: DatasetId,
    started_at: EnvironmentText,
    logger: logging.Logger,
    external_workers: WorkerCount = 1,
) -> FabridCampaign:
    paths = context.paths
    config = context.config.protocol
    _run_verified_external_replication(context, logger, external_workers)
    _run_verified_event_level(context, logger)
    primary: PrimaryInference | None = None
    gates: PracticalGateAnalysis | None = None
    if evaluations:
        primary = analyze_primary_inference(evaluations, config.budgets, config.statistics)
        gates = analyze_practical_gates(
            evaluations,
            config.budgets,
            config.statistics,
            config.practical_gates,
            expected_seed_count=len(config.seeds),
        )
        write_typed_json(
            primary,
            TypeAdapter(PrimaryInference),
            paths.campaign_analysis_path(CampaignAnalysisId.PRIMARY_INFERENCE),
        )
        write_typed_json(
            gates,
            TypeAdapter(PracticalGateAnalysis),
            paths.campaign_analysis_path(CampaignAnalysisId.PRACTICAL_GATES),
        )
        logger.info("[SAVE] primary inference and practical gates")
    else:
        logger.info("[SKIP] primary inference and practical gates; no matched-budget evaluations")
    logger.info(
        "[RESOURCES] campaign peak_rss_mb=%d peak_gpu_mb=%d",
        peak_rss_mb(),
        peak_gpu_mb(),
    )
    evidence, report_path = build_campaign_report(paths, config)
    reproducibility = capture_reproducibility_metadata(
        config.seeds,
        (DatasetChecksum(dataset_id=dataset_id, digest=prepared.raw_digest),),
        command_line=" ".join(sys.argv),
        started_at=started_at,
        finished_at=now_iso(),
    )
    write_typed_json(
        reproducibility,
        TypeAdapter(ReproducibilityMetadata),
        paths.campaign_reproducibility_path(),
    )
    bundle = package_results(config, paths, reproducibility, evidence)
    logger.info("[SAVE] results bundle %s", bundle)
    _require_campaign_completion(context, logger)
    report_resources(logger, "end")
    return FabridCampaign(
        matched_budget=evaluations,
        records=records,
        primary_inference=primary,
        practical_gates=gates,
        report_path=report_path,
        results_bundle_path=bundle,
    )


def run_campaign_finalize(context: ApplicationContext) -> FabridCampaign:
    application = context.config
    paths = context.paths
    config = application.protocol
    require_static_preflight(application, repository_root=context.repository_root)
    logger = configure_campaign_logging(paths)
    report_resources(logger, "start")
    started_at = now_iso()
    write_typed_json(config, TypeAdapter(FabridConfig), paths.campaign_protocol_path())
    logger.info("[SAVE] protocol snapshot %s", config.protocol_version)
    prepared = prepare_federation(context, overwrite=False)
    dataset_id = prepared.provenance.dataset_id
    evaluations = load_evaluations(paths, config)
    for seed in config.seeds:
        load_or_train_seed(prepared, paths, config, seed)
    return _finalize_campaign(context, prepared, evaluations, (), dataset_id, started_at, logger)


def pending_experiments(context: ApplicationContext, *, overwrite: bool) -> frozenset[ExperimentId]:
    if overwrite:
        return frozenset(
            experiment_id
            for experiment_id in ExperimentId
            if context.config.experiments.enabled(experiment_id)
        )
    return frozenset(
        experiment_id
        for experiment_id in ExperimentId
        if context.config.experiments.enabled(experiment_id)
        and verify_experiment(context.config, context.paths, experiment_id).state
        is not ExperimentState.PASSED
    )


def run_fabrid_campaign(
    context: ApplicationContext,
    *,
    overwrite: bool = False,
    seed_workers: WorkerCount | None = None,
    evaluation_workers: WorkerCount | None = None,
) -> FabridCampaign:
    application = context.config
    selected_seed_workers = (
        application.execution.seed_workers if seed_workers is None else seed_workers
    )
    selected_evaluation_workers = (
        application.execution.evaluation_workers
        if evaluation_workers is None
        else evaluation_workers
    )
    if selected_seed_workers < 1:
        raise ValueError("seed_workers must be at least one")
    if selected_evaluation_workers < 1:
        raise ValueError("evaluation_workers must be at least one")
    paths = context.paths
    config = application.protocol
    require_static_preflight(application, repository_root=context.repository_root)
    logger = configure_campaign_logging(paths)
    report_resources(logger, "start")
    started_at = now_iso()
    write_typed_json(config, TypeAdapter(FabridConfig), paths.campaign_protocol_path())
    logger.info("[SAVE] protocol snapshot %s", config.protocol_version)
    prepared = prepare_federation(context, overwrite=overwrite)
    dataset_id = prepared.provenance.dataset_id
    pending = pending_experiments(context, overwrite=overwrite)
    for experiment_id in ExperimentId:
        if context.config.experiments.enabled(experiment_id) and experiment_id not in pending:
            logger.info("[SKIP] %s already passed (verified evidence)", experiment_id.value)
    evaluations: list[SeedBudgetEvaluation] = []
    records: list[ClientResultRecord] = []
    primary_seed_pending = primary_seed_experiments(context, pending)
    seeds_to_run = tuple(
        seed
        for seed in config.seeds
        if not _seed_evidence_is_complete(context, primary_seed_pending, seed)
    )
    for seed in config.seeds:
        if seed not in seeds_to_run:
            logger.info("[SKIP] seed %d already has verified selected evidence", seed)
    workers = min(selected_seed_workers, len(seeds_to_run))
    cpu_workers = min(selected_evaluation_workers, len(seeds_to_run))
    logger.info("[RUN] preparation workers=%d evaluation workers=%d", workers, cpu_workers)
    if workers == 0:
        pass
    elif workers == 1:
        for seed in seeds_to_run:
            _prepare_seed_worker(context, prepared, seed)
            seed_evaluations, seed_records = _run_seed_evaluation_worker(
                context, prepared, pending, seed
            )
            evaluations.extend(seed_evaluations)
            records.extend(seed_records)
    else:
        with (
            ProcessPoolExecutor(
                max_workers=workers,
                mp_context=get_context("spawn"),
            ) as preparation_executor,
            ProcessPoolExecutor(
                max_workers=cpu_workers,
                mp_context=get_context("spawn"),
            ) as evaluation_executor,
        ):
            preparation_futures = {
                preparation_executor.submit(_prepare_seed_worker, context, prepared, seed): seed
                for seed in seeds_to_run
            }
            evaluation_futures = cast(
                dict[
                    DetectorSeed,
                    Future[
                        tuple[
                            tuple[SeedBudgetEvaluation, ...],
                            tuple[ClientResultRecord, ...],
                        ]
                    ],
                ],
                {},
            )
            for preparation_future in as_completed(preparation_futures):
                seed = preparation_future.result()
                evaluation_futures[seed] = evaluation_executor.submit(
                    _run_seed_evaluation_worker, context, prepared, pending, seed
                )
            for seed in seeds_to_run:
                seed_evaluations, seed_records = evaluation_futures[seed].result()
                evaluations.extend(seed_evaluations)
                records.extend(seed_records)

    verified_evaluations = load_evaluations(paths, config)
    return _finalize_campaign(
        context,
        prepared,
        verified_evaluations,
        tuple(records),
        dataset_id,
        started_at,
        logger,
        external_workers=selected_seed_workers,
    )
