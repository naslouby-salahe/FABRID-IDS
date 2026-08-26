from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from fabrid.allocation.optimization import AllocationSnapshot
from fabrid.artifacts.json import protocol_config_digest, read_typed_json
from fabrid.artifacts.paths import (
    AllocationCoordinate,
    ArtifactFileName,
    ArtifactPaths,
    DetectorCoordinate,
    ExperimentCoordinate,
)
from fabrid.config import (
    AllocationPolicy,
    AnalysisArtifactId,
    ApplicationConfig,
    ArtifactCount,
    ArtifactDigest,
    BudgetLevel,
    DatasetId,
    DetectorSeed,
    EnvironmentText,
    ExperimentId,
    ExperimentVariantId,
    FabridConfig,
    WeightMode,
)
from fabrid.detector.training import CheckpointMetadata
from fabrid.evaluation.metrics import ClientResultRecord, SeedBudgetEvaluation
from fabrid.validation.datasets import PreprocessingProvenance

logger = logging.getLogger(__name__)
_CONTRACT_BUDGET_POLICIES = (AllocationPolicy.FABRID_MACRO, AllocationPolicy.FABRID_CVAR)


class ExperimentState(StrEnum):
    NOT_STARTED = "not_started"
    EXECUTION_INCOMPLETE = "execution_incomplete"
    EVIDENCE_INCOMPLETE = "evidence_incomplete"
    STALE = "stale"
    FAILED = "failed"
    PASSED = "passed"


class CellState(StrEnum):
    MISSING = "missing"
    VALID = "valid"
    MALFORMED = "malformed"
    INCOMPLETE = "incomplete"
    STALE = "stale"


class ProvenanceKind(StrEnum):
    PRIMARY = "primary"
    EXTERNAL = "external"
    NONE = "none"


@dataclass(frozen=True, slots=True)
class CellContract:
    coordinate: ExperimentCoordinate
    requires_evaluation: bool
    requires_results: bool
    requires_allocations: tuple[AllocationPolicy, ...]
    requires_analysis: tuple[AnalysisArtifactId, ...]
    provenance: ProvenanceKind


@dataclass(frozen=True, slots=True)
class AggregateArtifact:
    path: Path
    artifact_name: EnvironmentText


@dataclass(frozen=True, slots=True)
class ExperimentContract:
    experiment_id: ExperimentId
    cells: tuple[CellContract, ...]
    aggregates: tuple[AggregateArtifact, ...]


@dataclass(frozen=True, slots=True)
class PrimarySnapshot:
    protocol_digest: ArtifactDigest
    feature_digest: ArtifactDigest
    split_digest: ArtifactDigest
    model_digests: tuple[tuple[DetectorSeed, ArtifactDigest], ...]
    stored_protocol_digest: ArtifactDigest | None


@dataclass(frozen=True, slots=True)
class ExternalSnapshot:
    protocol_digest: ArtifactDigest
    model_digests: tuple[tuple[DetectorSeed, ArtifactDigest], ...]


@dataclass(frozen=True, slots=True)
class CellVerification:
    coordinate: ExperimentCoordinate
    state: CellState
    issues: tuple[EnvironmentText, ...]


@dataclass(frozen=True, slots=True)
class ExperimentVerification:
    experiment_id: ExperimentId
    state: ExperimentState
    expected_cells: ArtifactCount
    valid_cells: ArtifactCount
    cells: tuple[CellVerification, ...]
    issues: tuple[EnvironmentText, ...]


def _evaluation_cells(
    experiment_id: ExperimentId,
    seeds: tuple[DetectorSeed, ...],
    budgets: tuple[BudgetLevel, ...],
    variant_id: ExperimentVariantId,
    dataset_id: DatasetId,
    provenance: ProvenanceKind,
) -> tuple[CellContract, ...]:
    return tuple(
        CellContract(
            coordinate=ExperimentCoordinate(
                experiment_id=experiment_id,
                variant_id=variant_id,
                dataset_id=dataset_id,
                detector_seed=seed,
                budget_id=budget.budget_id,
                budget=budget.value,
                weight_mode=WeightMode.EQUAL_CLIENT,
            ),
            requires_evaluation=True,
            requires_results=True,
            requires_allocations=_CONTRACT_BUDGET_POLICIES,
            requires_analysis=(),
            provenance=provenance,
        )
        for seed in seeds
        for budget in budgets
    )


def experiment_contract(
    application: ApplicationConfig,
    experiment_id: ExperimentId,
) -> ExperimentContract:
    protocol = application.protocol
    dataset_id = protocol.dataset_id
    seeds = protocol.seeds
    budgets = protocol.budgets
    paths = ArtifactPaths.from_settings(application.paths)
    if experiment_id is ExperimentId.MATCHED_BUDGET:
        cells = _evaluation_cells(
            experiment_id,
            seeds,
            budgets,
            ExperimentVariantId.PRIMARY,
            dataset_id,
            ProvenanceKind.PRIMARY,
        )
        return ExperimentContract(experiment_id, cells, ())
    if experiment_id is ExperimentId.ATTACK_SUBTYPE_DISJOINT:
        cells = tuple(
            cell
            for rotation in protocol.generalization.rotations
            for cell in _evaluation_cells(
                experiment_id,
                seeds,
                budgets,
                rotation.variant_id,
                dataset_id,
                ProvenanceKind.PRIMARY,
            )
        )
        return ExperimentContract(experiment_id, cells, ())
    if experiment_id is ExperimentId.BOTNET_FAMILY_DISJOINT:
        cells = tuple(
            cell
            for direction in protocol.generalization.family_directions
            for cell in _evaluation_cells(
                experiment_id,
                seeds,
                budgets,
                direction.variant_id,
                dataset_id,
                ProvenanceKind.PRIMARY,
            )
        )
        return ExperimentContract(experiment_id, cells, ())
    if experiment_id is ExperimentId.WEIGHT_SENSITIVITY:
        cells = tuple(
            CellContract(
                coordinate=ExperimentCoordinate(
                    experiment_id=experiment_id,
                    variant_id=variant.variant_id,
                    dataset_id=dataset_id,
                    detector_seed=seed,
                    budget_id=budget.budget_id,
                    budget=budget.value,
                    weight_mode=WeightMode.DATASET_COUNT_PROXY,
                ),
                requires_evaluation=True,
                requires_results=True,
                requires_allocations=_CONTRACT_BUDGET_POLICIES,
                requires_analysis=(),
                provenance=ProvenanceKind.PRIMARY,
            )
            for variant in protocol.sensitivity.weight_gamma_variants
            for seed in seeds
            for budget in budgets
        )
        return ExperimentContract(experiment_id, cells, ())
    if experiment_id is ExperimentId.ALLOCATION_STABILITY:
        cells = tuple(
            CellContract(
                coordinate=ExperimentCoordinate(
                    experiment_id=experiment_id,
                    variant_id=ExperimentVariantId.PRIMARY,
                    dataset_id=dataset_id,
                    detector_seed=seed,
                    budget_id=budget.budget_id,
                    budget=budget.value,
                    weight_mode=WeightMode.EQUAL_CLIENT,
                ),
                requires_evaluation=False,
                requires_results=False,
                requires_allocations=(),
                requires_analysis=(
                    AnalysisArtifactId.STABILITY_REPLICATES,
                    AnalysisArtifactId.STABILITY_SUMMARY,
                ),
                provenance=ProvenanceKind.NONE,
            )
            for seed in seeds
            for budget in budgets
        )
        return ExperimentContract(experiment_id, cells, ())
    if experiment_id is ExperimentId.CONSERVATIVE_UTILITY:
        cells = tuple(
            CellContract(
                coordinate=ExperimentCoordinate(
                    experiment_id=experiment_id,
                    variant_id=variant,
                    dataset_id=dataset_id,
                    detector_seed=seed,
                    budget_id=budget.budget_id,
                    budget=budget.value,
                    weight_mode=WeightMode.EQUAL_CLIENT,
                ),
                requires_evaluation=True,
                requires_results=True,
                requires_allocations=(),
                requires_analysis=(
                    (AnalysisArtifactId.CONSERVATIVE_UTILITY_CURVES,)
                    if variant is ExperimentVariantId.CONSERVATIVE_MACRO
                    else ()
                ),
                provenance=ProvenanceKind.PRIMARY,
            )
            for variant in (
                ExperimentVariantId.CONSERVATIVE_MACRO,
                ExperimentVariantId.CONSERVATIVE_CVAR,
            )
            for seed in seeds
            for budget in budgets
        )
        return ExperimentContract(experiment_id, cells, ())
    if experiment_id is ExperimentId.UTILITY_HETEROGENEITY:
        cells = tuple(
            CellContract(
                coordinate=ExperimentCoordinate(
                    experiment_id=experiment_id,
                    variant_id=ExperimentVariantId.PRIMARY,
                    dataset_id=dataset_id,
                    detector_seed=seed,
                    budget_id=budgets[0].budget_id,
                    budget=budgets[0].value,
                    weight_mode=WeightMode.EQUAL_CLIENT,
                ),
                requires_evaluation=False,
                requires_results=False,
                requires_allocations=(),
                requires_analysis=(AnalysisArtifactId.UTILITY_HETEROGENEITY,),
                provenance=ProvenanceKind.NONE,
            )
            for seed in seeds
        )
        return ExperimentContract(experiment_id, cells, ())
    if experiment_id is ExperimentId.EXTERNAL_REPLICATION:
        external = application.external_replication
        cells = _evaluation_cells(
            experiment_id,
            external.detector.seeds,
            external.budgets,
            ExperimentVariantId.EXTERNAL_PRIMARY,
            external.dataset_id,
            ProvenanceKind.EXTERNAL,
        )
        aggregates = tuple(
            AggregateArtifact(
                path=paths.checkpoint_model_path(
                    DetectorCoordinate(dataset_id=external.dataset_id, detector_seed=seed)
                ),
                artifact_name="cic checkpoint model",
            )
            for seed in external.detector.seeds
        )
        return ExperimentContract(experiment_id, cells, aggregates)
    if experiment_id is ExperimentId.EVENT_LEVEL:
        event = application.event_level
        event_dir = paths.event_analysis_dir()
        aggregates = tuple(
            AggregateArtifact(path=event_dir / name.value, artifact_name=name.value)
            for name in (
                ArtifactFileName.EVENT_METRICS,
                ArtifactFileName.EVENT_ROBUSTNESS,
                ArtifactFileName.EVENT_GINI,
                ArtifactFileName.EVENT_BUDGET,
            )
        )
        score_seeds = tuple(
            AggregateArtifact(
                path=paths.scores_dir() / event.dataset_id.value / f"seed-{seed:03d}",
                artifact_name="event scores",
            )
            for seed in event.detector.seeds
        )
        return ExperimentContract(experiment_id, (), aggregates + score_seeds)
    raise ValueError(f"no completion contract for experiment {experiment_id.value}")


def primary_snapshot(
    application: ApplicationConfig,
    paths: ArtifactPaths,
) -> PrimarySnapshot:
    protocol = application.protocol
    dataset_id = protocol.dataset_id
    protocol_digest = protocol_config_digest(protocol)
    provenance_path = paths.preprocessing_provenance_path(dataset_id)
    feature_digest = ""
    split_digest = ""
    if provenance_path.is_file():
        try:
            provenance = read_typed_json(provenance_path, TypeAdapter(PreprocessingProvenance))
        except (ValidationError, ValueError, OSError):
            provenance = None
        else:
            feature_digest = provenance.feature_manifest_digest
            split_digest = provenance.split_manifest_digest
    model_digests: list[tuple[DetectorSeed, ArtifactDigest]] = []
    for seed in protocol.seeds:
        metadata_path = paths.checkpoint_metadata_path(
            DetectorCoordinate(dataset_id=dataset_id, detector_seed=seed)
        )
        if metadata_path.is_file():
            try:
                metadata = read_typed_json(metadata_path, TypeAdapter(CheckpointMetadata))
            except (ValidationError, ValueError, OSError):
                continue
            model_digests.append((seed, metadata.model_digest))
    stored_protocol_path = paths.campaign_protocol_path()
    stored_protocol_digest = None
    if stored_protocol_path.is_file():
        try:
            stored = read_typed_json(stored_protocol_path, TypeAdapter(FabridConfig))
        except (ValidationError, ValueError, OSError):
            pass
        else:
            stored_protocol_digest = protocol_config_digest(stored)
    return PrimarySnapshot(
        protocol_digest=protocol_digest,
        feature_digest=feature_digest,
        split_digest=split_digest,
        model_digests=tuple(model_digests),
        stored_protocol_digest=stored_protocol_digest,
    )


def external_snapshot(
    application: ApplicationConfig,
    paths: ArtifactPaths,
) -> ExternalSnapshot:
    external = application.external_replication
    protocol = external.protocol_for(application.protocol)
    protocol_digest = protocol_config_digest(protocol)
    model_digests: list[tuple[DetectorSeed, ArtifactDigest]] = []
    for seed in external.detector.seeds:
        metadata_path = paths.checkpoint_metadata_path(
            DetectorCoordinate(dataset_id=external.dataset_id, detector_seed=seed)
        )
        if metadata_path.is_file():
            try:
                metadata = read_typed_json(metadata_path, TypeAdapter(CheckpointMetadata))
            except (ValidationError, ValueError, OSError):
                continue
            model_digests.append((seed, metadata.model_digest))
    return ExternalSnapshot(protocol_digest=protocol_digest, model_digests=tuple(model_digests))


def _model_digest_for(
    model_digests: tuple[tuple[DetectorSeed, ArtifactDigest], ...],
    seed: DetectorSeed,
) -> ArtifactDigest | None:
    for saved_seed, digest in model_digests:
        if saved_seed == seed:
            return digest
    return None


def _verify_evaluation_cell(
    paths: ArtifactPaths,
    contract: CellContract,
) -> tuple[CellState, tuple[EnvironmentText, ...]]:
    coordinate = contract.coordinate
    issues: list[EnvironmentText] = []
    evaluation_path = paths.evaluation_summary_path(coordinate)
    try:
        evaluation = read_typed_json(evaluation_path, TypeAdapter(SeedBudgetEvaluation))
    except OSError:
        return CellState.INCOMPLETE, ("evaluation.json missing",)
    except (ValidationError, ValueError):
        return CellState.MALFORMED, ("evaluation.json unparseable",)
    if evaluation.experiment != coordinate:
        return CellState.MALFORMED, ("evaluation.json declares a different coordinate",)
    return CellState.VALID, tuple(issues)


def _verify_primary_result(
    first: ClientResultRecord,
    snapshot: PrimarySnapshot,
    expected_model: ArtifactDigest | None,
) -> tuple[CellState, tuple[EnvironmentText, ...]]:
    if not snapshot.feature_digest or not snapshot.split_digest:
        return CellState.STALE, ("preprocessing provenance missing",)
    if first.model_sha256 != expected_model:
        return CellState.STALE, ("model digest mismatch",)
    if first.feature_sha256 != snapshot.feature_digest:
        return CellState.STALE, ("feature manifest digest mismatch",)
    if first.split_sha256 != snapshot.split_digest:
        return CellState.STALE, ("split manifest digest mismatch",)
    if first.protocol_sha256 != snapshot.protocol_digest:
        return CellState.STALE, ("protocol digest mismatch",)
    return CellState.VALID, ()


def _verify_external_result(
    first: ClientResultRecord,
    snapshot: ExternalSnapshot,
    expected_model: ArtifactDigest | None,
) -> tuple[CellState, tuple[EnvironmentText, ...]]:
    if expected_model is None:
        return CellState.STALE, ("external checkpoint metadata missing",)
    if first.model_sha256 != expected_model:
        return CellState.STALE, ("external model digest mismatch",)
    if first.protocol_sha256 != snapshot.protocol_digest:
        return CellState.STALE, ("external protocol digest mismatch",)
    return CellState.VALID, ()


def _verify_result_records(
    paths: ArtifactPaths,
    contract: CellContract,
    snapshot: PrimarySnapshot | ExternalSnapshot | None,
) -> tuple[CellState, tuple[EnvironmentText, ...]]:
    coordinate = contract.coordinate
    result_path = paths.result_path(coordinate)
    try:
        records = read_parquet_records(result_path)
    except OSError:
        return CellState.INCOMPLETE, ("results.parquet missing",)
    except (ValidationError, ValueError):
        return CellState.MALFORMED, ("results.parquet unparseable",)
    if not records:
        return CellState.INCOMPLETE, ("results.parquet empty",)
    if snapshot is None:
        return CellState.VALID, ()
    first = records[0]
    expected_model = _model_digest_for(snapshot.model_digests, coordinate.detector_seed)
    if isinstance(snapshot, PrimarySnapshot):
        return _verify_primary_result(first, snapshot, expected_model)
    return _verify_external_result(first, snapshot, expected_model)


def read_parquet_records(path: Path) -> tuple[ClientResultRecord, ...]:
    from fabrid.artifacts.parquet import read_parquet_models

    return read_parquet_models(path, ClientResultRecord)


def _verify_analysis_artifact(path: Path) -> tuple[CellState, tuple[EnvironmentText, ...]]:
    import polars as pl
    from polars.exceptions import PolarsError

    try:
        frame = pl.read_parquet(path)
    except OSError:
        return CellState.INCOMPLETE, ("analysis artifact missing",)
    except PolarsError:
        return CellState.MALFORMED, ("analysis artifact unparseable",)
    if frame.height == 0:
        return CellState.INCOMPLETE, ("analysis artifact empty",)
    return CellState.VALID, ()


def _verify_cell(
    paths: ArtifactPaths,
    contract: CellContract,
    snapshot: PrimarySnapshot | ExternalSnapshot | None,
) -> CellVerification:
    root = paths.run_root(contract.coordinate)
    if not root.is_dir():
        return CellVerification(contract.coordinate, CellState.MISSING, ())
    issues: list[EnvironmentText] = []
    worst = CellState.VALID
    if contract.requires_evaluation:
        state, cell_issues = _verify_evaluation_cell(paths, contract)
        issues.extend(cell_issues)
        worst = _worsen(worst, state)
    if contract.requires_results:
        state, cell_issues = _verify_result_records(paths, contract, snapshot)
        issues.extend(cell_issues)
        worst = _worsen(worst, state)
    for policy in contract.requires_allocations:
        path = paths.allocation_path(AllocationCoordinate(contract.coordinate, policy))
        if not path.is_file():
            issues.append(f"{policy.value} allocation snapshot missing")
            worst = _worsen(worst, CellState.INCOMPLETE)
            continue
        try:
            read_typed_json(path, TypeAdapter(AllocationSnapshot))
        except (ValidationError, ValueError, OSError):
            issues.append(f"{policy.value} allocation snapshot unparseable")
            worst = _worsen(worst, CellState.MALFORMED)
    for name in contract.requires_analysis:
        state, cell_issues = _verify_analysis_artifact(
            paths.analysis_path(contract.coordinate, name)
        )
        issues.extend(cell_issues)
        worst = _worsen(worst, state)
    return CellVerification(contract.coordinate, worst, tuple(issues))


def _worsen(current: CellState, incoming: CellState) -> CellState:
    order = (
        CellState.MISSING,
        CellState.INCOMPLETE,
        CellState.STALE,
        CellState.MALFORMED,
        CellState.VALID,
    )
    return incoming if order.index(incoming) < order.index(current) else current


def _experiment_snapshot(
    application: ApplicationConfig,
    paths: ArtifactPaths,
    contract: ExperimentContract,
) -> PrimarySnapshot | ExternalSnapshot | None:
    if not contract.cells:
        return None
    if contract.cells[0].provenance is ProvenanceKind.PRIMARY:
        return primary_snapshot(application, paths)
    if contract.cells[0].provenance is ProvenanceKind.EXTERNAL:
        return external_snapshot(application, paths)
    return None


def _experiment_state(
    cells: tuple[CellVerification, ...],
    contract: ExperimentContract,
    present_cells: ArtifactCount,
    aggregates_present: ArtifactCount,
) -> ExperimentState:
    if (not cells and not contract.aggregates) or (present_cells == 0 and aggregates_present == 0):
        return ExperimentState.NOT_STARTED
    if present_cells < len(cells) or (
        contract.aggregates and aggregates_present < len(contract.aggregates)
    ):
        return ExperimentState.EXECUTION_INCOMPLETE
    if any(cell.state is CellState.MALFORMED for cell in cells):
        return ExperimentState.FAILED
    if any(cell.state is CellState.INCOMPLETE for cell in cells):
        return ExperimentState.EVIDENCE_INCOMPLETE
    if any(cell.state is CellState.STALE for cell in cells):
        return ExperimentState.STALE
    return ExperimentState.PASSED


def verify_experiment(
    application: ApplicationConfig,
    paths: ArtifactPaths,
    experiment_id: ExperimentId,
) -> ExperimentVerification:
    contract = experiment_contract(application, experiment_id)
    snapshot = _experiment_snapshot(application, paths, contract)
    cells = (
        tuple(_verify_cell(paths, cell, snapshot) for cell in contract.cells)
        if contract.cells
        else ()
    )
    issues: list[EnvironmentText] = []
    present_cells = sum(1 for cell in cells if cell.state is not CellState.MISSING)
    valid_cells = sum(1 for cell in cells if cell.state is CellState.VALID)
    aggregates_present = sum(1 for artifact in contract.aggregates if artifact.path.exists())
    for artifact in contract.aggregates:
        if not artifact.path.exists():
            issues.append(f"{artifact.artifact_name} missing at {artifact.path}")
    state = _experiment_state(cells, contract, present_cells, aggregates_present)
    return ExperimentVerification(
        experiment_id=experiment_id,
        state=state,
        expected_cells=len(cells),
        valid_cells=valid_cells,
        cells=cells,
        issues=tuple(issues),
    )


def verify_campaign_evidence(
    application: ApplicationConfig,
    paths: ArtifactPaths,
) -> tuple[ExperimentVerification, ...]:
    enabled = tuple(
        experiment_id
        for experiment_id in ExperimentId
        if application.experiments.enabled(experiment_id)
    )
    return tuple(verify_experiment(application, paths, experiment_id) for experiment_id in enabled)


def experiment_owned_paths(
    paths: ArtifactPaths,
    experiment_id: ExperimentId,
) -> tuple[Path, ...]:
    owned = [paths.runs_dir() / experiment_id.value]
    if experiment_id is ExperimentId.EVENT_LEVEL:
        owned.append(paths.scores_dir() / "gotham")
    elif experiment_id is ExperimentId.EXTERNAL_REPLICATION:
        owned.append(paths.checkpoints_dir() / "cic_iot_diad")
    return tuple(owned)


def delete_experiment_evidence(paths: ArtifactPaths, experiment_id: ExperimentId) -> None:
    outputs_root = paths.outputs_root.resolve()
    for path in experiment_owned_paths(paths, experiment_id):
        resolved = path.resolve()
        if not resolved.is_relative_to(outputs_root):
            raise ValueError(f"refusing to delete outside outputs root: {path}")
        if resolved.is_dir():
            shutil.rmtree(resolved)
        elif resolved.exists():
            resolved.unlink()
        logger.info("[DELETE] experiment %s evidence at %s", experiment_id.value, path)
    bundle_copy = paths.publication().experiment_dir(experiment_id)
    resolved = bundle_copy.resolve()
    results_root = paths.results_root.resolve()
    if resolved.is_relative_to(results_root) and resolved.is_dir():
        shutil.rmtree(resolved)
        logger.info("[DELETE] experiment %s bundle copy at %s", experiment_id.value, bundle_copy)


def delete_results_bundle(paths: ArtifactPaths) -> None:
    root = paths.results_root.resolve()
    if root.is_dir():
        shutil.rmtree(root)


def delete_campaign_evidence(paths: ArtifactPaths) -> None:
    directories = (
        paths.preprocessed_root,
        paths.checkpoints_dir(),
        paths.scores_dir(),
        paths.runs_dir(),
        paths.tables_dir(),
        paths.figures_dir(),
        paths.results_root,
    )
    for directory in directories:
        resolved = directory.resolve()
        if resolved.is_dir():
            shutil.rmtree(resolved)
            logger.info("[DELETE] campaign evidence at %s", directory)
    report_path = paths.report_path()
    if report_path.is_file():
        report_path.unlink()
        logger.info("[DELETE] campaign evidence at %s", report_path)
