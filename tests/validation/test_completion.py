from __future__ import annotations

from pathlib import Path

import polars as pl
from pydantic import TypeAdapter

from fabrid.allocation.optimization import (
    AllocationSnapshot,
    SolverEvidence,
    SolverStatus,
)
from fabrid.artifacts.json import digest_file, write_typed_json
from fabrid.artifacts.parquet import write_parquet_models
from fabrid.artifacts.paths import (
    AllocationCoordinate,
    ArtifactPaths,
    DetectorCoordinate,
    ExperimentCoordinate,
)
from fabrid.config import (
    AllocationPolicy,
    AnalysisArtifactId,
    ApplicationConfig,
    ArtifactDigest,
    ExperimentId,
    ExperimentVariantId,
    WeightMode,
)
from fabrid.detector.training import (
    CheckpointMetadata,
    DetectorInputSpace,
    FederatedTrainingSettings,
)
from fabrid.evaluation.metrics import ClientResultRecord, SeedBudgetEvaluation
from fabrid.reporting.build import (
    ArtifactCounts,
    BundleKind,
    KindCount,
    ManifestEntry,
    ResultsManifest,
    verify_results_bundle,
)
from fabrid.validation.completion import (
    CellState,
    ExperimentState,
    delete_campaign_evidence,
    delete_experiment_evidence,
    delete_results_bundle,
    experiment_contract,
    experiment_owned_paths,
    verify_campaign_evidence,
    verify_experiment,
)
from fabrid.validation.datasets import PreprocessingProvenance
from tests.support import isolated_paths, smoke_application

_DIGEST: ArtifactDigest = "0" * 64


def _protocol_digest(application: ApplicationConfig) -> ArtifactDigest:
    from fabrid.artifacts.json import protocol_config_digest

    return protocol_config_digest(application.protocol)


def _write_provenance(
    paths: ArtifactPaths, application: ApplicationConfig
) -> tuple[ArtifactDigest, ArtifactDigest]:
    feature_digest = _DIGEST
    split_digest = "1" * 64
    provenance = PreprocessingProvenance(
        dataset_id=application.protocol.dataset_id,
        schema_version=application.protocol.protocol_version,
        raw_digest="2" * 64,
        preprocessing_config_digest=_protocol_digest(application),
        feature_manifest_digest=feature_digest,
        split_manifest_digest=split_digest,
        features_digest="4" * 64,
        built_at="test",
    )
    write_typed_json(
        provenance,
        TypeAdapter(PreprocessingProvenance),
        paths.preprocessing_provenance_path(application.protocol.dataset_id),
    )
    return feature_digest, split_digest


def _write_checkpoints(paths: ArtifactPaths, application: ApplicationConfig) -> ArtifactDigest:
    model_digest = "3" * 64
    metadata = CheckpointMetadata(
        dataset_id=application.protocol.dataset_id,
        detector_seed=application.protocol.seeds[0],
        feature_count=8,
        hidden_layers=(8, 4),
        input_space=DetectorInputSpace.PER_CLIENT_ZSCORE,
        training=FederatedTrainingSettings(
            learning_rate=0.001, local_epochs=1, rounds=2, batch_size=32
        ),
        preprocessing_digest="5" * 64,
        model_digest=model_digest,
        scaler_digests=(),
    )
    for seed in application.protocol.seeds:
        write_typed_json(
            metadata,
            TypeAdapter(CheckpointMetadata),
            paths.checkpoint_metadata_path(
                DetectorCoordinate(dataset_id=application.protocol.dataset_id, detector_seed=seed)
            ),
        )
    return model_digest


def _record(
    coordinate: ExperimentCoordinate,
    application: ApplicationConfig,
    feature_digest: ArtifactDigest,
    split_digest: ArtifactDigest,
    model_digest: ArtifactDigest,
    *,
    policy: AllocationPolicy = AllocationPolicy.FABRID_MACRO,
) -> ClientResultRecord:
    return ClientResultRecord(
        experiment_id=coordinate.experiment_id,
        dataset_id=coordinate.dataset_id,
        seed=coordinate.detector_seed,
        budget_id=coordinate.budget_id,
        budget_value=coordinate.budget,
        weight_mode=coordinate.weight_mode,
        policy=policy,
        client_id="c1",
        alpha_selected=0.001,
        threshold=0.5,
        calibration_n=10,
        calibration_distinct_score_n=10,
        calibration_tied_score_n=0,
        calibration_smallest_attainable_rate=0.1,
        calibration_attained_rate=0.0,
        calibration_requested_attainment_error=0.001,
        nominal_weight=1.0,
        realized_weight=1.0,
        n_benign_test=10,
        n_attack_test=0,
        attack_subtype=None,
        tp=0,
        fn=0,
        fp=0,
        tn=10,
        fpr=0.0,
        tpr=0.0,
        macro_attack_recall=0.0,
        false_alert_count=0,
        solver_status=SolverStatus.NOT_APPLICABLE,
        solver_objective=None,
        solver_gap=None,
        solver_runtime_ms=None,
        model_sha256=model_digest,
        score_sha256="4" * 64,
        split_sha256=split_digest,
        feature_sha256=feature_digest,
        protocol_sha256=_protocol_digest(application),
        git_commit="0" * 40,
    )


def _allocation_snapshot(policy: AllocationPolicy) -> AllocationSnapshot:
    return AllocationSnapshot(
        policy=policy,
        decisions=(),
        solver=SolverEvidence(status=SolverStatus.NOT_APPLICABLE, stages=()),
    )


def write_valid_experiment(tmp_path: Path, application: ApplicationConfig) -> ArtifactPaths:
    paths = isolated_paths(tmp_path)
    feature_digest, split_digest = _write_provenance(paths, application)
    model_digest = _write_checkpoints(paths, application)
    contract = experiment_contract(application, ExperimentId.MATCHED_BUDGET)
    for cell in contract.cells:
        coordinate = cell.coordinate
        write_parquet_models(
            paths.result_path(coordinate),
            (
                _record(coordinate, application, feature_digest, split_digest, model_digest),
                _record(
                    coordinate,
                    application,
                    feature_digest,
                    split_digest,
                    model_digest,
                    policy=AllocationPolicy.FABRID_CVAR,
                ),
            ),
        )
        write_typed_json(
            SeedBudgetEvaluation(experiment=coordinate, policies=()),
            TypeAdapter(SeedBudgetEvaluation),
            paths.evaluation_summary_path(coordinate),
        )
        for policy in cell.requires_allocations:
            write_typed_json(
                _allocation_snapshot(policy),
                TypeAdapter(AllocationSnapshot),
                paths.allocation_path(AllocationCoordinate(coordinate, policy)),
            )
    return paths


def test_not_started_when_no_evidence(tmp_path: Path) -> None:
    application = smoke_application()
    paths = isolated_paths(tmp_path)
    verification = verify_experiment(application, paths, ExperimentId.MATCHED_BUDGET)
    assert verification.state is ExperimentState.NOT_STARTED
    assert verification.expected_cells == 1


def test_complete_evidence_passes(tmp_path: Path) -> None:
    application = smoke_application()
    paths = write_valid_experiment(tmp_path, application)
    verification = verify_experiment(application, paths, ExperimentId.MATCHED_BUDGET)
    assert verification.state is ExperimentState.PASSED
    assert verification.valid_cells == verification.expected_cells
    assert all(cell.state is CellState.VALID for cell in verification.cells)


def test_verification_is_read_only(tmp_path: Path) -> None:
    application = smoke_application()
    paths = write_valid_experiment(tmp_path, application)
    before = {
        path.relative_to(tmp_path): digest_file(path)
        for path in sorted(tmp_path.rglob("*"))
        if path.is_file()
    }
    verify_experiment(application, paths, ExperimentId.MATCHED_BUDGET)
    after = {
        path.relative_to(tmp_path): digest_file(path)
        for path in sorted(tmp_path.rglob("*"))
        if path.is_file()
    }
    assert before == after


def test_missing_evaluation_json_prevents_pass(tmp_path: Path) -> None:
    application = smoke_application()
    paths = write_valid_experiment(tmp_path, application)
    coordinate = experiment_contract(application, ExperimentId.MATCHED_BUDGET).cells[0].coordinate
    paths.evaluation_summary_path(coordinate).unlink()
    verification = verify_experiment(application, paths, ExperimentId.MATCHED_BUDGET)
    assert verification.state is ExperimentState.EVIDENCE_INCOMPLETE


def test_malformed_evaluation_json_fails(tmp_path: Path) -> None:
    application = smoke_application()
    paths = write_valid_experiment(tmp_path, application)
    coordinate = experiment_contract(application, ExperimentId.MATCHED_BUDGET).cells[0].coordinate
    paths.evaluation_summary_path(coordinate).write_text("{not json", encoding="utf-8")
    verification = verify_experiment(application, paths, ExperimentId.MATCHED_BUDGET)
    assert verification.state is ExperimentState.FAILED


def test_empty_results_prevent_pass(tmp_path: Path) -> None:
    application = smoke_application()
    paths = write_valid_experiment(tmp_path, application)
    coordinate = experiment_contract(application, ExperimentId.MATCHED_BUDGET).cells[0].coordinate
    pl.DataFrame().write_parquet(paths.result_path(coordinate))
    verification = verify_experiment(application, paths, ExperimentId.MATCHED_BUDGET)
    assert verification.state is ExperimentState.EVIDENCE_INCOMPLETE


def test_stale_when_protocol_changes(tmp_path: Path) -> None:
    application = smoke_application()
    paths = write_valid_experiment(tmp_path, application)
    changed = application.model_copy(
        update={
            "protocol": application.protocol.model_copy(
                update={
                    "benign_splits": application.protocol.benign_splits.model_copy(
                        update={"train_end": 0.6}
                    )
                }
            )
        }
    )
    verification = verify_experiment(changed, paths, ExperimentId.MATCHED_BUDGET)
    assert verification.state is ExperimentState.STALE


def test_stale_when_checkpoint_digest_changes(tmp_path: Path) -> None:
    application = smoke_application()
    paths = write_valid_experiment(tmp_path, application)
    for seed in application.protocol.seeds:
        metadata_path = paths.checkpoint_metadata_path(
            DetectorCoordinate(dataset_id=application.protocol.dataset_id, detector_seed=seed)
        )
        metadata = TypeAdapter(CheckpointMetadata).validate_json(metadata_path.read_bytes())
        write_typed_json(
            metadata.model_copy(update={"model_digest": "5" * 64}),
            TypeAdapter(CheckpointMetadata),
            metadata_path,
        )
    verification = verify_experiment(application, paths, ExperimentId.MATCHED_BUDGET)
    assert verification.state is ExperimentState.STALE


def test_snapshot_missing_provenance_marks_cells_stale(tmp_path: Path) -> None:
    application = smoke_application()
    paths = write_valid_experiment(tmp_path, application)
    paths.preprocessing_provenance_path(application.protocol.dataset_id).unlink()
    verification = verify_experiment(application, paths, ExperimentId.MATCHED_BUDGET)
    assert verification.state is ExperimentState.STALE


def test_overwrite_deletes_only_owned_evidence(tmp_path: Path) -> None:
    application = smoke_application()
    paths = write_valid_experiment(tmp_path, application)
    weight_contract = experiment_contract(application, ExperimentId.WEIGHT_SENSITIVITY)
    for cell in weight_contract.cells:
        (paths.run_root(cell.coordinate) / "marker").mkdir(parents=True)
    bundle_copy = paths.publication().experiment_dir(ExperimentId.MATCHED_BUDGET)
    bundle_copy.mkdir(parents=True)
    (bundle_copy / "old.json").write_text("{}", encoding="utf-8")
    owned = experiment_owned_paths(paths, ExperimentId.MATCHED_BUDGET)
    delete_experiment_evidence(paths, ExperimentId.MATCHED_BUDGET)
    for path in owned:
        assert not path.exists()
    assert not bundle_copy.exists()
    assert (paths.runs_dir() / ExperimentId.WEIGHT_SENSITIVITY.value).is_dir()
    assert paths.preprocessing_provenance_path(application.protocol.dataset_id).is_file()
    assert paths.checkpoint_metadata_path(
        DetectorCoordinate(
            dataset_id=application.protocol.dataset_id,
            detector_seed=application.protocol.seeds[0],
        )
    ).is_file()


def test_campaign_overwrite_clears_regenerable_roots(tmp_path: Path) -> None:
    application = smoke_application()
    paths = write_valid_experiment(tmp_path, application)
    (tmp_path / "raw").mkdir(exist_ok=True)
    (tmp_path / "raw" / "dataset").write_text("raw", encoding="utf-8")
    delete_campaign_evidence(paths)
    assert not paths.runs_dir().exists()
    assert not paths.preprocessed_root.exists()
    assert not paths.results_root.exists()
    assert (tmp_path / "raw" / "dataset").is_file()


def test_results_bundle_verification(tmp_path: Path) -> None:
    application = smoke_application()
    paths = isolated_paths(tmp_path)
    bundle = paths.publication().bundle_dir()
    bundle.mkdir(parents=True)
    (bundle / "file.json").write_text("{}", encoding="utf-8")
    manifest = ResultsManifest(
        bundle_version="1",
        generated_at="test",
        protocol_version=application.protocol.protocol_version,
        git_commit="0" * 40,
        evidence_digest=_DIGEST,
        entries=(
            ManifestEntry(
                path="file.json",
                digest=digest_file(bundle / "file.json"),
                kind=BundleKind.JSON,
                owning_experiment=None,
            ),
        ),
        counts=ArtifactCounts(
            files=1,
            by_kind=(KindCount(kind=BundleKind.JSON, count=1),),
            tables=0,
            figures=0,
            contrasts=0,
            budget_policy_rows=0,
        ),
    )
    write_typed_json(manifest, TypeAdapter(ResultsManifest), paths.publication().manifest_path())
    ok, issues = verify_results_bundle(paths, _DIGEST)
    assert ok, issues
    ok, issues = verify_results_bundle(paths, "6" * 64)
    assert not ok
    assert any("evidence digest" in issue for issue in issues)


def test_results_bundle_detects_missing_and_corrupt_entries(tmp_path: Path) -> None:
    application = smoke_application()
    paths = isolated_paths(tmp_path)
    bundle = paths.publication().bundle_dir()
    bundle.mkdir(parents=True)
    (bundle / "file.json").write_text("{}", encoding="utf-8")
    manifest = ResultsManifest(
        bundle_version="1",
        generated_at="test",
        protocol_version=application.protocol.protocol_version,
        git_commit="0" * 40,
        evidence_digest=_DIGEST,
        entries=(
            ManifestEntry(
                path="file.json",
                digest=digest_file(bundle / "file.json"),
                kind=BundleKind.JSON,
                owning_experiment=None,
            ),
            ManifestEntry(
                path="missing.json",
                digest=_DIGEST,
                kind=BundleKind.JSON,
                owning_experiment=None,
            ),
        ),
        counts=ArtifactCounts(
            files=2,
            by_kind=(KindCount(kind=BundleKind.JSON, count=2),),
            tables=0,
            figures=0,
            contrasts=0,
            budget_policy_rows=0,
        ),
    )
    write_typed_json(manifest, TypeAdapter(ResultsManifest), paths.publication().manifest_path())
    ok, issues = verify_results_bundle(paths)
    assert not ok
    assert any("missing" in issue for issue in issues)
    (bundle / "file.json").write_text("corrupted", encoding="utf-8")
    ok, issues = verify_results_bundle(paths)
    assert not ok
    assert any("digest mismatch" in issue for issue in issues)


def test_results_bundle_requires_manifest(tmp_path: Path) -> None:
    paths = isolated_paths(tmp_path)
    ok, issues = verify_results_bundle(paths)
    assert not ok
    assert issues


def test_delete_results_bundle_only_clears_delivery(tmp_path: Path) -> None:
    application = smoke_application()
    paths = write_valid_experiment(tmp_path, application)
    bundle = paths.publication().bundle_dir()
    bundle.mkdir(parents=True)
    (bundle / "artifact.json").write_text("{}", encoding="utf-8")
    delete_results_bundle(paths)
    assert not bundle.exists()
    assert (paths.runs_dir() / ExperimentId.MATCHED_BUDGET.value).is_dir()


def test_pending_experiments_skip_passed_evidence(tmp_path: Path) -> None:
    from fabrid.execution.application import ApplicationContext
    from fabrid.execution.campaign import pending_experiments

    application = smoke_application()
    paths = write_valid_experiment(tmp_path, application)
    context = ApplicationContext(config=application, paths=paths, repository_root=tmp_path)
    pending = pending_experiments(context, overwrite=False)
    assert ExperimentId.MATCHED_BUDGET not in pending
    pending_all = pending_experiments(context, overwrite=True)
    assert ExperimentId.MATCHED_BUDGET in pending_all


def test_pending_experiments_include_stale_evidence(tmp_path: Path) -> None:
    from fabrid.execution.application import ApplicationContext
    from fabrid.execution.campaign import pending_experiments

    application = smoke_application()
    paths = write_valid_experiment(tmp_path, application)
    changed = application.model_copy(
        update={
            "protocol": application.protocol.model_copy(
                update={
                    "benign_splits": application.protocol.benign_splits.model_copy(
                        update={"train_end": 0.6}
                    )
                }
            )
        }
    )
    context = ApplicationContext(config=changed, paths=paths, repository_root=tmp_path)
    pending = pending_experiments(context, overwrite=False)
    assert ExperimentId.MATCHED_BUDGET in pending


def test_campaign_evidence_verifies_enabled_experiments(tmp_path: Path) -> None:
    application = smoke_application()
    paths = write_valid_experiment(tmp_path, application)
    verifications = verify_campaign_evidence(application, paths)
    by_id = {verification.experiment_id: verification for verification in verifications}
    assert ExperimentId.MATCHED_BUDGET in by_id
    assert by_id[ExperimentId.MATCHED_BUDGET].state is ExperimentState.PASSED
    enabled = tuple(
        experiment_id
        for experiment_id in ExperimentId
        if application.experiments.enabled(experiment_id)
    )
    assert set(by_id) == set(enabled)


def test_cross_experiment_isolation(tmp_path: Path) -> None:
    application = smoke_application()
    paths = isolated_paths(tmp_path)
    feature_digest, split_digest = _write_provenance(paths, application)
    model_digest = _write_checkpoints(paths, application)
    matched = experiment_contract(application, ExperimentId.MATCHED_BUDGET)
    for cell in matched.cells:
        coordinate = cell.coordinate
        write_parquet_models(
            paths.result_path(coordinate),
            (_record(coordinate, application, feature_digest, split_digest, model_digest),),
        )
        write_typed_json(
            SeedBudgetEvaluation(experiment=coordinate, policies=()),
            TypeAdapter(SeedBudgetEvaluation),
            paths.evaluation_summary_path(coordinate),
        )
    delete_experiment_evidence(paths, ExperimentId.MATCHED_BUDGET)
    assert verify_experiment(application, paths, ExperimentId.MATCHED_BUDGET).state is (
        ExperimentState.NOT_STARTED
    )
    assert verify_experiment(application, paths, ExperimentId.UTILITY_HETEROGENEITY).state is (
        ExperimentState.NOT_STARTED
    )


def test_contract_expected_cell_counts(tmp_path: Path) -> None:
    application = smoke_application()
    protocol = application.protocol
    assert len(experiment_contract(application, ExperimentId.MATCHED_BUDGET).cells) == (
        len(protocol.seeds) * len(protocol.budgets)
    )
    assert len(experiment_contract(application, ExperimentId.ATTACK_SUBTYPE_DISJOINT).cells) == (
        len(protocol.seeds) * len(protocol.generalization.rotations) * len(protocol.budgets)
    )
    assert len(experiment_contract(application, ExperimentId.BOTNET_FAMILY_DISJOINT).cells) == (
        len(protocol.seeds) * len(protocol.generalization.family_directions) * len(protocol.budgets)
    )
    assert len(experiment_contract(application, ExperimentId.WEIGHT_SENSITIVITY).cells) == (
        len(protocol.seeds)
        * len(protocol.sensitivity.weight_gamma_variants)
        * len(protocol.budgets)
    )
    assert len(experiment_contract(application, ExperimentId.ALLOCATION_STABILITY).cells) == (
        len(protocol.seeds) * len(protocol.budgets)
    )
    assert len(experiment_contract(application, ExperimentId.CONSERVATIVE_UTILITY).cells) == (
        len(protocol.seeds) * 2 * len(protocol.budgets)
    )
    assert len(experiment_contract(application, ExperimentId.UTILITY_HETEROGENEITY).cells) == (
        len(protocol.seeds)
    )
    external = application.external_replication
    assert len(experiment_contract(application, ExperimentId.EXTERNAL_REPLICATION).cells) == (
        len(external.detector.seeds) * len(external.budgets)
    )


def test_contract_weight_modes_and_conservative_shape(tmp_path: Path) -> None:
    application = smoke_application()
    matched = experiment_contract(application, ExperimentId.MATCHED_BUDGET).cells[0]
    assert matched.coordinate.weight_mode is WeightMode.EQUAL_CLIENT
    assert matched.requires_allocations
    weighted = experiment_contract(application, ExperimentId.WEIGHT_SENSITIVITY).cells[0]
    assert weighted.coordinate.weight_mode is WeightMode.DATASET_COUNT_PROXY
    conservative = experiment_contract(application, ExperimentId.CONSERVATIVE_UTILITY)
    macro_cells = tuple(
        cell
        for cell in conservative.cells
        if cell.coordinate.variant_id is ExperimentVariantId.CONSERVATIVE_MACRO
    )
    cvar_cells = tuple(
        cell
        for cell in conservative.cells
        if cell.coordinate.variant_id is ExperimentVariantId.CONSERVATIVE_CVAR
    )
    assert macro_cells
    assert cvar_cells
    assert macro_cells[0].requires_allocations == ()
    assert macro_cells[0].requires_analysis == (AnalysisArtifactId.CONSERVATIVE_UTILITY_CURVES,)
    assert cvar_cells[0].requires_analysis == ()


def test_snapshot_matches_contract_provenance_kinds(tmp_path: Path) -> None:
    application = smoke_application()
    contract = experiment_contract(application, ExperimentId.MATCHED_BUDGET)
    assert contract.cells[0].provenance.value == "primary"
    external_contract = experiment_contract(application, ExperimentId.EXTERNAL_REPLICATION)
    assert external_contract.cells[0].provenance.value == "external"
    assert len(external_contract.aggregates) == len(application.external_replication.detector.seeds)
    event_contract = experiment_contract(application, ExperimentId.EVENT_LEVEL)
    assert event_contract.cells == ()
    assert event_contract.aggregates
