from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pydantic import TypeAdapter

from fabrid.allocation.problem import equal_client_weights
from fabrid.artifacts.json import write_typed_json
from fabrid.artifacts.paths import ExperimentCoordinate
from fabrid.config import (
    AllocationPolicy,
    AnalysisArtifactId,
    CampaignAnalysisId,
    DatasetId,
    ExperimentId,
    ExperimentVariantId,
    WeightMode,
)
from fabrid.evaluation.gates import PracticalGateAnalysis
from fabrid.evaluation.inference import PrimaryInference
from fabrid.evaluation.metrics import CompletedPolicyEvaluation
from tests.support import isolated_context, smoke_application

pytestmark = pytest.mark.smoke
_FEATURE_COUNT = 115
_BENIGN_ROWS = 300
_ATTACK_ROWS = 400


def _write_feature_csv(path: Path, rows: int, mean: float, seed: int) -> None:
    rng = np.random.default_rng(seed)
    values = rng.normal(mean, 0.5, size=(rows, _FEATURE_COUNT))
    header = ",".join(f"f{index}" for index in range(_FEATURE_COUNT))
    np.savetxt(path, values, delimiter=",", header=header, comments="")


def _build_synthetic_raw_tree(raw_root: Path) -> None:
    layout = smoke_application().datasets.nbaiot
    dataset_root = raw_root / layout.directory_name
    for device_index, device in enumerate(layout.devices):
        device_dir = dataset_root / device
        device_dir.mkdir(parents=True, exist_ok=True)
        _write_feature_csv(
            device_dir / "benign_traffic.csv",
            _BENIGN_ROWS,
            0.0,
            seed=100 + device_index,
        )
        attack_mean = 1.5 + 0.25 * device_index
        for family, subtypes in (
            ("gafgyt_attacks", ("combo", "junk")),
            ("mirai_attacks", ("ack", "scan")),
        ):
            family_dir = device_dir / family
            family_dir.mkdir(parents=True, exist_ok=True)
            for subtype_index, subtype in enumerate(subtypes):
                _write_feature_csv(
                    family_dir / f"{subtype}.csv",
                    _ATTACK_ROWS,
                    attack_mean + 0.2 * subtype_index,
                    seed=200 + device_index * 10 + subtype_index,
                )


def _assert_artifact(path: Path, message: str) -> None:
    assert path.exists(), message


def test_smoke_workflow(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    _build_synthetic_raw_tree(raw_root)
    context = isolated_context(smoke_application(), tmp_path, raw_data_root=raw_root)
    application = context.config
    paths = context.paths
    config = application.protocol
    from fabrid.execution.campaign import evaluation_provenance
    from fabrid.execution.prerequisites import (
        load_or_materialize_scores,
        load_or_train_seed,
        prepare_federation,
    )
    from fabrid.execution.resources import configure_campaign_logging, report_resources
    from fabrid.experiments.matched_budget import (
        persist_seed_budget,
        run_seed_budget,
    )

    campaign_logger = configure_campaign_logging(paths)
    report_resources(campaign_logger, "start")
    _assert_artifact(paths.campaign_log_path(), "campaign log")
    prepared = prepare_federation(context)
    _assert_artifact(
        paths.preprocessing_features_path(DatasetId.NBAIOT), "preprocessed features parquet"
    )
    _assert_artifact(paths.preprocessing_manifest_path(DatasetId.NBAIOT), "feature manifest")
    prepared_again = prepare_federation(context)
    assert prepared_again.provenance.built_at == prepared.provenance.built_at
    seed = config.seeds[0]
    trained = load_or_train_seed(prepared, paths, config, seed)
    _assert_artifact(paths.checkpoint_model_path(trained.coordinate), "checkpoint model")
    trained_again = load_or_train_seed(prepared, paths, config, seed)
    assert trained_again.metadata.model_digest == trained.metadata.model_digest
    stored = load_or_materialize_scores(prepared, trained, paths, config)
    assert stored.score_sha256
    stored_again = load_or_materialize_scores(prepared, trained, paths, config)
    assert stored_again.score_sha256 == stored.score_sha256
    provenance = evaluation_provenance(prepared, stored, trained, config)
    budget_level = config.budgets[0]
    run = run_seed_budget(
        ExperimentCoordinate(
            experiment_id=ExperimentId.MATCHED_BUDGET,
            variant_id=ExperimentVariantId.PRIMARY,
            dataset_id=DatasetId.NBAIOT,
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
    assert run.records
    completed_policies = {
        evaluation.policy
        for evaluation in run.evaluation.policies
        if isinstance(evaluation, CompletedPolicyEvaluation)
    }
    assert AllocationPolicy.FABRID_MACRO in completed_policies
    assert AllocationPolicy.FABRID_MACRO_FINITE_SAFE in completed_policies
    assert AllocationPolicy.FABRID_CVAR in completed_policies
    assert AllocationPolicy.FABRID_CVAR_FINITE_SAFE in completed_policies
    assert AllocationPolicy.FABRID_MACRO_NO_RATE_MINIMIZATION in completed_policies
    assert AllocationPolicy.COMMON_RATE_MACRO in completed_policies
    assert AllocationPolicy.EQ_FPR_MATCHED_MACRO in completed_policies
    assert AllocationPolicy.EQ_FPR_MATCHED_CVAR in completed_policies
    assert len(run.matched_equal_rate_selections) == 2
    assert len(run.equal_rate_calibration_frontier) == 2 * len(config.alpha_grid)
    assert sum(item.selected for item in run.equal_rate_calibration_frontier) == 2
    persist_seed_budget(paths, run)
    _assert_artifact(paths.evaluation_summary_path(run.evaluation.experiment), "evaluation summary")
    _assert_artifact(
        paths.analysis_path(
            run.evaluation.experiment, AnalysisArtifactId.EQUAL_RATE_CALIBRATION_FRONTIER
        ),
        "equal-rate calibration frontier",
    )
    from fabrid.evaluation.gates import analyze_practical_gates
    from fabrid.evaluation.inference import analyze_primary_inference

    primary = analyze_primary_inference((run.evaluation,), config.budgets, config.statistics)
    gates = analyze_practical_gates(
        (run.evaluation,),
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
    from fabrid.reporting.build import build_campaign_report, package_results
    from fabrid.validation.reproducibility import (
        ReproducibilityMetadata,
        capture_reproducibility_metadata,
        now_iso,
    )

    evidence, report_path = build_campaign_report(paths, config)
    _assert_artifact(report_path, "report.json")
    assert evidence.tables
    assert evidence.figures
    reproducibility = capture_reproducibility_metadata(
        config.seeds,
        (),
        command_line="smoke test",
        started_at=now_iso(),
        finished_at=now_iso(),
    )
    write_typed_json(
        reproducibility,
        TypeAdapter(ReproducibilityMetadata),
        paths.campaign_reproducibility_path(),
    )
    bundle = package_results(config, paths, reproducibility, evidence)
    _assert_artifact(bundle / "manifest.json", "results manifest")
    _assert_artifact(bundle / "configuration.json", "results configuration")
    _assert_artifact(bundle / "reproducibility.json", "results reproducibility")
    _assert_artifact(bundle / "summary.json", "results summary")
    _assert_artifact(bundle / "report.json", "bundle report.json")
    assert not (bundle / "report").exists(), "results/report/ subdirectory must not exist"
    assert not any(bundle.rglob("*.md")), "results bundle must contain no Markdown files"
    assert not (paths.outputs_root / "reports").exists(), "outputs/reports/ must not exist"
    assert paths.report_path() == paths.outputs_root / "report.json"
    layout = smoke_application().datasets.nbaiot
    mutated_device_dir = raw_root / layout.directory_name / layout.devices[0]
    _write_feature_csv(
        mutated_device_dir / "benign_traffic.csv",
        _BENIGN_ROWS,
        mean=5.0,
        seed=999,
    )
    mutated_prepared = prepare_federation(context)
    assert mutated_prepared.raw_digest != prepared.raw_digest
    assert mutated_prepared.provenance.built_at != prepared.provenance.built_at
    assert mutated_prepared.feature_manifest.columns == prepared.feature_manifest.columns
    assert len(mutated_prepared.feature_manifest.columns) == _FEATURE_COUNT
    mutated_trained = load_or_train_seed(mutated_prepared, paths, config, seed)
    assert mutated_trained.metadata.model_digest != trained.metadata.model_digest
    assert mutated_trained.metadata.feature_count == trained.metadata.feature_count
    assert mutated_trained.metadata.hidden_layers == trained.metadata.hidden_layers
    mutated_stored = load_or_materialize_scores(mutated_prepared, mutated_trained, paths, config)
    assert mutated_stored.score_sha256 != stored.score_sha256
    mutated_prepared_again = prepare_federation(context)
    assert mutated_prepared_again.provenance.built_at == mutated_prepared.provenance.built_at
    mutated_trained_again = load_or_train_seed(mutated_prepared, paths, config, seed)
    assert mutated_trained_again.metadata.model_digest == mutated_trained.metadata.model_digest
    mutated_stored_again = load_or_materialize_scores(
        mutated_prepared, mutated_trained, paths, config
    )
    assert mutated_stored_again.score_sha256 == mutated_stored.score_sha256
