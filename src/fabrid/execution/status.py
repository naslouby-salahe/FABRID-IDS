from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from pydantic import TypeAdapter

from fabrid.artifacts.json import digest_text, read_typed_json
from fabrid.artifacts.paths import ArtifactFileName, ArtifactPaths, DetectorCoordinate
from fabrid.config import (
    ApplicationConfig,
    ArtifactCount,
    CampaignAnalysisId,
    EnvironmentText,
    ExperimentId,
    FabridConfig,
)
from fabrid.execution.application import ApplicationContext
from fabrid.reporting.build import verify_results_bundle
from fabrid.validation.completion import (
    ExperimentState,
    verify_campaign_evidence,
    verify_experiment,
)


@dataclass(frozen=True, slots=True)
class ComponentStatusRow:
    component: EnvironmentText
    state: ExperimentState | ComponentState
    expected_count: ArtifactCount
    present_count: ArtifactCount
    detail: EnvironmentText
    issues: tuple[EnvironmentText, ...] = ()


class ComponentState(StrEnum):
    VERIFIED = "verified"
    STALE = "stale"
    MISSING = "missing"
    PARTIAL = "partial"
    INVALID = "invalid"


def _stored_protocol_current(
    application: ApplicationConfig,
    context: ApplicationContext,
) -> bool:
    path = context.paths.campaign_protocol_path()
    if not path.is_file():
        return False
    try:
        stored = read_typed_json(path, TypeAdapter(FabridConfig))
    except (ValueError, OSError):
        return False
    current = digest_text(
        (application.protocol.protocol_version, application.protocol.model_dump_json())
    )
    return digest_text((stored.protocol_version, stored.model_dump_json())) == current


def _checkpoint_count(context: ApplicationContext) -> ArtifactCount:
    dataset_id = context.config.protocol.dataset_id
    return sum(
        context.paths.checkpoint_metadata_path(
            DetectorCoordinate(dataset_id=dataset_id, detector_seed=seed)
        ).is_file()
        for seed in context.config.protocol.seeds
    )


def _checkpoint_state(present: ArtifactCount, expected: ArtifactCount) -> ComponentState:
    if present == expected:
        return ComponentState.VERIFIED
    if present:
        return ComponentState.PARTIAL
    return ComponentState.MISSING


def _bundle_state(bundle_ok: bool, results_root_exists: bool) -> ComponentState:
    if bundle_ok:
        return ComponentState.VERIFIED
    if not results_root_exists:
        return ComponentState.MISSING
    return ComponentState.INVALID


def _protocol_row(
    application: ApplicationConfig, context: ApplicationContext
) -> ComponentStatusRow:
    path = context.paths.campaign_protocol_path()
    present = path.is_file()
    if present:
        state = (
            ComponentState.VERIFIED
            if _stored_protocol_current(application, context)
            else ComponentState.STALE
        )
    else:
        state = ComponentState.MISSING
    return ComponentStatusRow(
        component="protocol",
        state=state,
        expected_count=1,
        present_count=1 if present else 0,
        detail=ArtifactFileName.PROTOCOL,
    )


def _preprocessing_row(paths: ArtifactPaths, protocol: FabridConfig) -> ComponentStatusRow:
    features = paths.preprocessing_features_path(protocol.dataset_id)
    provenance = paths.preprocessing_provenance_path(protocol.dataset_id)
    if features.is_file() and provenance.is_file():
        state = ComponentState.VERIFIED
    elif features.is_file():
        state = ComponentState.PARTIAL
    else:
        state = ComponentState.MISSING
    return ComponentStatusRow(
        component="preprocessing",
        state=state,
        expected_count=1,
        present_count=1 if features.is_file() else 0,
        detail=ArtifactFileName.FEATURES,
    )


def _checkpoint_row(context: ApplicationContext, protocol: FabridConfig) -> ComponentStatusRow:
    present = _checkpoint_count(context)
    expected = len(protocol.seeds)
    return ComponentStatusRow(
        component="checkpoints",
        state=_checkpoint_state(present, expected),
        expected_count=expected,
        present_count=present,
        detail="per-seed checkpoints",
    )


def _experiment_rows(
    application: ApplicationConfig, paths: ArtifactPaths
) -> tuple[ComponentStatusRow, ...]:
    rows: list[ComponentStatusRow] = []
    for experiment_id in ExperimentId:
        if not application.experiments.enabled(experiment_id):
            continue
        verification = verify_experiment(application, paths, experiment_id)
        rows.append(
            ComponentStatusRow(
                component=experiment_id.value,
                state=verification.state,
                expected_count=verification.expected_cells,
                present_count=verification.valid_cells,
                detail="cells verified",
                issues=verification.issues,
            )
        )
    return tuple(rows)


def _analysis_rows(paths: ArtifactPaths) -> tuple[ComponentStatusRow, ...]:
    rows: list[ComponentStatusRow] = []
    for name in (
        CampaignAnalysisId.PRIMARY_INFERENCE,
        CampaignAnalysisId.PRACTICAL_GATES,
    ):
        present = paths.campaign_analysis_path(name).is_file()
        rows.append(
            ComponentStatusRow(
                component=name.value,
                state=ComponentState.VERIFIED if present else ComponentState.MISSING,
                expected_count=1,
                present_count=1 if present else 0,
                detail=f"{name.value}.json",
            )
        )
    return tuple(rows)


def _report_row(paths: ArtifactPaths) -> ComponentStatusRow:
    report_present = paths.report_path().is_file()
    return ComponentStatusRow(
        component="report",
        state=ComponentState.VERIFIED if report_present else ComponentState.MISSING,
        expected_count=1,
        present_count=1 if report_present else 0,
        detail=ArtifactFileName.REPORT,
    )


def campaign_evidence_currency(
    application: ApplicationConfig,
    paths: ArtifactPaths,
) -> tuple[bool, tuple[EnvironmentText, ...]]:
    stale = tuple(
        f"{verification.experiment_id.value}={verification.state.value}"
        for verification in verify_campaign_evidence(application, paths)
        if verification.state is not ExperimentState.PASSED
    )
    bundle_ok, bundle_issues = verify_results_bundle(paths)
    return (not stale and bundle_ok), stale + bundle_issues


def _bundle_row(application: ApplicationConfig, paths: ArtifactPaths) -> ComponentStatusRow:
    bundle_ok, bundle_issues = campaign_evidence_currency(application, paths)
    return ComponentStatusRow(
        component="results bundle",
        state=_bundle_state(bundle_ok, paths.results_root.is_dir()),
        expected_count=1,
        present_count=1 if paths.publication().manifest_path().is_file() else 0,
        detail=ArtifactFileName.MANIFEST,
        issues=bundle_issues,
    )


def inspect_campaign_status(context: ApplicationContext) -> tuple[ComponentStatusRow, ...]:
    application = context.config
    paths = context.paths
    protocol = application.protocol
    return (
        _protocol_row(application, context),
        _preprocessing_row(paths, protocol),
        _checkpoint_row(context, protocol),
        *_experiment_rows(application, paths),
        *_analysis_rows(paths),
        _report_row(paths),
        _bundle_row(application, paths),
    )
