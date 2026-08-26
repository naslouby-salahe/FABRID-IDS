from __future__ import annotations

from pathlib import Path

from fabrid.artifacts.paths import ArtifactPaths
from fabrid.config import (
    ApplicationConfig,
    EventCriterionId,
    FabridConfig,
    GateStatus,
    PathSettings,
    load_application_config,
)
from fabrid.datasets.registry import EventCriterionEvidence, EventProvenanceEvidence
from fabrid.execution.application import ApplicationContext

SMOKE_CONFIG_PATH = Path("tests/fixtures/smoke.yaml")


def production_application() -> ApplicationConfig:
    return load_application_config()


def smoke_application() -> ApplicationConfig:
    return ApplicationConfig.from_yaml(SMOKE_CONFIG_PATH)


def production_protocol() -> FabridConfig:
    return production_application().protocol


def smoke_protocol() -> FabridConfig:
    return smoke_application().protocol


def isolated_paths(tmp_path: Path, *, raw_data_root: Path | None = None) -> ArtifactPaths:
    return ArtifactPaths.from_settings(
        PathSettings(
            raw_data_root=tmp_path / "raw" if raw_data_root is None else raw_data_root,
            preprocessed_root=tmp_path / "preprocessed",
            outputs_root=tmp_path / "outputs",
            results_root=tmp_path / "results",
        )
    )


def isolated_context(
    application: ApplicationConfig,
    tmp_path: Path,
    *,
    raw_data_root: Path | None = None,
) -> ApplicationContext:
    paths = isolated_paths(tmp_path, raw_data_root=raw_data_root)
    isolated = application.model_copy(
        update={
            "paths": PathSettings(
                raw_data_root=paths.raw_data_root,
                preprocessed_root=paths.preprocessed_root,
                outputs_root=paths.outputs_root,
                results_root=paths.results_root,
            )
        }
    )
    return ApplicationContext(config=isolated, paths=paths, repository_root=tmp_path)


def event_evidence(*, failing: EventCriterionId | None = None) -> EventProvenanceEvidence:
    return EventProvenanceEvidence(
        criteria=tuple(
            EventCriterionEvidence(
                criterion=criterion,
                status=(
                    GateStatus.FAIL
                    if failing is not None and criterion is failing
                    else GateStatus.PASS
                ),
                detail="synthetic provenance",
            )
            for criterion in EventCriterionId
        )
    )
