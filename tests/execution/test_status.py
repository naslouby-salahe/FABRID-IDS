from __future__ import annotations

from pathlib import Path

from pydantic import TypeAdapter

from fabrid.artifacts.json import digest_file, write_typed_json
from fabrid.artifacts.paths import ArtifactPaths, DetectorCoordinate
from fabrid.config import ApplicationConfig, ArtifactDigest
from fabrid.detector.training import CheckpointMetadata
from fabrid.execution.status import campaign_evidence_currency
from fabrid.reporting.build import (
    ArtifactCounts,
    BundleKind,
    KindCount,
    ManifestEntry,
    ResultsManifest,
)
from tests.support import smoke_application
from tests.validation.test_completion import write_valid_experiment

_DIGEST: ArtifactDigest = "0" * 64


def _write_self_consistent_bundle(paths: ArtifactPaths, application: ApplicationConfig) -> None:
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


def test_campaign_evidence_currency_true_when_evidence_passed_and_bundle_consistent(
    tmp_path: Path,
) -> None:
    application = smoke_application()
    paths = write_valid_experiment(tmp_path, application)
    _write_self_consistent_bundle(paths, application)
    ok, issues = campaign_evidence_currency(application, paths)
    assert ok, issues


def test_campaign_evidence_currency_false_when_experiment_evidence_stale(
    tmp_path: Path,
) -> None:
    application = smoke_application()
    paths = write_valid_experiment(tmp_path, application)
    _write_self_consistent_bundle(paths, application)
    ok, _ = campaign_evidence_currency(application, paths)
    assert ok
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
    ok, issues = campaign_evidence_currency(application, paths)
    assert not ok
    assert any("stale" in issue.lower() or "matched_budget" in issue for issue in issues)


def test_campaign_evidence_currency_false_when_bundle_corrupted(tmp_path: Path) -> None:
    application = smoke_application()
    paths = write_valid_experiment(tmp_path, application)
    _write_self_consistent_bundle(paths, application)
    (paths.publication().bundle_dir() / "file.json").write_text("corrupted", encoding="utf-8")
    ok, issues = campaign_evidence_currency(application, paths)
    assert not ok
    assert issues
