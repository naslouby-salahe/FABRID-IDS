from __future__ import annotations

from pathlib import Path

from fabrid.validation.architecture import (
    ArchitectureFindingKind,
    audit_repository_architecture,
)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_architecture_audit_accepts_clean_dependency_direction(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "fabrid" / "domain" / "value.py",
        "from dataclasses import dataclass\n@dataclass(frozen=True, slots=True)\nclass Value:\n    value: int\n",
    )
    _write(
        tmp_path / "src" / "fabrid" / "pipeline" / "run.py",
        "from fabrid.domain.value import Value\ndef run(value: Value) -> Value:\n    return value\n",
    )
    (tmp_path / "tests").mkdir()

    audit = audit_repository_architecture(tmp_path)

    assert audit.findings == ()


def test_architecture_audit_rejects_legacy_packages_and_root_scripts(tmp_path: Path) -> None:
    (tmp_path / "src" / "fabrid" / "schemas").mkdir(parents=True)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "tests").mkdir()

    findings = audit_repository_architecture(tmp_path).findings
    kinds = tuple(finding.kind for finding in findings)

    assert ArchitectureFindingKind.FORBIDDEN_PACKAGE in kinds
    assert ArchitectureFindingKind.FORBIDDEN_ROOT_DIRECTORY in kinds


def test_architecture_audit_rejects_any_mapping_and_lower_layer_pipeline_imports(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "src" / "fabrid" / "allocation" / "bad.py",
        "from typing import Any, Mapping\n"
        "from fabrid.pipeline.context import PipelinePaths\n"
        "def bad(value: Mapping[str, Any]) -> Any:\n"
        "    return value\n",
    )
    (tmp_path / "tests").mkdir()

    findings = audit_repository_architecture(tmp_path).findings
    kinds = tuple(finding.kind for finding in findings)

    assert ArchitectureFindingKind.FORBIDDEN_IMPORT in kinds
    assert ArchitectureFindingKind.FORBIDDEN_ANNOTATION in kinds
    assert ArchitectureFindingKind.LOWER_LAYER_PIPELINE_IMPORT in kinds


def test_architecture_audit_rejects_pickle_and_stale_flat_policy_import(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "fabrid" / "pipeline" / "bad.py",
        "import pickle\n"
        "from fabrid.allocation.greedy import allocate_greedy\n"
        "def persist(value: object) -> bytes:\n"
        "    return pickle.dumps(value)\n",
    )
    (tmp_path / "tests").mkdir()

    findings = audit_repository_architecture(tmp_path).findings
    kinds = tuple(finding.kind for finding in findings)

    assert ArchitectureFindingKind.PICKLE_PERSISTENCE in kinds
    assert ArchitectureFindingKind.STALE_ALLOCATION_IMPORT in kinds
