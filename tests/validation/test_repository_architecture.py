from __future__ import annotations

from pathlib import Path

from fabrid.validation.architecture import audit_repository_architecture


def test_repository_satisfies_architecture_contract() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    audit = audit_repository_architecture(repository_root)

    assert audit.findings == (), "\n".join(
        f"{finding.kind.value}: {finding.path}"
        + ("" if finding.line is None else f":{finding.line.value}")
        + f" — {finding.detail}"
        for finding in audit.findings
    )
