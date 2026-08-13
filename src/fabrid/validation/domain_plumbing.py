from __future__ import annotations

import ast
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from fabrid.domain.identifiers import FailureReason
from fabrid.validation.architecture import SourceLine


class DomainPlumbingFindingKind(StrEnum):
    FORBIDDEN_NAMESPACE_IMPORT = "forbidden_namespace_import"
    DICTIONARY_ANNOTATION = "dictionary_annotation"
    PRIMITIVE_DATACLASS_FIELD = "primitive_dataclass_field"


@dataclass(frozen=True, slots=True)
class DomainPlumbingFinding:
    kind: DomainPlumbingFindingKind
    path: Path
    line: SourceLine
    detail: FailureReason


@dataclass(frozen=True, slots=True)
class DomainPlumbingAudit:
    findings: tuple[DomainPlumbingFinding, ...]

    @property
    def passes(self) -> bool:
        return not self.findings


_FORBIDDEN_NAMESPACES = tuple(
    f"fabrid.{name}"
    for name in (
        "audit",
        "config",
        "data",
        "experiments",
        "frontier",
        "optimization",
        "schemas",
        "scoring",
        "statistics",
    )
)
_PRIMITIVES = frozenset(("bool", "float", "int", "str"))


def _name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _name(node.value)
        return None if parent is None else f"{parent}.{node.attr}"
    if isinstance(node, ast.Subscript):
        return _name(node.value)
    return None


def _symbols(annotation: ast.AST | None) -> frozenset[str]:
    if annotation is None:
        return frozenset()
    return frozenset(
        name.split(".")[-1]
        for node in ast.walk(annotation)
        if (name := _name(node)) is not None
    )


def _is_dataclass(node: ast.ClassDef) -> bool:
    return any(
        _name(decorator.func if isinstance(decorator, ast.Call) else decorator)
        == "dataclass"
        for decorator in node.decorator_list
    )


def _namespace_forbidden(module: str) -> bool:
    return any(
        module == namespace or module.startswith(f"{namespace}.")
        for namespace in _FORBIDDEN_NAMESPACES
    )


def _finding(
    kind: DomainPlumbingFindingKind,
    path: Path,
    node: ast.AST,
    detail: str,
) -> DomainPlumbingFinding:
    return DomainPlumbingFinding(
        kind=kind,
        path=path,
        line=SourceLine(node.lineno),  # type: ignore[attr-defined]
        detail=FailureReason(detail),
    )


def _file_findings(path: Path) -> tuple[DomainPlumbingFinding, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    findings: list[DomainPlumbingFinding] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and _namespace_forbidden(node.module or ""):
            findings.append(
                _finding(
                    DomainPlumbingFindingKind.FORBIDDEN_NAMESPACE_IMPORT,
                    path,
                    node,
                    f"superseded namespace {node.module}",
                )
            )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if _namespace_forbidden(alias.name):
                    findings.append(
                        _finding(
                            DomainPlumbingFindingKind.FORBIDDEN_NAMESPACE_IMPORT,
                            path,
                            node,
                            f"superseded namespace {alias.name}",
                        )
                    )

        if isinstance(node, ast.AnnAssign) and "dict" in _symbols(node.annotation):
            findings.append(
                _finding(
                    DomainPlumbingFindingKind.DICTIONARY_ANNOTATION,
                    path,
                    node,
                    "dictionary-shaped annotation",
                )
            )

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            annotations = tuple(
                argument.annotation
                for argument in (
                    *node.args.posonlyargs,
                    *node.args.args,
                    *node.args.kwonlyargs,
                )
            ) + (node.returns,)
            if any("dict" in _symbols(annotation) for annotation in annotations):
                findings.append(
                    _finding(
                        DomainPlumbingFindingKind.DICTIONARY_ANNOTATION,
                        path,
                        node,
                        f"function {node.name} exposes dictionary plumbing",
                    )
                )

        if isinstance(node, ast.ClassDef) and _is_dataclass(node):
            fields = tuple(item for item in node.body if isinstance(item, ast.AnnAssign))
            for field in fields:
                target = field.target.id if isinstance(field.target, ast.Name) else ""
                primitive = _symbols(field.annotation) & _PRIMITIVES
                if primitive and not (len(fields) == 1 and target == "value"):
                    findings.append(
                        _finding(
                            DomainPlumbingFindingKind.PRIMITIVE_DATACLASS_FIELD,
                            path,
                            field,
                            f"field {target} uses raw {sorted(primitive)[0]}",
                        )
                    )
    return tuple(findings)


def audit_domain_plumbing(repository_root: Path) -> DomainPlumbingAudit:
    findings: list[DomainPlumbingFinding] = []
    for path in sorted((repository_root / "src" / "fabrid").rglob("*.py")):
        findings.extend(_file_findings(path))
    return DomainPlumbingAudit(tuple(findings))
