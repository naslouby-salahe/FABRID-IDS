from __future__ import annotations

import ast
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class DomainPlumbingFindingKind(StrEnum):
    FORBIDDEN_NAMESPACE_IMPORT = "forbidden_namespace_import"
    DICTIONARY_ANNOTATION = "dictionary_annotation"
    PRIMITIVE_DATACLASS_FIELD = "primitive_dataclass_field"


@dataclass(frozen=True, slots=True)
class DomainPlumbingFinding:
    kind: DomainPlumbingFindingKind
    path: Path
    line: int
    detail: str


@dataclass(frozen=True, slots=True)
class DomainPlumbingAudit:
    findings: tuple[DomainPlumbingFinding, ...]

    @property
    def passes(self) -> bool:
        return not self.findings


_FORBIDDEN_NAMESPACES = tuple(
    f"fabrid.{name}"
    for name in (
        "audit", "config", "data", "experiments", "frontier",
        "optimization", "schemas", "scoring", "statistics",
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
        _name(decorator.func if isinstance(decorator, ast.Call) else decorator) == "dataclass"
        for decorator in node.decorator_list
    )


def _namespace_forbidden(module: str) -> bool:
    return any(
        module == namespace or module.startswith(f"{namespace}.")
        for namespace in _FORBIDDEN_NAMESPACES
    )
