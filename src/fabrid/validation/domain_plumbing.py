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
