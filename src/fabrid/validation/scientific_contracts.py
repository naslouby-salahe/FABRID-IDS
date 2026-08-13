from __future__ import annotations

import ast
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from fabrid.domain.identifiers import FailureReason
from fabrid.validation.architecture import SourceLine


class ScientificContractFindingKind(StrEnum):
    FORBIDDEN_NAMESPACE = "forbidden_namespace"
    DICTIONARY_ANNOTATION = "dictionary_annotation"
    PRIMITIVE_FIELD = "primitive_field"


@dataclass(frozen=True, slots=True)
class ScientificContractFinding:
    kind: ScientificContractFindingKind
    path: Path
    line: SourceLine
    detail: FailureReason


@dataclass(frozen=True, slots=True)
class ScientificContractAudit:
    findings: tuple[ScientificContractFinding, ...]

    @property
    def passes(self) -> bool:
        return not self.findings


_PRIMITIVES = frozenset(("bool", "float", "int", "str"))
