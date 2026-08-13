from __future__ import annotations

import ast
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class ArchitectureFindingKind(StrEnum):
    FORBIDDEN_PACKAGE = "forbidden_package"
    FORBIDDEN_ROOT_DIRECTORY = "forbidden_root_directory"
    FORBIDDEN_IMPORT = "forbidden_import"
    FORBIDDEN_ANNOTATION = "forbidden_annotation"
    PICKLE_PERSISTENCE = "pickle_persistence"
    LOWER_LAYER_PIPELINE_IMPORT = "lower_layer_pipeline_import"
    STALE_ALLOCATION_IMPORT = "stale_allocation_import"


@dataclass(frozen=True, slots=True)
class SourceLine:
    value: int

    def __post_init__(self) -> None:
        if self.value < 1:
            raise ValueError("source line must be positive")


@dataclass(frozen=True, slots=True)
class ArchitectureFinding:
    kind: ArchitectureFindingKind
    path: Path
    line: SourceLine | None
    detail: str

    def __post_init__(self) -> None:
        if not self.detail.strip():
            raise ValueError("architecture finding detail must not be empty")


@dataclass(frozen=True, slots=True)
class ArchitectureAudit:
    findings: tuple[ArchitectureFinding, ...]

    @property
    def passes(self) -> bool:
        return not self.findings


_FORBIDDEN_PACKAGES = (
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
_FORBIDDEN_ROOT_DIRECTORIES = (
    "scripts",
    "download_datasets",
)
_FORBIDDEN_TYPING_SYMBOLS = (
    "Any",
    "NewType",
    "Mapping",
    "MutableMapping",
)
_LOWER_LAYERS = (
    "domain",
    "protocol",
    "datasets",
    "detector",
    "calibration",
    "allocation",
    "evaluation",
    "analysis",
    "artifacts",
    "validation",
)
_STALE_ALLOCATION_MODULES = (
    "fabrid.allocation.equal_alert",
    "fabrid.allocation.equal_fpr",
    "fabrid.allocation.fabrid_macro",
    "fabrid.allocation.fabrid_minimax",
    "fabrid.allocation.greedy",
    "fabrid.allocation.pooled_shared",
    "fabrid.allocation.test_oracle",
)


def _dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted_name(node.value)
        if parent is None:
            return None
        return f"{parent}.{node.attr}"
    if isinstance(node, ast.Subscript):
        return _dotted_name(node.value)
    return None


def _annotation_symbols(annotation: ast.AST | None) -> tuple[str, ...]:
    if annotation is None:
        return ()
    return tuple(
        name
        for node in ast.walk(annotation)
        if (name := _dotted_name(node)) is not None
    )


def _module_layer(path: Path, source_root: Path) -> str | None:
    try:
        relative = path.relative_to(source_root / "fabrid")
    except ValueError:
        return None
    if len(relative.parts) < 2:
        return None
    return relative.parts[0]


def _python_findings(path: Path, source_root: Path) -> tuple[ArchitectureFinding, ...]:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))
    findings: list[ArchitectureFinding] = []
    layer = _module_layer(path, source_root)

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module in {"typing", "collections.abc"}:
                for alias in node.names:
                    if alias.name in _FORBIDDEN_TYPING_SYMBOLS:
                        findings.append(
                            ArchitectureFinding(
                                kind=ArchitectureFindingKind.FORBIDDEN_IMPORT,
                                path=path,
                                line=SourceLine(node.lineno),
                                detail=f"forbidden typing symbol {alias.name}",
                            )
                        )
            if layer in _LOWER_LAYERS and module.startswith("fabrid.pipeline"):
                findings.append(
                    ArchitectureFinding(
                        kind=ArchitectureFindingKind.LOWER_LAYER_PIPELINE_IMPORT,
                        path=path,
                        line=SourceLine(node.lineno),
                        detail=f"lower layer {layer} imports pipeline module {module}",
                    )
                )
            if module in _STALE_ALLOCATION_MODULES:
                findings.append(
                    ArchitectureFinding(
                        kind=ArchitectureFindingKind.STALE_ALLOCATION_IMPORT,
                        path=path,
                        line=SourceLine(node.lineno),
                        detail=f"stale allocation ownership import {module}",
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
            for annotation in annotations:
                symbols = _annotation_symbols(annotation)
                forbidden = tuple(
                    symbol
                    for symbol in symbols
                    if symbol.split(".")[-1] in _FORBIDDEN_TYPING_SYMBOLS
                )
                for symbol in forbidden:
                    findings.append(
                        ArchitectureFinding(
                            kind=ArchitectureFindingKind.FORBIDDEN_ANNOTATION,
                            path=path,
                            line=SourceLine(node.lineno),
                            detail=f"forbidden API annotation {symbol}",
                        )
                    )

        if isinstance(node, ast.Call):
            called = _dotted_name(node.func)
            if called in {"pickle.dump", "pickle.dumps", "pickle.load", "pickle.loads"}:
                findings.append(
                    ArchitectureFinding(
                        kind=ArchitectureFindingKind.PICKLE_PERSISTENCE,
                        path=path,
                        line=SourceLine(node.lineno),
                        detail=f"pickle persistence call {called}",
                    )
                )

    return tuple(findings)


def audit_repository_architecture(repository_root: Path) -> ArchitectureAudit:
    source_root = repository_root / "src"
    package_root = source_root / "fabrid"
    findings: list[ArchitectureFinding] = []

    for package in _FORBIDDEN_PACKAGES:
        path = package_root / package
        if path.exists():
            findings.append(
                ArchitectureFinding(
                    kind=ArchitectureFindingKind.FORBIDDEN_PACKAGE,
                    path=path,
                    line=None,
                    detail=f"superseded package {package} must not exist",
                )
            )

    for directory in _FORBIDDEN_ROOT_DIRECTORIES:
        path = repository_root / directory
        if path.exists():
            findings.append(
                ArchitectureFinding(
                    kind=ArchitectureFindingKind.FORBIDDEN_ROOT_DIRECTORY,
                    path=path,
                    line=None,
                    detail=f"root utility directory {directory} must not exist",
                )
            )

    for path in sorted((repository_root / "src").rglob("*.py")):
        findings.extend(_python_findings(path, source_root))
    for path in sorted((repository_root / "tests").rglob("*.py")):
        findings.extend(_python_findings(path, source_root))

    return ArchitectureAudit(tuple(findings))
