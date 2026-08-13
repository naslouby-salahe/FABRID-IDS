from __future__ import annotations

import re
from dataclasses import dataclass

_IDENTIFIER_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{7,40}$")


def _require_text(name: str, value: str) -> None:
    if not value.strip():
        raise ValueError(f"{name} must not be empty")


def _require_identifier(name: str, value: str) -> None:
    if not _IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError(f"invalid {name}: {value!r}")


@dataclass(frozen=True, slots=True)
class ClientId:
    value: str

    def __post_init__(self) -> None:
        _require_text("client id", self.value)

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class AttackSubtypeId:
    value: str

    def __post_init__(self) -> None:
        _require_text("attack subtype id", self.value)

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class ColumnName:
    value: str

    def __post_init__(self) -> None:
        _require_text("column name", self.value)

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class SourceLabel:
    value: str

    def __post_init__(self) -> None:
        _require_text("source label", self.value)

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class SampleId:
    value: str

    def __post_init__(self) -> None:
        _require_text("sample id", self.value)

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class SourceFileId:
    value: str

    def __post_init__(self) -> None:
        _require_text("source file id", self.value)

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class FailureReason:
    value: str

    def __post_init__(self) -> None:
        _require_text("failure reason", self.value)

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class CampaignId:
    value: str

    def __post_init__(self) -> None:
        _require_identifier("campaign id", self.value)

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class ArtifactName:
    value: str

    def __post_init__(self) -> None:
        _require_identifier("artifact name", self.value)

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class ArtifactDigest:
    value: str

    def __post_init__(self) -> None:
        if not _SHA256_PATTERN.fullmatch(self.value):
            raise ValueError("artifact digest must be a lowercase SHA-256 hex digest")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class GitCommit:
    value: str

    def __post_init__(self) -> None:
        if not _GIT_SHA_PATTERN.fullmatch(self.value):
            raise ValueError("git commit must be a hexadecimal commit id")

    def __str__(self) -> str:
        return self.value
