from __future__ import annotations

import re
from dataclasses import dataclass

_IDENTIFIER_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{7,40}$")


@dataclass(frozen=True, slots=True)
class ClientId:
    value: str

    def __post_init__(self) -> None:
        if not self.value.strip():
            raise ValueError("client id must not be empty")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class AttackSubtypeId:
    value: str

    def __post_init__(self) -> None:
        if not self.value.strip():
            raise ValueError("attack subtype id must not be empty")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class SampleId:
    value: str

    def __post_init__(self) -> None:
        if not self.value.strip():
            raise ValueError("sample id must not be empty")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class BudgetId:
    value: str

    def __post_init__(self) -> None:
        if not _IDENTIFIER_PATTERN.fullmatch(self.value):
            raise ValueError(f"invalid budget id: {self.value!r}")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class CampaignId:
    value: str

    def __post_init__(self) -> None:
        if not _IDENTIFIER_PATTERN.fullmatch(self.value):
            raise ValueError(f"invalid campaign id: {self.value!r}")

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
