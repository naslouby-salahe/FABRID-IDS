from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from pydantic import TypeAdapter

from fabrid.artifacts.digests import digest_file
from fabrid.domain.identifiers import ArtifactDigest

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class StoredJsonArtifact:
    digest: ArtifactDigest
    path: Path


def write_typed_json(
    value: T,
    adapter: TypeAdapter[T],
    path: Path,
) -> StoredJsonArtifact:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(adapter.dump_json(value, indent=2) + b"\n")
    return StoredJsonArtifact(digest=digest_file(path), path=path)


def read_typed_json(path: Path, adapter: TypeAdapter[T]) -> T:
    return adapter.validate_json(path.read_bytes())
