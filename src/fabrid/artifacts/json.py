from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path

from pydantic import TypeAdapter

from fabrid.config import ArtifactDigest, EnvironmentText, FabridConfig

_CHUNK_SIZE = 1 << 20
_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def validated_digest(hex_value: ArtifactDigest) -> ArtifactDigest:
    if not _DIGEST_PATTERN.fullmatch(hex_value):
        raise ValueError("artifact digest must be a lowercase SHA-256 hex digest")
    return hex_value


def digest_file(path: Path) -> ArtifactDigest:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_SIZE):
            digest.update(chunk)
    return validated_digest(digest.hexdigest())


def digest_text(parts: tuple[EnvironmentText, ...]) -> ArtifactDigest:
    digest = hashlib.sha256()
    for part in parts:
        encoded = part.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)
    return validated_digest(digest.hexdigest())


def protocol_config_digest(protocol: FabridConfig) -> ArtifactDigest:
    payload = protocol.model_dump_json(exclude_none=True)
    return digest_text((protocol.protocol_version, payload))


@dataclass(frozen=True, slots=True)
class StoredJsonArtifact:
    digest: ArtifactDigest
    path: Path


def write_typed_json[T](artifact: T, adapter: TypeAdapter[T], path: Path) -> StoredJsonArtifact:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = adapter.dump_json(artifact, indent=2) + b"\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)
    return StoredJsonArtifact(digest=digest_file(path), path=path)


def read_typed_json[T](path: Path, adapter: TypeAdapter[T]) -> T:
    return adapter.validate_json(path.read_bytes())
