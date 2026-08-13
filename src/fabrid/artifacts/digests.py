from __future__ import annotations

import hashlib
from pathlib import Path

from fabrid.domain.identifiers import ArtifactDigest

_CHUNK_SIZE_BYTES = 1 << 20


def digest_file(path: Path) -> ArtifactDigest:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_SIZE_BYTES), b""):
            digest.update(chunk)
    return ArtifactDigest(digest.hexdigest())
