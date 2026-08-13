from __future__ import annotations

from dataclasses import dataclass

from pydantic import TypeAdapter

from fabrid.allocation.contracts import Allocation
from fabrid.allocation.solver import SolverEvidence
from fabrid.artifacts.json_store import StoredJsonArtifact, write_typed_json
from fabrid.artifacts.layout import ArtifactLayout
from fabrid.domain.coordinates import AllocationCoordinate


@dataclass(frozen=True, slots=True)
class AllocationArtifact:
    coordinate: AllocationCoordinate
    allocation: Allocation
    solver: SolverEvidence

    def __post_init__(self) -> None:
        if self.coordinate.policy is not self.allocation.policy:
            raise ValueError("allocation artifact policy does not match its coordinate")


_ALLOCATION_ADAPTER = TypeAdapter(AllocationArtifact)


def persist_allocation(
    artifact: AllocationArtifact,
    layout: ArtifactLayout,
) -> StoredJsonArtifact:
    return write_typed_json(
        artifact,
        _ALLOCATION_ADAPTER,
        layout.allocation_path(artifact.coordinate),
    )
