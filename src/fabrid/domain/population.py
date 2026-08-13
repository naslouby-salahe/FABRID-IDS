from __future__ import annotations

from dataclasses import dataclass

from fabrid.domain.identifiers import ClientId
from fabrid.domain.values import RowCount


@dataclass(frozen=True, slots=True)
class ClientPopulation:
    clients: tuple[ClientId, ...]

    def __post_init__(self) -> None:
        if not self.clients:
            raise ValueError("client population must not be empty")
        if len(set(self.clients)) != len(self.clients):
            raise ValueError("client population contains duplicate clients")

    @property
    def size(self) -> RowCount:
        return RowCount(len(self.clients))
