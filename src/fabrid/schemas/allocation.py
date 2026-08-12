"""Typed contracts shared by every allocation policy module.

Allocation policies never operate on raw floats/dicts directly at their public
boundary — they consume `ClientUtilityCurve` and produce `Allocation`, both
validated on construction.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from fabrid.evaluation.record_level import ClientId

_FEASIBILITY_TOLERANCE = 1e-12


class AllocationPolicy(StrEnum):
    EQ_FPR = "EQ_FPR"
    EQ_ALERT = "EQ_ALERT"
    GREEDY = "GREEDY"
    FABRID_MACRO = "FABRID_MACRO"
    FABRID_MINIMAX = "FABRID_MINIMAX"
    POOLED_SHARED = "POOLED_SHARED"
    TEST_ORACLE = "TEST_ORACLE"


@dataclass(frozen=True, slots=True)
class ClientUtilityCurve:
    """u_{k,j} for one client across the full candidate target-rate grid.

    `alpha_grid` must be strictly ascending, start at 0.0, and `utility` must
    be the same length, each value in [0, 1] (a mean of subtype TPRs).
    """

    client_id: ClientId
    alpha_grid: tuple[float, ...]
    utility: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.alpha_grid) != len(self.utility):
            raise ValueError(
                f"alpha_grid ({len(self.alpha_grid)}) and utility "
                f"({len(self.utility)}) must have the same length"
            )
        if len(self.alpha_grid) < 1:
            raise ValueError("alpha_grid must contain at least one candidate")
        if self.alpha_grid[0] != 0.0:
            raise ValueError("alpha_grid must start at 0.0")
        if list(self.alpha_grid) != sorted(self.alpha_grid):
            raise ValueError("alpha_grid must be strictly ascending")
        if len(set(self.alpha_grid)) != len(self.alpha_grid):
            raise ValueError("alpha_grid must not contain duplicates")
        if any(not (0.0 <= u <= 1.0) for u in self.utility):
            raise ValueError("utility values must be in [0, 1]")

    def utility_at_index(self, index: int) -> float:
        return self.utility[index]

    def utility_at_alpha(self, alpha: float) -> float:
        index = self.alpha_grid.index(alpha)
        return self.utility[index]


@dataclass(frozen=True, slots=True)
class AllocationDecision:
    client_id: ClientId
    alpha_selected: float

    def __post_init__(self) -> None:
        if self.alpha_selected < 0.0:
            raise ValueError(f"alpha_selected must be non-negative, got {self.alpha_selected}")


@dataclass(frozen=True, slots=True)
class Allocation:
    policy: AllocationPolicy
    decisions: Mapping[ClientId, AllocationDecision]

    def alpha_of(self, client_id: ClientId) -> float:
        return self.decisions[client_id].alpha_selected

    def total_weighted_cost(self, weight: Mapping[ClientId, float]) -> float:
        return sum(weight[c] * decision.alpha_selected for c, decision in self.decisions.items())

    def is_budget_feasible(self, weight: Mapping[ClientId, float], budget: float) -> bool:
        return self.total_weighted_cost(weight) <= budget + _FEASIBILITY_TOLERANCE
