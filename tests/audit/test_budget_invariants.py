from __future__ import annotations

import pytest

from fabrid.audit.budget_invariants import (
    BudgetInvariantError,
    assert_budget_feasible,
    assert_one_target_per_client,
)
from fabrid.evaluation.record_level import ClientId
from fabrid.schemas.allocation import Allocation, AllocationDecision, AllocationPolicy


def _allocation(alphas: dict[str, float]) -> Allocation:
    decisions = {
        ClientId(c): AllocationDecision(client_id=ClientId(c), alpha_selected=a)
        for c, a in alphas.items()
    }
    return Allocation(policy=AllocationPolicy.EQ_FPR, decisions=decisions)


def test_feasible_budget_passes() -> None:
    allocation = _allocation({"1": 0.01, "2": 0.01})
    weight = {ClientId("1"): 0.5, ClientId("2"): 0.5}
    assert_budget_feasible(allocation, weight, budget=0.01)


def test_infeasible_budget_raises() -> None:
    allocation = _allocation({"1": 0.05, "2": 0.05})
    weight = {ClientId("1"): 0.5, ClientId("2"): 0.5}
    with pytest.raises(BudgetInvariantError):
        assert_budget_feasible(allocation, weight, budget=0.01)


def test_complete_coverage_passes() -> None:
    allocation = _allocation({"1": 0.01, "2": 0.01})
    assert_one_target_per_client(allocation, [ClientId("1"), ClientId("2")])


def test_missing_client_raises() -> None:
    allocation = _allocation({"1": 0.01})
    with pytest.raises(BudgetInvariantError):
        assert_one_target_per_client(allocation, [ClientId("1"), ClientId("2")])


def test_extra_client_raises() -> None:
    allocation = _allocation({"1": 0.01, "2": 0.01})
    with pytest.raises(BudgetInvariantError):
        assert_one_target_per_client(allocation, [ClientId("1")])
