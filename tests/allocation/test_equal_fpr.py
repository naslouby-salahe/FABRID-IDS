from __future__ import annotations

import pytest

from fabrid.allocation.equal_fpr import allocate_equal_fpr
from fabrid.evaluation.record_level import ClientId
from fabrid.schemas.allocation import AllocationPolicy


def test_every_client_gets_the_budget_rate() -> None:
    clients = [ClientId(str(i)) for i in range(9)]
    allocation = allocate_equal_fpr(clients, budget=0.01, alpha_max=0.05)
    assert allocation.policy is AllocationPolicy.EQ_FPR
    for client in clients:
        assert allocation.alpha_of(client) == pytest.approx(0.01)


def test_budget_above_alpha_max_rejected() -> None:
    with pytest.raises(ValueError):
        allocate_equal_fpr([ClientId("1")], budget=0.06, alpha_max=0.05)


def test_negative_budget_rejected() -> None:
    with pytest.raises(ValueError):
        allocate_equal_fpr([ClientId("1")], budget=-0.01, alpha_max=0.05)


def test_empty_client_list_rejected() -> None:
    with pytest.raises(ValueError):
        allocate_equal_fpr([], budget=0.01, alpha_max=0.05)


def test_equal_weighting_implies_mean_equals_budget() -> None:
    clients = [ClientId(str(i)) for i in range(9)]
    allocation = allocate_equal_fpr(clients, budget=0.02, alpha_max=0.05)
    weight = {c: 1 / 9 for c in clients}
    assert allocation.total_weighted_cost(weight) == pytest.approx(0.02)
    assert allocation.is_budget_feasible(weight, budget=0.02)
