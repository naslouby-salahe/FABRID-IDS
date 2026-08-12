from __future__ import annotations

import pytest

from fabrid.allocation.equal_alert import allocate_equal_alert
from fabrid.evaluation.record_level import ClientId
from fabrid.schemas.allocation import AllocationPolicy


def test_equal_weights_rejected() -> None:
    weight = {ClientId(str(i)): 1 / 9 for i in range(9)}
    with pytest.raises(ValueError):
        allocate_equal_alert(weight, budget=0.01, alpha_max=0.05)


def test_unequal_weights_respect_budget() -> None:
    weight = {ClientId("big"): 0.7, ClientId("small"): 0.3}
    budget = 0.01
    allocation = allocate_equal_alert(weight, budget=budget, alpha_max=0.05)
    assert allocation.policy is AllocationPolicy.EQ_ALERT
    assert allocation.is_budget_feasible(weight, budget)
    # equal absolute alert *contribution* -> larger-weight client gets a smaller rate.
    assert allocation.alpha_of(ClientId("big")) < allocation.alpha_of(ClientId("small"))


def test_alpha_max_cap_respected() -> None:
    weight = {ClientId("tiny"): 0.01, ClientId("huge"): 0.99}
    allocation = allocate_equal_alert(weight, budget=0.5, alpha_max=0.05)
    for client_id in weight:
        assert allocation.alpha_of(client_id) <= 0.05 + 1e-9


def test_negative_budget_rejected() -> None:
    with pytest.raises(ValueError):
        allocate_equal_alert({ClientId("a"): 0.6, ClientId("b"): 0.4}, budget=-0.1, alpha_max=0.05)


def test_empty_weight_rejected() -> None:
    with pytest.raises(ValueError):
        allocate_equal_alert({}, budget=0.01, alpha_max=0.05)
