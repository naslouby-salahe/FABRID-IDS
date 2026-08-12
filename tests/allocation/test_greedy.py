from __future__ import annotations

import pytest

from fabrid.allocation.greedy import allocate_greedy
from fabrid.evaluation.record_level import ClientId
from fabrid.schemas.allocation import AllocationPolicy, ClientUtilityCurve

_GRID = (0.0, 0.01, 0.02, 0.03)


def _curve(client: str, utility: tuple[float, ...]) -> ClientUtilityCurve:
    return ClientUtilityCurve(client_id=ClientId(client), alpha_grid=_GRID, utility=utility)


def test_greedy_prefers_higher_efficiency_client() -> None:
    # client "a" gains 0.5 utility per 0.01 alpha step (efficiency 50 at equal weight);
    # client "b" gains 0.1 utility per step (efficiency 10). Budget only affords one step
    # total under equal weight 0.5 each, so "a" must be chosen.
    curves = {
        ClientId("a"): _curve("a", (0.0, 0.5, 0.6, 0.65)),
        ClientId("b"): _curve("b", (0.0, 0.1, 0.15, 0.18)),
    }
    weight = {ClientId("a"): 0.5, ClientId("b"): 0.5}
    allocation = allocate_greedy(curves, weight, budget=0.005, alpha_max=0.05)
    assert allocation.policy is AllocationPolicy.GREEDY
    assert allocation.alpha_of(ClientId("a")) == pytest.approx(0.01)
    assert allocation.alpha_of(ClientId("b")) == pytest.approx(0.0)


def test_greedy_respects_alpha_max_cap() -> None:
    curves = {ClientId("a"): _curve("a", (0.0, 0.5, 0.9, 0.95))}
    weight = {ClientId("a"): 1.0}
    allocation = allocate_greedy(curves, weight, budget=1.0, alpha_max=0.01)
    assert allocation.alpha_of(ClientId("a")) == pytest.approx(0.01)


def test_greedy_zero_budget_allocates_nothing() -> None:
    curves = {ClientId("a"): _curve("a", (0.0, 0.5, 0.9, 0.95))}
    weight = {ClientId("a"): 1.0}
    allocation = allocate_greedy(curves, weight, budget=0.0, alpha_max=0.05)
    assert allocation.alpha_of(ClientId("a")) == 0.0


def test_greedy_never_exceeds_budget() -> None:
    curves = {ClientId(str(i)): _curve(str(i), (0.0, 0.3, 0.5, 0.6)) for i in range(5)}
    weight = {ClientId(str(i)): 0.2 for i in range(5)}
    budget = 0.017
    allocation = allocate_greedy(curves, weight, budget=budget, alpha_max=0.05)
    assert allocation.total_weighted_cost(weight) <= budget + 1e-9


def test_greedy_tie_break_prefers_lower_client_id() -> None:
    # identical curves and weights for both clients -> exact tie on efficiency/delta_u/cost.
    curves = {
        ClientId("2"): _curve("2", (0.0, 0.5, 0.6, 0.65)),
        ClientId("1"): _curve("1", (0.0, 0.5, 0.6, 0.65)),
    }
    weight = {ClientId("1"): 1.0, ClientId("2"): 1.0}
    allocation = allocate_greedy(curves, weight, budget=0.01, alpha_max=0.05)
    assert allocation.alpha_of(ClientId("1")) == pytest.approx(0.01)
    assert allocation.alpha_of(ClientId("2")) == pytest.approx(0.0)


def test_greedy_mismatched_grids_rejected() -> None:
    a = ClientUtilityCurve(ClientId("a"), (0.0, 0.01), (0.0, 0.1))
    b = ClientUtilityCurve(ClientId("b"), (0.0, 0.02), (0.0, 0.1))
    with pytest.raises(ValueError):
        allocate_greedy(
            {ClientId("a"): a, ClientId("b"): b},
            {ClientId("a"): 1.0, ClientId("b"): 1.0},
            0.01,
            0.05,
        )


def test_utility_curve_rejects_non_ascending_grid() -> None:
    with pytest.raises(ValueError):
        ClientUtilityCurve(ClientId("a"), (0.0, 0.02, 0.01), (0.0, 0.1, 0.2))


def test_utility_curve_rejects_grid_not_starting_at_zero() -> None:
    with pytest.raises(ValueError):
        ClientUtilityCurve(ClientId("a"), (0.01, 0.02), (0.1, 0.2))


def test_utility_curve_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError):
        ClientUtilityCurve(ClientId("a"), (0.0, 0.01), (0.1,))
