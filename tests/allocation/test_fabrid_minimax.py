from __future__ import annotations

import itertools

import pytest

from fabrid.allocation.fabrid_minimax import allocate_fabrid_minimax
from fabrid.config.protocol import SolverSettings
from fabrid.evaluation.record_level import ClientId
from fabrid.schemas.allocation import AllocationPolicy, ClientUtilityCurve

_SETTINGS = SolverSettings(mip_rel_gap=0.0, time_limit_seconds=10.0, accept_mip_gap_leq=1e-9)
_GRID = (0.0, 0.01, 0.02, 0.03)


def _curve(client: str, utility: tuple[float, ...]) -> ClientUtilityCurve:
    return ClientUtilityCurve(client_id=ClientId(client), alpha_grid=_GRID, utility=utility)


def _brute_force_minimax(
    curves: dict[ClientId, ClientUtilityCurve], weight: dict[ClientId, float], budget: float
) -> float:
    client_ids = sorted(curves.keys())
    best_min_utility = -1.0
    for combo in itertools.product(range(len(_GRID)), repeat=len(client_ids)):
        cost = sum(weight[c] * _GRID[j] for c, j in zip(client_ids, combo, strict=True))
        if cost > budget + 1e-12:
            continue
        min_utility = min(
            curves[c].utility_at_index(j) for c, j in zip(client_ids, combo, strict=True)
        )
        best_min_utility = max(best_min_utility, min_utility)
    return best_min_utility


def test_brute_force_parity_worst_client_objective() -> None:
    curves = {
        ClientId("1"): _curve("1", (0.0, 0.3, 0.55, 0.6)),
        ClientId("2"): _curve("2", (0.0, 0.1, 0.5, 0.52)),
        ClientId("3"): _curve("3", (0.0, 0.4, 0.42, 0.9)),
    }
    weight = {ClientId("1"): 0.4, ClientId("2"): 0.3, ClientId("3"): 0.3}
    budget = 0.015

    brute_force_min_utility = _brute_force_minimax(curves, weight, budget)
    allocation = allocate_fabrid_minimax(curves, weight, budget, _SETTINGS)

    achieved_min = min(curves[c].utility_at_alpha(allocation.alpha_of(c)) for c in curves)
    assert achieved_min == pytest.approx(brute_force_min_utility, abs=1e-6)
    assert allocation.policy is AllocationPolicy.FABRID_MINIMAX
    assert allocation.is_budget_feasible(weight, budget)


def test_determinism_100_repeated_solves() -> None:
    curves = {
        ClientId("1"): _curve("1", (0.0, 0.3, 0.55, 0.6)),
        ClientId("2"): _curve("2", (0.0, 0.1, 0.5, 0.52)),
        ClientId("3"): _curve("3", (0.0, 0.4, 0.42, 0.9)),
    }
    weight = {ClientId("1"): 0.4, ClientId("2"): 0.3, ClientId("3"): 0.3}
    budget = 0.015

    first = allocate_fabrid_minimax(curves, weight, budget, _SETTINGS)
    first_alphas = {c: first.alpha_of(c) for c in curves}
    for _ in range(99):
        allocation = allocate_fabrid_minimax(curves, weight, budget, _SETTINGS)
        assert {c: allocation.alpha_of(c) for c in curves} == first_alphas


def test_zero_budget_allocates_nothing() -> None:
    curves = {ClientId("1"): _curve("1", (0.0, 0.3, 0.55, 0.6))}
    weight = {ClientId("1"): 1.0}
    allocation = allocate_fabrid_minimax(curves, weight, remaining_budget=0.0, settings=_SETTINGS)
    assert allocation.alpha_of(ClientId("1")) == 0.0


def test_protects_the_worse_off_client_relative_to_macro_objective() -> None:
    # client "2" has much lower utility everywhere; minimax should favor it
    # over macro's mean-utility-maximizing choice when budget allows a trade-off.
    curves = {
        ClientId("1"): _curve("1", (0.0, 0.9, 0.92, 0.93)),
        ClientId("2"): _curve("2", (0.0, 0.05, 0.5, 0.55)),
    }
    weight = {ClientId("1"): 0.5, ClientId("2"): 0.5}
    budget = 0.01
    allocation = allocate_fabrid_minimax(curves, weight, budget, _SETTINGS)
    assert allocation.alpha_of(ClientId("2")) >= allocation.alpha_of(ClientId("1"))
