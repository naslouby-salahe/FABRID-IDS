from __future__ import annotations

import itertools

import pytest

from fabrid.allocation.fabrid_macro import allocate_fabrid_macro
from fabrid.config.protocol import SolverSettings
from fabrid.evaluation.record_level import ClientId
from fabrid.schemas.allocation import AllocationPolicy, ClientUtilityCurve

_SETTINGS = SolverSettings(mip_rel_gap=0.0, time_limit_seconds=10.0, accept_mip_gap_leq=1e-9)
_GRID = (0.0, 0.01, 0.02, 0.03)


def _curve(client: str, utility: tuple[float, ...]) -> ClientUtilityCurve:
    return ClientUtilityCurve(client_id=ClientId(client), alpha_grid=_GRID, utility=utility)


def _brute_force_macro(
    curves: dict[ClientId, ClientUtilityCurve], weight: dict[ClientId, float], budget: float
) -> tuple[float, dict[ClientId, float]]:
    client_ids = sorted(curves.keys())
    best_mean_utility = -1.0
    best_alphas: dict[ClientId, float] = {}
    for combo in itertools.product(range(len(_GRID)), repeat=len(client_ids)):
        cost = sum(weight[c] * _GRID[j] for c, j in zip(client_ids, combo, strict=True))
        if cost > budget + 1e-12:
            continue
        mean_utility = sum(
            curves[c].utility_at_index(j) for c, j in zip(client_ids, combo, strict=True)
        ) / len(client_ids)
        if mean_utility > best_mean_utility:
            best_mean_utility = mean_utility
            best_alphas = {c: _GRID[j] for c, j in zip(client_ids, combo, strict=True)}
    return best_mean_utility, best_alphas


def test_brute_force_parity_three_clients_four_candidates() -> None:
    curves = {
        ClientId("1"): _curve("1", (0.0, 0.3, 0.55, 0.6)),
        ClientId("2"): _curve("2", (0.0, 0.1, 0.5, 0.52)),
        ClientId("3"): _curve("3", (0.0, 0.4, 0.42, 0.9)),
    }
    weight = {ClientId("1"): 0.4, ClientId("2"): 0.3, ClientId("3"): 0.3}
    budget = 0.015

    brute_force_utility, _ = _brute_force_macro(curves, weight, budget)
    allocation = allocate_fabrid_macro(curves, weight, budget, _SETTINGS)

    achieved_utility = sum(
        curves[c].utility_at_alpha(allocation.alpha_of(c)) for c in curves
    ) / len(curves)
    assert achieved_utility == pytest.approx(brute_force_utility, abs=1e-6)
    assert allocation.is_budget_feasible(weight, budget)


def test_determinism_100_repeated_solves() -> None:
    curves = {
        ClientId("1"): _curve("1", (0.0, 0.3, 0.55, 0.6)),
        ClientId("2"): _curve("2", (0.0, 0.1, 0.5, 0.52)),
        ClientId("3"): _curve("3", (0.0, 0.4, 0.42, 0.9)),
    }
    weight = {ClientId("1"): 0.4, ClientId("2"): 0.3, ClientId("3"): 0.3}
    budget = 0.015

    first = allocate_fabrid_macro(curves, weight, budget, _SETTINGS)
    first_alphas = {c: first.alpha_of(c) for c in curves}
    for _ in range(99):
        allocation = allocate_fabrid_macro(curves, weight, budget, _SETTINGS)
        assert {c: allocation.alpha_of(c) for c in curves} == first_alphas


def test_zero_budget_allocates_nothing() -> None:
    curves = {ClientId("1"): _curve("1", (0.0, 0.3, 0.55, 0.6))}
    weight = {ClientId("1"): 1.0}
    allocation = allocate_fabrid_macro(curves, weight, remaining_budget=0.0, settings=_SETTINGS)
    assert allocation.alpha_of(ClientId("1")) == 0.0


def test_single_client_reduces_to_best_affordable_point() -> None:
    curves = {ClientId("1"): _curve("1", (0.0, 0.3, 0.55, 0.6))}
    weight = {ClientId("1"): 1.0}
    allocation = allocate_fabrid_macro(curves, weight, remaining_budget=0.02, settings=_SETTINGS)
    assert allocation.alpha_of(ClientId("1")) == pytest.approx(0.02)
    assert allocation.policy is AllocationPolicy.FABRID_MACRO


def test_equal_utility_curves_no_unexplained_advantage() -> None:
    curves = {ClientId(str(i)): _curve(str(i), (0.0, 0.3, 0.5, 0.6)) for i in range(4)}
    weight = {ClientId(str(i)): 0.25 for i in range(4)}
    budget = 0.01
    allocation = allocate_fabrid_macro(curves, weight, budget, _SETTINGS)
    achieved_utility = sum(
        curves[c].utility_at_alpha(allocation.alpha_of(c)) for c in curves
    ) / len(curves)
    # equal curves -> best achievable mean utility is the same as a single client's
    # best affordable point under the shared budget (equal allocation is optimal).
    equal_alpha = min(a for a in _GRID if 0.25 * a <= budget + 1e-12)
    expected = curves[ClientId("0")].utility_at_alpha(max(a for a in _GRID if a <= equal_alpha))
    assert achieved_utility >= expected - 1e-9


def test_monotone_budget_feasibility() -> None:
    curves = {
        ClientId("1"): _curve("1", (0.0, 0.3, 0.55, 0.6)),
        ClientId("2"): _curve("2", (0.0, 0.1, 0.5, 0.52)),
    }
    weight = {ClientId("1"): 0.5, ClientId("2"): 0.5}
    budgets = [0.0, 0.005, 0.01, 0.015]
    achieved: list[float] = []
    for budget in budgets:
        allocation = allocate_fabrid_macro(curves, weight, budget, _SETTINGS)
        achieved.append(
            sum(curves[c].utility_at_alpha(allocation.alpha_of(c)) for c in curves) / len(curves)
        )
    for lower, higher in itertools.pairwise(achieved):
        assert higher >= lower - 1e-9
