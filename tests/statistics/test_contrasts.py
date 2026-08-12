from __future__ import annotations

import pytest

from fabrid.experiments.main_experiment import SeedBudgetResult
from fabrid.schemas.allocation import AllocationPolicy
from fabrid.statistics.contrasts import macro_recall_contrast, worst_client_recall_contrast


def _result(
    seed: int, macro: dict[AllocationPolicy, float], worst: dict[AllocationPolicy, float]
) -> SeedBudgetResult:
    return SeedBudgetResult(
        seed=seed, budget=0.01, macro_recall_by_policy=macro, worst_client_recall_by_policy=worst
    )


def test_macro_recall_contrast_pairs_available_seeds() -> None:
    results = (
        _result(
            0,
            {AllocationPolicy.EQ_FPR: 0.5, AllocationPolicy.FABRID_MACRO: 0.6},
            {AllocationPolicy.EQ_FPR: 0.4, AllocationPolicy.FABRID_MACRO: 0.45},
        ),
        _result(
            1,
            {AllocationPolicy.EQ_FPR: 0.5, AllocationPolicy.FABRID_MACRO: 0.55},
            {AllocationPolicy.EQ_FPR: 0.4, AllocationPolicy.FABRID_MACRO: 0.42},
        ),
    )
    contrast = macro_recall_contrast(
        results,
        AllocationPolicy.FABRID_MACRO,
        AllocationPolicy.EQ_FPR,
        bootstrap_resamples=1000,
        bootstrap_seed=0,
    )
    assert contrast.paired_differences == pytest.approx((0.1, 0.05))
    assert contrast.included_seeds == (0, 1)
    assert contrast.excluded_seeds == ()
    assert contrast.sign_flip.enumerated_sign_assignments == 4


def test_solver_invalid_seed_excluded_from_contrast() -> None:
    results = (
        _result(
            0,
            {AllocationPolicy.EQ_FPR: 0.5, AllocationPolicy.FABRID_MACRO: 0.6},
            {},
        ),
        _result(1, {AllocationPolicy.EQ_FPR: 0.5}, {}),  # FABRID_MACRO excluded this seed
    )
    contrast = macro_recall_contrast(
        results,
        AllocationPolicy.FABRID_MACRO,
        AllocationPolicy.EQ_FPR,
        bootstrap_resamples=1000,
        bootstrap_seed=0,
    )
    assert contrast.included_seeds == (0,)
    assert contrast.excluded_seeds == (1,)
    assert len(contrast.paired_differences) == 1


def test_worst_client_recall_contrast() -> None:
    results = (
        _result(
            0,
            {},
            {AllocationPolicy.EQ_FPR: 0.3, AllocationPolicy.FABRID_MINIMAX: 0.5},
        ),
    )
    contrast = worst_client_recall_contrast(
        results,
        AllocationPolicy.FABRID_MINIMAX,
        AllocationPolicy.EQ_FPR,
        bootstrap_resamples=1000,
        bootstrap_seed=0,
    )
    assert contrast.metric_name == "WorstClientRecall"
    assert contrast.paired_differences == pytest.approx((0.2,))


def test_no_overlapping_seeds_rejected() -> None:
    results = (_result(0, {AllocationPolicy.EQ_FPR: 0.5}, {}),)
    with pytest.raises(ValueError):
        macro_recall_contrast(
            results,
            AllocationPolicy.FABRID_MACRO,
            AllocationPolicy.EQ_FPR,
            bootstrap_resamples=1000,
            bootstrap_seed=0,
        )
