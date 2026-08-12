"""Paired seed-level contrasts between a treatment and baseline policy.

A seed is included only if both policies produced a value at that seed
(a `SOLVER_INVALID` exclusion drops that seed from the contrast, shrinking
`n` below 10 — this must be reported explicitly, never silently padded).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from fabrid.experiments.main_experiment import SeedBudgetResult
from fabrid.schemas.allocation import AllocationPolicy
from fabrid.statistics.bootstrap import BootstrapResult, paired_bootstrap_ci
from fabrid.statistics.sign_flip import SignFlipResult, exact_sign_flip_test


@dataclass(frozen=True, slots=True)
class Contrast:
    treatment: AllocationPolicy
    baseline: AllocationPolicy
    metric_name: str
    paired_differences: tuple[float, ...]
    included_seeds: tuple[int, ...]
    excluded_seeds: tuple[int, ...]
    sign_flip: SignFlipResult
    bootstrap: BootstrapResult


def _paired_differences(
    results: tuple[SeedBudgetResult, ...],
    metric_by_policy: Callable[[SeedBudgetResult], dict[AllocationPolicy, float]],
    treatment: AllocationPolicy,
    baseline: AllocationPolicy,
) -> tuple[tuple[float, ...], tuple[int, ...], tuple[int, ...]]:
    differences: list[float] = []
    included: list[int] = []
    excluded: list[int] = []
    for result in results:
        metrics = metric_by_policy(result)
        if treatment in metrics and baseline in metrics:
            differences.append(metrics[treatment] - metrics[baseline])
            included.append(result.seed)
        else:
            excluded.append(result.seed)
    return tuple(differences), tuple(included), tuple(excluded)


def _build_contrast(
    results: tuple[SeedBudgetResult, ...],
    metric_by_policy: Callable[[SeedBudgetResult], dict[AllocationPolicy, float]],
    metric_name: str,
    treatment: AllocationPolicy,
    baseline: AllocationPolicy,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> Contrast:
    differences, included, excluded = _paired_differences(
        results, metric_by_policy, treatment, baseline
    )
    if not differences:
        raise ValueError(f"no seed has both {treatment} and {baseline} results")
    return Contrast(
        treatment=treatment,
        baseline=baseline,
        metric_name=metric_name,
        paired_differences=differences,
        included_seeds=included,
        excluded_seeds=excluded,
        sign_flip=exact_sign_flip_test(differences),
        bootstrap=paired_bootstrap_ci(
            np.array(differences), resamples=bootstrap_resamples, seed=bootstrap_seed
        ),
    )


def macro_recall_contrast(
    results: tuple[SeedBudgetResult, ...],
    treatment: AllocationPolicy,
    baseline: AllocationPolicy,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> Contrast:
    """Contrast A: e.g. FABRID_MACRO - EQ_FPR on MacroRecall."""
    return _build_contrast(
        results,
        lambda r: r.macro_recall_by_policy,
        "MacroRecall",
        treatment,
        baseline,
        bootstrap_resamples,
        bootstrap_seed,
    )


def worst_client_recall_contrast(
    results: tuple[SeedBudgetResult, ...],
    treatment: AllocationPolicy,
    baseline: AllocationPolicy,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> Contrast:
    """Contrast B: e.g. FABRID_MINIMAX - EQ_FPR on WorstClientRecall."""
    return _build_contrast(
        results,
        lambda r: r.worst_client_recall_by_policy,
        "WorstClientRecall",
        treatment,
        baseline,
        bootstrap_resamples,
        bootstrap_seed,
    )
