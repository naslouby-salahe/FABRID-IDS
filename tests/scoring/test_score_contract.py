from __future__ import annotations

import numpy as np
import pytest

from fabrid.scoring.score_contract import assert_auroc_invariant, compute_auroc, decide


def test_decide_is_strict_greater_than() -> None:
    scores = np.array([1.0, 2.0, 2.0, 3.0])
    alerts = decide(scores, threshold=2.0)
    np.testing.assert_array_equal(alerts, [False, False, False, True])


def test_compute_auroc_perfect_separation() -> None:
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    is_attack = np.array([False, False, True, True])
    assert compute_auroc(scores, is_attack) == pytest.approx(1.0)


def test_compute_auroc_chance_level() -> None:
    # positives {1,4}, negatives {2,3}: 2 of 4 positive/negative pairs favor
    # the positive -> AUROC 0.5.
    scores = np.array([1.0, 2.0, 3.0, 4.0])
    is_attack = np.array([True, False, False, True])
    assert compute_auroc(scores, is_attack) == pytest.approx(0.5)


def test_compute_auroc_requires_both_classes() -> None:
    with pytest.raises(ValueError):
        compute_auroc(np.array([1.0, 2.0]), np.array([True, True]))
    with pytest.raises(ValueError):
        compute_auroc(np.array([1.0, 2.0]), np.array([False, False]))


def test_auroc_invariant_holds_across_identical_scores() -> None:
    scores = np.array([0.1, 0.5, 0.6, 0.9])
    is_attack = np.array([False, True, False, True])
    auroc = compute_auroc(scores, is_attack)
    assert_auroc_invariant({"EQ_FPR": auroc, "GREEDY": auroc, "FABRID_MACRO": auroc})


def test_auroc_invariant_detects_violation() -> None:
    with pytest.raises(ValueError):
        assert_auroc_invariant({"EQ_FPR": 0.8, "GREEDY": 0.9})


def test_auroc_invariant_requires_at_least_one_policy() -> None:
    with pytest.raises(ValueError):
        assert_auroc_invariant({})
