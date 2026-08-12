from __future__ import annotations

import pytest

from fabrid.evaluation.record_level import ClientId
from fabrid.evaluation.secondary_metrics import (
    ClientConfusion,
    balanced_accuracy,
    macro_f1,
    pooled_recall,
)


def test_client_confusion_rejects_negative_counts() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        ClientConfusion(true_positive=-1, false_negative=0, false_positive=0, true_negative=0)


def test_pooled_recall_pools_across_clients() -> None:
    # client A: 9/10 attack rows caught; client B: 1/100 caught.
    # pooled recall weights by row count, not per-client average.
    confusion = {
        ClientId("A"): ClientConfusion(
            true_positive=9, false_negative=1, false_positive=0, true_negative=10
        ),
        ClientId("B"): ClientConfusion(
            true_positive=1, false_negative=99, false_positive=0, true_negative=10
        ),
    }
    assert pooled_recall(confusion) == pytest.approx(10 / 110)


def test_pooled_recall_requires_attack_rows() -> None:
    confusion = {
        ClientId("A"): ClientConfusion(
            true_positive=0, false_negative=0, false_positive=0, true_negative=10
        )
    }
    with pytest.raises(ValueError):
        pooled_recall(confusion)


def test_macro_f1_perfect_detection() -> None:
    confusion = {
        ClientId("A"): ClientConfusion(
            true_positive=10, false_negative=0, false_positive=0, true_negative=10
        )
    }
    assert macro_f1(confusion) == pytest.approx(1.0)


def test_macro_f1_zero_denominator_contributes_zero() -> None:
    confusion = {
        ClientId("A"): ClientConfusion(
            true_positive=0, false_negative=5, false_positive=0, true_negative=10
        )
    }
    assert macro_f1(confusion) == pytest.approx(0.0)


def test_balanced_accuracy_perfect_detection() -> None:
    confusion = {
        ClientId("A"): ClientConfusion(
            true_positive=10, false_negative=0, false_positive=0, true_negative=10
        )
    }
    assert balanced_accuracy(confusion) == pytest.approx(1.0)


def test_balanced_accuracy_averages_sensitivity_and_specificity() -> None:
    # sensitivity 0.5 (5/10), specificity 1.0 (10/10) -> balanced accuracy 0.75.
    confusion = {
        ClientId("A"): ClientConfusion(
            true_positive=5, false_negative=5, false_positive=0, true_negative=10
        )
    }
    assert balanced_accuracy(confusion) == pytest.approx(0.75)


def test_balanced_accuracy_requires_both_classes_present() -> None:
    confusion = {
        ClientId("A"): ClientConfusion(
            true_positive=0, false_negative=0, false_positive=0, true_negative=10
        )
    }
    with pytest.raises(ValueError):
        balanced_accuracy(confusion)
