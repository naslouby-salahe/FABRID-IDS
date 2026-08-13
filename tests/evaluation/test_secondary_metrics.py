from __future__ import annotations

import pytest

from fabrid.domain.identifiers import ClientId
from fabrid.domain.values import BalancedAccuracy, MacroF1, PooledRecall, RowCount
from fabrid.evaluation.results import ConfusionCounts
from fabrid.evaluation.secondary_metrics import (
    ClientConfusion,
    FederationConfusions,
    balanced_accuracy,
    macro_f1,
    pooled_recall,
)


def _client(
    client_id: str,
    true_positive: int,
    false_negative: int,
    false_positive: int,
    true_negative: int,
) -> ClientConfusion:
    return ClientConfusion(
        client_id=ClientId(client_id),
        counts=ConfusionCounts(
            true_positive=RowCount(true_positive),
            false_negative=RowCount(false_negative),
            false_positive=RowCount(false_positive),
            true_negative=RowCount(true_negative),
        ),
    )


def test_pooled_recall_weights_by_attack_rows() -> None:
    confusions = FederationConfusions(
        (
            _client("A", 9, 1, 0, 10),
            _client("B", 1, 99, 0, 10),
        )
    )

    assert pooled_recall(confusions).value == pytest.approx(10 / 110)


def test_pooled_recall_requires_attack_rows() -> None:
    with pytest.raises(ValueError):
        pooled_recall(FederationConfusions((_client("A", 0, 0, 0, 10),)))


def test_macro_f1_and_balanced_accuracy_return_semantic_values() -> None:
    perfect = FederationConfusions((_client("A", 10, 0, 0, 10),))
    mixed = FederationConfusions((_client("A", 5, 5, 0, 10),))

    assert macro_f1(perfect) == MacroF1(1.0)
    assert balanced_accuracy(perfect) == BalancedAccuracy(1.0)
    assert balanced_accuracy(mixed) == BalancedAccuracy(0.75)


def test_macro_f1_zero_precision_or_recall_contributes_zero() -> None:
    confusions = FederationConfusions((_client("A", 0, 5, 0, 10),))

    assert macro_f1(confusions) == MacroF1(0.0)


def test_balanced_accuracy_requires_both_classes_per_client() -> None:
    with pytest.raises(ValueError):
        balanced_accuracy(FederationConfusions((_client("A", 0, 0, 0, 10),)))


def test_federation_confusions_require_unique_nonempty_clients() -> None:
    with pytest.raises(ValueError):
        FederationConfusions(())
    with pytest.raises(ValueError):
        FederationConfusions((_client("A", 1, 0, 0, 1), _client("A", 1, 0, 0, 1)))


def test_secondary_metric_value_objects_reject_out_of_range_values() -> None:
    with pytest.raises(ValueError):
        PooledRecall(1.1)
    with pytest.raises(ValueError):
        MacroF1(-0.1)
    with pytest.raises(ValueError):
        BalancedAccuracy(1.1)
