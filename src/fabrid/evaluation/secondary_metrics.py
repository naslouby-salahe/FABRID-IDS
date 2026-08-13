from __future__ import annotations

from dataclasses import dataclass

from fabrid.domain.identifiers import ClientId
from fabrid.domain.values import BalancedAccuracy, MacroF1, PooledRecall
from fabrid.evaluation.results import ConfusionCounts


@dataclass(frozen=True, slots=True)
class ClientConfusion:
    client_id: ClientId
    counts: ConfusionCounts


@dataclass(frozen=True, slots=True)
class FederationConfusions:
    clients: tuple[ClientConfusion, ...]

    def __post_init__(self) -> None:
        if not self.clients:
            raise ValueError("federation confusions require at least one client")
        client_ids = tuple(client.client_id for client in self.clients)
        if len(set(client_ids)) != len(client_ids):
            raise ValueError("federation confusions contain duplicate clients")


def pooled_recall(confusions: FederationConfusions) -> PooledRecall:
    total_true_positive = sum(
        client.counts.true_positive.value for client in confusions.clients
    )
    total_false_negative = sum(
        client.counts.false_negative.value for client in confusions.clients
    )
    denominator = total_true_positive + total_false_negative
    if denominator == 0:
        raise ValueError("pooled recall requires at least one attack row")
    return PooledRecall(total_true_positive / denominator)


def _client_f1(confusion: ConfusionCounts) -> MacroF1:
    precision_denominator = (
        confusion.true_positive.value + confusion.false_positive.value
    )
    recall_denominator = (
        confusion.true_positive.value + confusion.false_negative.value
    )
    if precision_denominator == 0 or recall_denominator == 0:
        return MacroF1(0.0)
    precision = confusion.true_positive.value / precision_denominator
    recall = confusion.true_positive.value / recall_denominator
    if precision + recall == 0.0:
        return MacroF1(0.0)
    return MacroF1(2.0 * precision * recall / (precision + recall))


def macro_f1(confusions: FederationConfusions) -> MacroF1:
    return MacroF1(
        sum(_client_f1(client.counts).value for client in confusions.clients)
        / len(confusions.clients)
    )


def _client_balanced_accuracy(confusion: ConfusionCounts) -> BalancedAccuracy:
    attack_denominator = (
        confusion.true_positive.value + confusion.false_negative.value
    )
    benign_denominator = (
        confusion.true_negative.value + confusion.false_positive.value
    )
    if attack_denominator == 0 or benign_denominator == 0:
        raise ValueError(
            "balanced accuracy requires attack and benign rows for every client"
        )
    sensitivity = confusion.true_positive.value / attack_denominator
    specificity = confusion.true_negative.value / benign_denominator
    return BalancedAccuracy((sensitivity + specificity) / 2.0)


def balanced_accuracy(confusions: FederationConfusions) -> BalancedAccuracy:
    return BalancedAccuracy(
        sum(
            _client_balanced_accuracy(client.counts).value
            for client in confusions.clients
        )
        / len(confusions.clients)
    )
