"""Secondary detection-quality metrics (roadmap section 56): pooled recall, Macro-F1,
balanced accuracy. AUROC/AUPRC describe the frozen detector's score ranking and are computed
once from raw scores (`scoring/score_contract.py`, `scoring/frontier_inputs.py`) rather than
here, since they never change with the chosen threshold/policy.

Every function here takes already-aggregated per-client confusion counts, matching
`record_level.py`'s convention of consuming typed results rather than raw scores.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from fabrid.evaluation.record_level import ClientId


@dataclass(frozen=True, slots=True)
class ClientConfusion:
    """A client's confusion counts, pooled across every eligible attack subtype for the
    true/false positive counts (attack rows) and the client's single benign-test confusion for
    the true/false negative counts.
    """

    true_positive: int
    false_negative: int
    false_positive: int
    true_negative: int

    def __post_init__(self) -> None:
        if (
            min(self.true_positive, self.false_negative, self.false_positive, self.true_negative)
            < 0
        ):
            raise ValueError(
                "confusion counts must be non-negative, got "
                f"tp={self.true_positive}, fn={self.false_negative}, "
                f"fp={self.false_positive}, tn={self.true_negative}"
            )


def pooled_recall(client_confusion: Mapping[ClientId, ClientConfusion]) -> float:
    """Recall pooled over every attack row across every client (not client-averaged), i.e.
    `sum(TP) / (sum(TP) + sum(FN))`. Contrasts with `MacroRecall`, which averages per-client
    recall and so weights every client equally regardless of its row count.
    """
    if not client_confusion:
        raise ValueError("pooled_recall requires at least one client")
    total_tp = sum(c.true_positive for c in client_confusion.values())
    total_fn = sum(c.false_negative for c in client_confusion.values())
    denominator = total_tp + total_fn
    if denominator == 0:
        raise ValueError("pooled_recall requires at least one attack row across all clients")
    return total_tp / denominator


def _client_f1(confusion: ClientConfusion) -> float:
    precision_denominator = confusion.true_positive + confusion.false_positive
    recall_denominator = confusion.true_positive + confusion.false_negative
    if precision_denominator == 0 or recall_denominator == 0:
        return 0.0
    precision = confusion.true_positive / precision_denominator
    recall = confusion.true_positive / recall_denominator
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def macro_f1(client_confusion: Mapping[ClientId, ClientConfusion]) -> float:
    """Mean of per-client F1 scores; a client with no true positives (via zero precision or
    zero recall denominator) contributes 0.0 rather than an undefined value.
    """
    if not client_confusion:
        raise ValueError("macro_f1 requires at least one client")
    return sum(_client_f1(c) for c in client_confusion.values()) / len(client_confusion)


def _client_balanced_accuracy(confusion: ClientConfusion) -> float:
    tpr_denominator = confusion.true_positive + confusion.false_negative
    tnr_denominator = confusion.true_negative + confusion.false_positive
    if tpr_denominator == 0 or tnr_denominator == 0:
        raise ValueError(
            "balanced accuracy requires at least one attack row and one benign row per client"
        )
    true_positive_rate = confusion.true_positive / tpr_denominator
    true_negative_rate = confusion.true_negative / tnr_denominator
    return (true_positive_rate + true_negative_rate) / 2


def balanced_accuracy(client_confusion: Mapping[ClientId, ClientConfusion]) -> float:
    """Mean of per-client `(TPR + TNR) / 2`."""
    if not client_confusion:
        raise ValueError("balanced_accuracy requires at least one client")
    return sum(_client_balanced_accuracy(c) for c in client_confusion.values()) / len(
        client_confusion
    )
