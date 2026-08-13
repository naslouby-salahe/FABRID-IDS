from __future__ import annotations

from fabrid.allocation.contracts import AllocationWeights, ClientBudgetWeight, FederationWeights
from fabrid.domain.identifiers import AttackSubtypeId, ClientId
from fabrid.domain.values import (
    BudgetUsageRatio,
    BudgetViolationRatio,
    ClientWeight,
    FalseAlertCount,
    FalsePositiveBudget,
    FalsePositiveRate,
    MacroRecall,
    Probability,
    TruePositiveRate,
    WorstClientRecall,
)
from fabrid.evaluation.metrics import (
    ClientFalseAlerts,
    ClientFalsePositiveRate,
    ClientMacroRecall,
    SubtypeRecall,
    budget_usage_ratio,
    budget_violation_ratio,
    client_macro_recall,
    false_alert_gini,
    federation_fpr,
    federation_macro_recall,
    fpr_dispersion,
    worst_client_recall,
)


def test_recall_metrics_average_subtypes_and_clients_not_rows() -> None:
    first = ClientMacroRecall(
        ClientId("first"),
        client_macro_recall(
            (
                SubtypeRecall(AttackSubtypeId("scan"), TruePositiveRate(1.0)),
                SubtypeRecall(AttackSubtypeId("udp"), TruePositiveRate(0.0)),
            )
        ),
    )
    second = ClientMacroRecall(ClientId("second"), MacroRecall(0.9))

    assert first.recall == MacroRecall(0.5)
    assert federation_macro_recall((first, second)) == MacroRecall(0.7)
    assert worst_client_recall((first, second)) == WorstClientRecall(0.5)


def test_federation_fpr_uses_typed_federation_weights() -> None:
    first = ClientId("first")
    second = ClientId("second")
    weights = FederationWeights(
        AllocationWeights(
            (
                ClientBudgetWeight(first, ClientWeight(0.25)),
                ClientBudgetWeight(second, ClientWeight(0.75)),
            )
        )
    )

    rate = federation_fpr(
        (
            ClientFalsePositiveRate(first, FalsePositiveRate(0.04)),
            ClientFalsePositiveRate(second, FalsePositiveRate(0.00)),
        ),
        weights,
    )

    assert rate == FalsePositiveRate(0.01)


def test_budget_usage_and_violation_are_not_clamped() -> None:
    usage = budget_usage_ratio(FalsePositiveRate(0.012), FalsePositiveBudget(0.01))

    assert usage == BudgetUsageRatio(1.2)
    assert budget_violation_ratio(usage) == BudgetViolationRatio(0.2)
    assert budget_violation_ratio(BudgetUsageRatio(0.8)) == BudgetViolationRatio(0.0)
    assert budget_usage_ratio(FalsePositiveRate(0.0), FalsePositiveBudget(0.0)) is None


def test_fpr_dispersion_and_false_alert_gini_are_typed() -> None:
    rates = (
        ClientFalsePositiveRate(ClientId("a"), FalsePositiveRate(0.01)),
        ClientFalsePositiveRate(ClientId("b"), FalsePositiveRate(0.02)),
        ClientFalsePositiveRate(ClientId("c"), FalsePositiveRate(0.03)),
        ClientFalsePositiveRate(ClientId("d"), FalsePositiveRate(0.04)),
    )
    dispersion = fpr_dispersion(rates)

    assert dispersion.minimum == FalsePositiveRate(0.01)
    assert dispersion.maximum == FalsePositiveRate(0.04)
    assert dispersion.median == FalsePositiveRate(0.025)
    assert dispersion.coefficient_of_variation is not None

    assert false_alert_gini(
        (
            ClientFalseAlerts(ClientId("a"), FalseAlertCount(100)),
            ClientFalseAlerts(ClientId("b"), FalseAlertCount(0)),
            ClientFalseAlerts(ClientId("c"), FalseAlertCount(0)),
        )
    ).value > Probability(0.0).value
