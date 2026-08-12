from __future__ import annotations

import math

import pytest

from fabrid.evaluation.record_level import (
    AttackSubtype,
    ClientId,
    ClientWeight,
    FalsePositiveRate,
    TruePositiveRate,
    budget_usage_ratio,
    budget_violation_ratio,
    client_macro_recall,
    false_alert_gini,
    federation_fpr,
    federation_macro_recall,
    fpr_dispersion,
    worst_client_recall,
)


def test_client_macro_recall_averages_subtypes_not_rows() -> None:
    subtype_tpr = {
        AttackSubtype("scan"): TruePositiveRate(1.0),
        AttackSubtype("udp"): TruePositiveRate(0.0),
    }
    assert client_macro_recall(subtype_tpr) == pytest.approx(0.5)


def test_client_macro_recall_requires_at_least_one_subtype() -> None:
    with pytest.raises(ValueError):
        client_macro_recall({})


def test_federation_macro_recall_and_worst_client() -> None:
    recall = {
        ClientId("a"): 0.9,
        ClientId("b"): 0.5,
        ClientId("c"): 0.7,
    }
    assert federation_macro_recall(recall) == pytest.approx(0.7)
    assert worst_client_recall(recall) == pytest.approx(0.5)


def test_federation_fpr_equal_client_weighting() -> None:
    fpr = {ClientId(str(i)): FalsePositiveRate(0.01) for i in range(9)}
    weight = {ClientId(str(i)): ClientWeight(1 / 9) for i in range(9)}
    assert federation_fpr(fpr, weight) == pytest.approx(0.01)


def test_federation_fpr_mismatched_client_sets_rejected() -> None:
    fpr = {ClientId("a"): FalsePositiveRate(0.01)}
    weight = {ClientId("b"): ClientWeight(1.0)}
    with pytest.raises(ValueError):
        federation_fpr(fpr, weight)


def test_budget_usage_ratio_not_clamped() -> None:
    assert budget_usage_ratio(0.012, budget=0.01) == pytest.approx(1.2)
    assert budget_usage_ratio(0.008, budget=0.01) == pytest.approx(0.8)


def test_budget_usage_ratio_rejects_nonpositive_budget() -> None:
    with pytest.raises(ValueError):
        budget_usage_ratio(0.01, budget=0.0)


def test_budget_violation_ratio() -> None:
    assert budget_violation_ratio(1.2) == pytest.approx(0.2)
    assert budget_violation_ratio(0.8) == pytest.approx(0.0)
    assert budget_violation_ratio(1.0) == pytest.approx(0.0)


def test_fpr_dispersion_reports_na_as_none_when_mean_zero() -> None:
    fpr = {ClientId(str(i)): FalsePositiveRate(0.0) for i in range(3)}
    dispersion = fpr_dispersion(fpr)
    assert dispersion.coefficient_of_variation is None
    assert dispersion.median == 0.0
    assert dispersion.minimum == 0.0
    assert dispersion.maximum == 0.0


def test_fpr_dispersion_nonzero() -> None:
    fpr = {
        ClientId("a"): FalsePositiveRate(0.01),
        ClientId("b"): FalsePositiveRate(0.02),
        ClientId("c"): FalsePositiveRate(0.03),
        ClientId("d"): FalsePositiveRate(0.04),
    }
    dispersion = fpr_dispersion(fpr)
    assert dispersion.minimum == pytest.approx(0.01)
    assert dispersion.maximum == pytest.approx(0.04)
    assert dispersion.median == pytest.approx(0.025)
    assert dispersion.coefficient_of_variation is not None
    assert dispersion.coefficient_of_variation > 0


def test_false_alert_gini_zero_when_no_false_alerts() -> None:
    counts = {ClientId(str(i)): 0 for i in range(5)}
    assert false_alert_gini(counts) == 0.0


def test_false_alert_gini_zero_when_perfectly_equal() -> None:
    counts = {ClientId(str(i)): 10 for i in range(5)}
    assert false_alert_gini(counts) == pytest.approx(0.0)


def test_false_alert_gini_positive_when_concentrated() -> None:
    counts = {ClientId("a"): 100, ClientId("b"): 0, ClientId("c"): 0}
    gini = false_alert_gini(counts)
    assert gini > 0
    assert not math.isnan(gini)


def test_invalid_rate_values_rejected() -> None:
    with pytest.raises(ValueError):
        TruePositiveRate(1.5)
    with pytest.raises(ValueError):
        FalsePositiveRate(-0.1)
    with pytest.raises(ValueError):
        ClientWeight(-0.1)
