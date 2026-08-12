from __future__ import annotations

import pytest

from fabrid.config.protocol import UtilityEligibilityGuardrails
from fabrid.data.eligibility import eligible_subtypes, fallback_rate, is_client_eligible
from fabrid.evaluation.record_level import AttackSubtype

_GUARDRAILS = UtilityEligibilityGuardrails(
    min_attack_validation_rows=200, min_eligible_subtypes=2, min_rows_per_eligible_subtype=50
)


def test_eligible_client_passes() -> None:
    counts = {AttackSubtype("scan"): 100, AttackSubtype("udp"): 100}
    assert is_client_eligible(counts, _GUARDRAILS)


def test_ineligible_total_rows_fails() -> None:
    counts = {AttackSubtype("scan"): 60, AttackSubtype("udp"): 60}
    assert not is_client_eligible(counts, _GUARDRAILS)


def test_ineligible_subtype_count_fails() -> None:
    # single subtype with plenty of rows still fails: need >= 2 eligible subtypes.
    counts = {AttackSubtype("scan"): 300}
    assert not is_client_eligible(counts, _GUARDRAILS)


def test_subtype_below_row_floor_is_not_eligible() -> None:
    counts = {AttackSubtype("scan"): 100, AttackSubtype("udp"): 49, AttackSubtype("tcp"): 100}
    eligible = eligible_subtypes(counts, _GUARDRAILS)
    assert AttackSubtype("udp") not in eligible
    assert AttackSubtype("scan") in eligible
    assert AttackSubtype("tcp") in eligible
    assert is_client_eligible(counts, _GUARDRAILS)


def test_empty_counts_ineligible() -> None:
    assert not is_client_eligible({}, _GUARDRAILS)


def test_fallback_rate() -> None:
    assert fallback_rate(eligible_client_count=7, total_client_count=9) == pytest.approx(2 / 9)
    assert fallback_rate(eligible_client_count=9, total_client_count=9) == 0.0
    assert fallback_rate(eligible_client_count=0, total_client_count=9) == 1.0


def test_fallback_rate_rejects_invalid_counts() -> None:
    with pytest.raises(ValueError):
        fallback_rate(eligible_client_count=10, total_client_count=9)
    with pytest.raises(ValueError):
        fallback_rate(eligible_client_count=0, total_client_count=0)
