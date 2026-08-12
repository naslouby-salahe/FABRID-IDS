"""Utility-eligibility gate for validation-informed FABRID allocation.

A client is eligible only if it has enough attack-validation rows overall
and enough eligible subtypes (each individually with enough rows). Ineligible
clients fall back to an equal-FPR allocation rather than entering the
optimization problem (see `fabrid.allocation.equal_fpr`).
"""

from __future__ import annotations

from collections.abc import Mapping

from fabrid.config.protocol import UtilityEligibilityGuardrails
from fabrid.evaluation.record_level import AttackSubtype


def eligible_subtypes(
    subtype_row_counts: Mapping[AttackSubtype, int], guardrails: UtilityEligibilityGuardrails
) -> frozenset[AttackSubtype]:
    return frozenset(
        subtype
        for subtype, count in subtype_row_counts.items()
        if count >= guardrails.min_rows_per_eligible_subtype
    )


def is_client_eligible(
    subtype_row_counts: Mapping[AttackSubtype, int], guardrails: UtilityEligibilityGuardrails
) -> bool:
    total_rows = sum(subtype_row_counts.values())
    if total_rows < guardrails.min_attack_validation_rows:
        return False
    return (
        len(eligible_subtypes(subtype_row_counts, guardrails)) >= guardrails.min_eligible_subtypes
    )


def fallback_rate(eligible_client_count: int, total_client_count: int) -> float:
    """FallbackRate = #fallback clients / K."""
    if total_client_count <= 0:
        raise ValueError(f"total_client_count must be positive, got {total_client_count}")
    if not (0 <= eligible_client_count <= total_client_count):
        raise ValueError(
            f"eligible_client_count ({eligible_client_count}) must be in "
            f"[0, total_client_count={total_client_count}]"
        )
    fallback_client_count = total_client_count - eligible_client_count
    return fallback_client_count / total_client_count
