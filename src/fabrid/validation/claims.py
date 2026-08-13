from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum


class ForbiddenClaimPhrase(StrEnum):
    FIRST_ALERT_BUDGET_IDS = "first alert-budget ids"
    FIRST_FEDERATED_THRESHOLDING_METHOD = "first federated thresholding method"
    FIRST_HETEROGENEOUS_DETECTOR_THRESHOLD_ALLOCATOR = (
        "first heterogeneous detector threshold allocator"
    )
    FIRST_IDS_RESOURCE_ALLOCATOR = "first ids resource allocator"
    NOVEL_MILP_OPTIMIZER = "novel milp optimizer"
    PRIVACY_PRESERVING = "privacy-preserving"
    DIFFERENTIALLY_PRIVATE = "differentially private"
    GUARANTEED_SOC_WORKLOAD = "guaranteed soc workload"
    GUARANTEED_FUTURE_FPR = "guaranteed future fpr"
    FULLY_UNSUPERVISED_FABRID = "fully unsupervised fabrid"
    ZERO_DAY_GUARANTEE = "zero-day guarantee"
    REAL_WORLD_DEPLOYMENT_VALIDATED = "real-world deployment validated"
    END_TO_END_OPTIMAL = "end-to-end optimal"
    GLOBALLY_OPTIMAL_SECURITY = "globally optimal security"
    ROBUST_TO_MALICIOUS_CLIENTS = "robust to malicious clients"
    CONCEPT_DRIFT_PROOF = "concept-drift proof"


@dataclass(frozen=True, slots=True)
class PublicationText:
    value: str

    def __post_init__(self) -> None:
        if not self.value.strip():
            raise ValueError("publication text must not be empty")


@dataclass(frozen=True, slots=True)
class ForbiddenClaimMatch:
    phrase: ForbiddenClaimPhrase
    context: PublicationText


def find_forbidden_claims(text: PublicationText) -> tuple[ForbiddenClaimMatch, ...]:
    normalized = text.value.lower()
    matches: list[ForbiddenClaimMatch] = []
    for phrase in ForbiddenClaimPhrase:
        for match in re.finditer(re.escape(phrase.value), normalized):
            start = max(0, match.start() - 30)
            end = min(len(text.value), match.end() + 30)
            matches.append(
                ForbiddenClaimMatch(
                    phrase=phrase,
                    context=PublicationText(text.value[start:end]),
                )
            )
    return tuple(matches)


def validate_no_forbidden_claims(text: PublicationText) -> None:
    matches = find_forbidden_claims(text)
    if matches:
        phrases = ", ".join(match.phrase.value for match in matches)
        raise ValueError(f"forbidden publication claims found: {phrases}")
