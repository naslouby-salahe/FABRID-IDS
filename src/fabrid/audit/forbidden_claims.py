"""Forbidden-claim detection: scan generated report/manuscript text for phrases the roadmap
explicitly prohibits, regardless of what the underlying results show.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_FORBIDDEN_PHRASES = (
    "first alert-budget ids",
    "first federated thresholding method",
    "first heterogeneous detector threshold allocator",
    "first ids resource allocator",
    "novel milp optimizer",
    "privacy-preserving",
    "differentially private",
    "guaranteed soc workload",
    "guaranteed future fpr",
    "fully unsupervised fabrid",
    "zero-day guarantee",
    "real-world deployment validated",
    "end-to-end optimal",
    "globally optimal security",
    "robust to malicious clients",
    "concept-drift proof",
)


@dataclass(frozen=True, slots=True)
class ForbiddenClaimMatch:
    phrase: str
    context: str


def find_forbidden_claims(text: str) -> tuple[ForbiddenClaimMatch, ...]:
    normalized = text.lower()
    matches: list[ForbiddenClaimMatch] = []
    for phrase in _FORBIDDEN_PHRASES:
        for match in re.finditer(re.escape(phrase), normalized):
            start = max(0, match.start() - 30)
            end = min(len(text), match.end() + 30)
            matches.append(ForbiddenClaimMatch(phrase=phrase, context=text[start:end]))
    return tuple(matches)


def assert_no_forbidden_claims(text: str) -> None:
    matches = find_forbidden_claims(text)
    if matches:
        formatted = "; ".join(f"'{m.phrase}' near '...{m.context}...'" for m in matches)
        raise ValueError(f"forbidden claim(s) found: {formatted}")
