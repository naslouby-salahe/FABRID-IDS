from __future__ import annotations

from dataclasses import dataclass

from fabrid.domain.enums import AllocationPolicy
from fabrid.domain.identifiers import ArtifactDigest
from fabrid.evaluation.ranking import PolicyRankingEvidence, assert_auroc_invariant


class ScoreIdentityValidationError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class PolicyScoreDigest:
    policy: AllocationPolicy
    digest: ArtifactDigest


def validate_score_digest_identity(
    evidence: tuple[PolicyScoreDigest, ...],
) -> None:
    if not evidence:
        raise ValueError("score identity validation requires at least one policy")
    policies = tuple(item.policy for item in evidence)
    if len(set(policies)) != len(policies):
        raise ValueError("score identity evidence contains duplicate policies")
    if len({item.digest for item in evidence}) != 1:
        raise ScoreIdentityValidationError(
            "policies reference different frozen score artifacts"
        )


def validate_auroc_identity(
    evidence: tuple[PolicyRankingEvidence, ...],
) -> None:
    assert_auroc_invariant(evidence)
