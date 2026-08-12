"""T07/T08: every policy at one dataset x seed x client coordinate must reference the same score
artifact, and their AUROC values (which depend only on score ranking) must be numerically identical.
"""

from __future__ import annotations

from collections.abc import Mapping

from fabrid.scoring.score_contract import assert_auroc_invariant


class ScoreIdentityError(Exception):
    pass


def assert_score_hash_identity(score_sha256_by_policy: Mapping[str, str]) -> None:
    if not score_sha256_by_policy:
        raise ValueError("assert_score_hash_identity requires at least one policy")
    hashes = set(score_sha256_by_policy.values())
    if len(hashes) > 1:
        raise ScoreIdentityError(
            f"policies reference different score artifacts: {dict(score_sha256_by_policy)}"
        )


def assert_auroc_identity(auroc_by_policy: Mapping[str, float]) -> None:
    assert_auroc_invariant(auroc_by_policy)
