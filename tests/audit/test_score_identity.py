from __future__ import annotations

import pytest

from fabrid.audit.score_identity import (
    ScoreIdentityError,
    assert_auroc_identity,
    assert_score_hash_identity,
)


def test_identical_hashes_pass() -> None:
    assert_score_hash_identity({"EQ_FPR": "abc", "GREEDY": "abc", "FABRID_MACRO": "abc"})


def test_differing_hashes_raise() -> None:
    with pytest.raises(ScoreIdentityError):
        assert_score_hash_identity({"EQ_FPR": "abc", "GREEDY": "def"})


def test_empty_mapping_rejected() -> None:
    with pytest.raises(ValueError):
        assert_score_hash_identity({})


def test_identical_auroc_passes() -> None:
    assert_auroc_identity({"EQ_FPR": 0.9, "GREEDY": 0.9})


def test_differing_auroc_raises() -> None:
    with pytest.raises(ValueError):
        assert_auroc_identity({"EQ_FPR": 0.9, "GREEDY": 0.8})
