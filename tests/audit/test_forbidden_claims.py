from __future__ import annotations

import pytest

from fabrid.audit.forbidden_claims import assert_no_forbidden_claims, find_forbidden_claims


def test_clean_text_has_no_matches() -> None:
    text = "FABRID reallocates a shared false-alert budget across heterogeneous clients."
    assert find_forbidden_claims(text) == ()
    assert_no_forbidden_claims(text)  # must not raise


def test_detects_forbidden_phrase_case_insensitively() -> None:
    text = "Our method is Privacy-Preserving by design."
    matches = find_forbidden_claims(text)
    assert len(matches) == 1
    assert matches[0].phrase == "privacy-preserving"


def test_detects_multiple_distinct_phrases() -> None:
    text = "This is the first IDS resource allocator and is fully unsupervised FABRID."
    matches = find_forbidden_claims(text)
    phrases = {m.phrase for m in matches}
    assert "first ids resource allocator" in phrases
    assert "fully unsupervised fabrid" in phrases


def test_assert_raises_with_context() -> None:
    with pytest.raises(ValueError, match="differentially private"):
        assert_no_forbidden_claims("The protocol is differentially private.")


def test_permitted_claims_pass() -> None:
    # roadmap section 101's explicitly permitted pre-registered claims.
    text = (
        "FABRID formulates the allocation of client operating points in federated IoT "
        "anomaly detection as a shared nominal false-alert budget problem."
    )
    assert_no_forbidden_claims(text)
