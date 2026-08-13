from __future__ import annotations

import pytest

from fabrid.domain.values import CandidateCount, ClientCount
from fabrid.evaluation.workload import (
    ClientUploadPayload,
    candidate_index_bits,
    candidate_index_bytes,
    federation_upload_bytes,
)

_ALPHA_GRID_SIZE = CandidateCount(207)


def test_client_payload_matches_published_896_bytes() -> None:
    payload = ClientUploadPayload(candidate_count=_ALPHA_GRID_SIZE)
    assert payload.utility_values_bytes.value == 828
    assert payload.total_bytes.value == 896


def test_nine_client_federation_matches_published_8064_bytes() -> None:
    payload = ClientUploadPayload(candidate_count=_ALPHA_GRID_SIZE)
    assert federation_upload_bytes(payload, ClientCount(9)).value == 8_064


def test_105_client_federation_matches_published_94080_bytes() -> None:
    payload = ClientUploadPayload(candidate_count=_ALPHA_GRID_SIZE)
    assert federation_upload_bytes(payload, ClientCount(105)).value == 94_080


def test_candidate_index_needs_8_bits_for_207_candidates() -> None:
    assert candidate_index_bits(_ALPHA_GRID_SIZE).value == 8
    assert candidate_index_bytes(_ALPHA_GRID_SIZE).value == 1


def test_invalid_candidate_count_rejected() -> None:
    with pytest.raises(ValueError):
        CandidateCount(0)


def test_invalid_client_count_rejected() -> None:
    with pytest.raises(ValueError):
        ClientCount(0)
