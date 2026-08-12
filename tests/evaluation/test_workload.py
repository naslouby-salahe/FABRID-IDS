from __future__ import annotations

import pytest

from fabrid.evaluation.workload import (
    ClientUploadPayload,
    candidate_index_bits,
    candidate_index_bytes,
    federation_upload_bytes,
)

_ALPHA_GRID_SIZE = 207


def test_client_payload_matches_published_896_bytes() -> None:
    payload = ClientUploadPayload(alpha_grid_size=_ALPHA_GRID_SIZE)
    assert payload.utility_values_bytes() == 828
    assert payload.total_bytes() == 896


def test_nine_client_federation_matches_published_8064_bytes() -> None:
    payload = ClientUploadPayload(alpha_grid_size=_ALPHA_GRID_SIZE)
    assert federation_upload_bytes(payload, client_count=9) == 8_064


def test_105_client_federation_matches_published_94080_bytes() -> None:
    payload = ClientUploadPayload(alpha_grid_size=_ALPHA_GRID_SIZE)
    assert federation_upload_bytes(payload, client_count=105) == 94_080


def test_candidate_index_needs_8_bits_for_207_candidates() -> None:
    assert candidate_index_bits(_ALPHA_GRID_SIZE) == 8
    assert candidate_index_bytes(_ALPHA_GRID_SIZE) == 1


def test_invalid_alpha_grid_size_rejected() -> None:
    with pytest.raises(ValueError):
        ClientUploadPayload(alpha_grid_size=0)
    with pytest.raises(ValueError):
        candidate_index_bits(0)


def test_invalid_client_count_rejected() -> None:
    payload = ClientUploadPayload(alpha_grid_size=_ALPHA_GRID_SIZE)
    with pytest.raises(ValueError):
        federation_upload_bytes(payload, client_count=0)
