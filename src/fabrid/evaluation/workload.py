from __future__ import annotations

import math
from dataclasses import dataclass

from fabrid.domain.values import BitCount, ByteCount, CandidateCount, ClientCount

_FLOAT32_BYTES = ByteCount(4)
_UUID_BYTES = ByteCount(16)
_UINT64_BYTES = ByteCount(8)
_UINT32_BYTES = ByteCount(4)
_UINT16_BYTES = ByteCount(2)
_SHA256_BYTES = ByteCount(32)


@dataclass(frozen=True, slots=True)
class ClientUploadPayload:
    candidate_count: CandidateCount

    @property
    def utility_values_bytes(self) -> ByteCount:
        return ByteCount(self.candidate_count.value * _FLOAT32_BYTES.value)

    @property
    def total_bytes(self) -> ByteCount:
        return ByteCount(
            self.utility_values_bytes.value
            + _UUID_BYTES.value
            + _UINT64_BYTES.value
            + _UINT32_BYTES.value
            + _UINT32_BYTES.value
            + _UINT16_BYTES.value
            + _UINT16_BYTES.value
            + _SHA256_BYTES.value
        )


def federation_upload_bytes(
    payload: ClientUploadPayload,
    client_count: ClientCount,
) -> ByteCount:
    return ByteCount(payload.total_bytes.value * client_count.value)


def candidate_index_bits(candidate_count: CandidateCount) -> BitCount:
    return BitCount(max(1, math.ceil(math.log2(candidate_count.value))))


def candidate_index_bytes(candidate_count: CandidateCount) -> ByteCount:
    bits = candidate_index_bits(candidate_count)
    return ByteCount(math.ceil(bits.value / 8))
