"""Communication-overhead accounting for the client-utility upload and server response.

Logical (pre-serialization) byte counts per the fixed message layout: 207
float32 utility values, a 128-bit client UUID, three count fields, an
eligible-subtype count, flags, and a config hash for the client-to-server
message; a single candidate-index byte plus identifying/integrity metadata
for the server response.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

_FLOAT32_BYTES = 4
_UUID_BYTES = 16
_UINT64_BYTES = 8
_UINT32_BYTES = 4
_UINT16_BYTES = 2
_SHA256_BYTES = 32


@dataclass(frozen=True, slots=True)
class ClientUploadPayload:
    alpha_grid_size: int

    def __post_init__(self) -> None:
        if self.alpha_grid_size < 1:
            raise ValueError(f"alpha_grid_size must be positive, got {self.alpha_grid_size}")

    def utility_values_bytes(self) -> int:
        return self.alpha_grid_size * _FLOAT32_BYTES

    def total_bytes(self) -> int:
        return (
            self.utility_values_bytes()
            + _UUID_BYTES
            + _UINT64_BYTES  # nominal/predeployment count
            + _UINT32_BYTES  # final-calibration count
            + _UINT32_BYTES  # validation-attack count
            + _UINT16_BYTES  # eligible-subtype count
            + _UINT16_BYTES  # flags
            + _SHA256_BYTES  # config sha256
        )


def federation_upload_bytes(payload: ClientUploadPayload, client_count: int) -> int:
    if client_count < 1:
        raise ValueError(f"client_count must be positive, got {client_count}")
    return payload.total_bytes() * client_count


def candidate_index_bits(alpha_grid_size: int) -> int:
    if alpha_grid_size < 1:
        raise ValueError(f"alpha_grid_size must be positive, got {alpha_grid_size}")
    return max(1, math.ceil(math.log2(alpha_grid_size)))


def candidate_index_bytes(alpha_grid_size: int) -> int:
    return math.ceil(candidate_index_bits(alpha_grid_size) / 8)
