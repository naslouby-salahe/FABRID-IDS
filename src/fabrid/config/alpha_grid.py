"""Candidate target-rate grid construction.

The grid is generated once, persisted to ``src/fabrid/config/alpha_grid.json``, and
must never be regenerated implicitly at run time. ``load_alpha_grid`` reads the
frozen artifact; ``build_alpha_grid`` exists only to produce/verify it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ALPHA_GRID_PATH = Path(__file__).with_name("alpha_grid.json")

_LOG_MIN = 1e-4
_LOG_MAX = 0.05
_LOG_POINTS = 201
_EXTRA_POINTS = (0.001, 0.0025, 0.005, 0.01, 0.02)
_DEDUP_ABS_TOL = 1e-12
_EXPECTED_GRID_SIZE = 207


@dataclass(frozen=True, slots=True)
class AlphaGrid:
    """Frozen, sorted, deduplicated candidate target false-positive-rate grid."""

    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.values) != _EXPECTED_GRID_SIZE:
            raise ValueError(
                f"alpha grid must contain {_EXPECTED_GRID_SIZE} unique values, "
                f"got {len(self.values)}"
            )
        if list(self.values) != sorted(self.values):
            raise ValueError("alpha grid must be sorted ascending")

    def __len__(self) -> int:
        return len(self.values)

    def __iter__(self):
        return iter(self.values)

    def as_array(self) -> np.ndarray:
        return np.asarray(self.values, dtype=np.float64)


def build_alpha_grid() -> AlphaGrid:
    """Construct the candidate grid: a log-spaced core plus explicit round-number points."""
    log_min_exp = np.log10(_LOG_MIN)
    log_max_exp = np.log10(_LOG_MAX)
    j = np.arange(_LOG_POINTS, dtype=np.float64)
    exponent = log_min_exp + j * (log_max_exp - log_min_exp) / (_LOG_POINTS - 1)
    log_space = np.power(10.0, exponent)

    raw = np.concatenate(
        [
            np.array([0.0], dtype=np.float64),
            log_space,
            np.array(_EXTRA_POINTS, dtype=np.float64),
        ]
    )
    raw.sort()

    deduped: list[float] = []
    for value in raw:
        if not deduped or abs(value - deduped[-1]) > _DEDUP_ABS_TOL:
            deduped.append(float(value))

    return AlphaGrid(values=tuple(deduped))


def persist_alpha_grid(grid: AlphaGrid, path: Path = ALPHA_GRID_PATH) -> None:
    path.write_text(json.dumps({"alpha_grid": list(grid.values)}, indent=2) + "\n")


def load_alpha_grid(path: Path = ALPHA_GRID_PATH) -> AlphaGrid:
    """Load the frozen grid artifact. Never regenerates it."""
    if not path.exists():
        raise FileNotFoundError(
            f"frozen alpha grid not found at {path}; run "
            "`python -m fabrid.config.alpha_grid` once to generate it"
        )
    payload = json.loads(path.read_text())
    return AlphaGrid(values=tuple(float(v) for v in payload["alpha_grid"]))


def main() -> None:
    grid = build_alpha_grid()
    persist_alpha_grid(grid)
    print(f"persisted {len(grid)} alpha values to {ALPHA_GRID_PATH}")


if __name__ == "__main__":
    main()
