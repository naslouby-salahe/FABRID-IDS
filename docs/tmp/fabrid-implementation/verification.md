# Batched Verification Cycles

## Cycle 0 (pre-Phase-2)

Only `src/fabrid/config/alpha_grid.py` executed directly: `python -m fabrid.config.alpha_grid` produced
207 unique sorted values, matching the required grid size. No `ruff`/`pyright`/`pytest` batched
cycle run yet — insufficient code volume to warrant it per prompt.md section 7 (batched, not
per-file, verification).

## Cycle 1 (partitioner + calibration)

`ruff format src tests`, `ruff check src tests`, `pyright`, `pytest tests/ -q` all run together after
landing `src/fabrid/data/partitioner.py`, `src/fabrid/calibration/order_statistic.py`, and
`src/fabrid/config/protocol.py` (typed protocol-constant loader). Result: all clean —
0 ruff findings, 0 pyright errors (after excluding `reportUnknownMemberType` for `tests/` only, scoped
to `pytest.approx`'s own partially-typed stubs — never affects `src/`), 42/42 pytest passed.

Follow-up from user feedback mid-session: renamed opaque `i1/i2/i3/n` boundary fields to descriptive
names (`train_end`/`frontier_end`/`final_cal_end`/`total_rows`), replaced hardcoded split-fraction
module constants with a typed `Protocol`/`BenignSplitFractions`/`AttackSplitFraction` loader reading
the single canonical `protocol.yaml`, added `RowCount`/`RowIndex` `NewType`s to avoid raw-int primitive
leakage, and removed all "roadmap section N" references from code/test docstrings and comments
(kept only in `docs/` where such references belong).
