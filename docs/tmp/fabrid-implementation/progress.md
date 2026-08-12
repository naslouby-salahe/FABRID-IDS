# Progress Log

## 2026-08-12 — Session 1 (Phase 0-1)

- Read `prompt.md` and the full FABRID-IDS roadmap (v2.0) end to end.
- Verified shared raw data at `/home/naslouby/Projects/datp-shared-data/raw`: N-BaIoT present
  (9 device directories), CIC_IOT_Dataset2023 present (not the same as CIC IoT-DIAD 2024 required
  for external replication), Edge-IIoTset present. CIC IoT-DIAD 2024, Gotham 2025, CICIoMT2024 not present.
- Discovered `/home/naslouby/Projects/datp-core` and `/home/naslouby/Projects/datp` as prior mature
  research stacks with federated detector training, scoring, checkpointing, and threshold/calibration
  modules — the "existing DATP experimental stack" referenced in roadmap section 18. Plan to depend on
  datp-core for frozen detector/score generation rather than reimplementing it inside fabrid-ids.
- Created `data/raw` symlink to shared raw data.
- Scaffolded `src/fabrid/` package per roadmap section 88 architecture, `pyproject.toml`.
- Implemented and froze Phase 0 identity constants (`src/fabrid/__init__.py`).
- Implemented Phase 1 protocol artifacts:
  - `src/fabrid/config/alpha_grid.py` — exact section-29 log-spaced + explicit-point grid construction,
    dedup at 1e-12 tolerance, frozen `AlphaGrid` dataclass enforcing count==207 and sortedness.
  - `src/fabrid/config/alpha_grid.json` — generated and verified (207 unique values).
  - `src/fabrid/config/protocol.yaml` — full frozen protocol (score contract, seeds, budgets, split
    fractions, finite-sample calibration formula, solver settings, tie-breaking sequences, statistics
    settings, practical gates, event gate parameters).
  - `src/fabrid/config/attack_folds.yaml` — section 58 fixed fold mapping + rotations, section 59
    botnet-family-disjoint directions.
  - `src/fabrid/config/datasets.yaml` — exact section 22/25 N-BaIoT 9-client table with exact split counts;
    CIC IoT-DIAD 2024 eligibility contract recorded with status `candidate_external_replication`.
- Created restart tracking folder (`docs/tmp/fabrid-implementation/`) and audit matrix skeleton.

Next: Phase 2 dataset provenance (partitioner + eligibility + exclusivity tests), after confirming
exact reuse boundary with datp-core's N-BaIoT reader.

## 2026-08-12 — Session 1 continued (Phase 2, split arithmetic)

- Read `datp_core.data.nbaiot.reader.NBaIoTReader`: confirmed it preserves source row order
  (`with_row_index`), attaches canonical `physical_client_id`, `attack_family`, `attack_subtype` columns
  from filename parsing, and validates finite feature values. This is the correct reuse point for raw
  N-BaIoT ingestion + provenance; `datp_core` package confirmed importable in this environment.
- Implemented `src/fabrid/data/partitioner.py`: exact roadmap section 24/26 floor-boundary formulas as
  typed, validated dataclasses (`BenignSplitBoundaries`, `AttackSplitBoundary`) with `BenignSplit`/
  `AttackSplit` `StrEnum`s — no magic strings for split names.
- Added `tests/data/test_partitioner.py` (28 tests): exact reproduction of all 9 roadmap-published
  N-BaIoT per-client benign split counts (section 25), exclusivity/coverage sweeps across representative
  n, zero/negative/out-of-range edge cases. All pass.
- Ran batched verification: `ruff check`, `ruff format --check`, `pytest` — all clean/green.
- Updated `pyproject.toml`: `tool.pytest.ini_options.pythonpath=["src"]`, `tool.pyright.extraPaths=["src"]`
  so the src-layout package resolves for both tools without an editable install.

Next: raw-data wiring (NBaIoTReader -> partitioner -> split manifest), provenance/eligibility modules.

## 2026-08-12 — Session 1 continued (baselines, MILP optimizer, FABRID_MACRO/MINIMAX)

- Added typed allocation contracts (`src/fabrid/schemas/allocation.py`), `EQ_FPR` and `GREEDY`
  baselines, `src/fabrid/optimization/milp.py` (strict `scipy.optimize.milp` wrapper), shared MILP
  formulation helpers, and `FABRID_MACRO`/`FABRID_MINIMAX` with full deterministic tie-breaking.
- Verified both FABRID policies against exhaustive brute-force enumeration on a 3-client/4-candidate
  synthetic case and 100x-repeated-solve determinism — both pass.
- Installed `scipy-stubs` for real `scipy.optimize` typing under strict pyright.

## 2026-08-12 — Course correction: standalone decoupling (decision D003)

User instruction: FABRID-IDS must be fully standalone with no scientific or runtime dependency on DATP,
not presented as a DATP extension/variant. Applied:
- Rewrote roadmap section 18 (Detector Contract) to be detector-agnostic/standalone, replacing the one
  DATP reference in the whole roadmap file.
- Updated the audit matrix: reworded `PREPROCESS-001`/`TRAIN-001`/`TRAIN-002`, added `ARCH-004` (no
  external dependency, VERIFIED via grep), `MODEL-003`, `SCORE-003`, `ARCH-005`, and a "Standalone/
  decoupling audit" cross-reference table covering the seven explicit checks requested.
- Marked decision D001 superseded, added D003 explaining the correction.
- Confirmed via `grep -rni datp src tests pyproject.toml`: zero matches — no source code required any
  change, since the decision layer built so far was already dependency-free by construction. The only
  planned dependency (detector training reusing an external stack) had not yet been implemented.
- Rewrote `state.md` (had drifted into repetitive, stale "previously completed" blocks) into one
  consolidated, current status section.

Next: finish the small open pyright findings on the MILP/minimax optimizer files, then start the
standalone detector/scoring substrate (Phase 2 real-data ingestion + Phase 3 training), implemented
directly in `fabrid` per D003.
