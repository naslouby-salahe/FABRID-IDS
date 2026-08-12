# State

canonical roadmap path: `docs/FABRID-IDS Roadmap.md` (v2.0, protocol date 2026-08-12)
current git commit (as of last update): 8eefad8 (pre-checkpoint; this chunk not yet committed)
current roadmap phase: Phase 2 (dataset provenance) in progress — split-boundary arithmetic done,
raw-data ingestion/provenance wiring not yet done
current requirement/group: SPLIT-001..004 (boundary arithmetic) DONE; DATASET/CLIENT provenance wiring
to `datp_core.data.nbaiot.NBaIoTReader` NEXT

last completed major implementation chunk (this entry, supersedes prior "this entry"):
- `src/fabrid/schemas/allocation.py`: `AllocationPolicy` StrEnum, validated `ClientUtilityCurve`
  (ascending grid starting at 0.0, no duplicates, utility in [0,1]), `AllocationDecision`, `Allocation`
  (with `total_weighted_cost`/`is_budget_feasible` helpers).
- `src/fabrid/allocation/equal_fpr.py`: EQ_FPR baseline.
- `src/fabrid/allocation/greedy.py`: GREEDY baseline with the exact 4-level tie-break order
  (largest efficiency, larger delta-utility, lower incremental cost, lower client_id, lower resulting
  alpha), refactored into `_feasible_increment`/`_best_increment`/`_validate_inputs` helpers to keep
  cyclomatic complexity down.
- `tests/allocation/test_equal_fpr.py`, `tests/allocation/test_greedy.py`: 14 tests incl. a hand-verified
  efficiency-ordering scenario, alpha_max capping, zero-budget, budget-never-exceeded sweep, tie-break
  determinism, mismatched-grid rejection, and `ClientUtilityCurve` invariant tests.
- Batched verification: ruff, pyright, pytest all clean — 70/70 tests total.

Previously completed (superseded entry, kept for history):
- `src/fabrid/evaluation/record_level.py`: MacroRecall, WorstClientRecall, federation FPR (weighted),
  BUR (never clamped), BVR, FPR dispersion (median/IQR/min/max/CV with `None`/NA when mean FPR is 0),
  false-alert Gini (0 when total is 0). Typed `ClientId`/`AttackSubtype` NewTypes, `ClientWeight`/
  `TruePositiveRate`/`FalsePositiveRate` validated value dataclasses — no raw floats crossing this
  module's public API.
- `tests/evaluation/test_record_level.py`: 14 tests covering formulas, equal-client weighting, BUR
  non-clamping, NA-vs-zero CV distinction, Gini edge cases, invalid-value rejection.
- Batched verification: ruff format/check, pyright, pytest — all clean, 56/56 tests passing total.

Previously completed (superseded entry, kept for history):
- `src/fabrid/data/partitioner.py`: pure index-arithmetic `compute_benign_split_boundaries` (i1/i2/i3
  floor rule, section 24) and `compute_attack_split_boundary` (j_a floor rule, section 26), typed via
  `BenignSplit`/`AttackSplit` StrEnums and frozen dataclasses with invariant validation in `__post_init__`.
- `tests/data/test_partitioner.py`: 28 tests, including exact reproduction of all 9 published N-BaIoT
  per-client split counts (roadmap section 25 table) — all pass. Also covers T01-style exclusivity/
  coverage, zero-row and negative-n edge cases, out-of-range row rejection.
- Confirmed `datp_core` is importable in this environment (`/home/naslouby/Projects/datp-core`, installed
  as `datp-core` 0.1.0) and its `datp_core.data.nbaiot.NBaIoTReader` already preserves source row order
  via `with_row_index` and attaches canonical device/attack-subtype columns from filename parsing — this
  is the right reuse point for raw ingestion + provenance (decision D001 confirmed for the reader; still
  need to check `datp_core.data.preprocessing` and `datp_core.detector` contracts before Phase 3).
- ruff check/format clean on `src/` + `tests/`; pytest full run green (28/28).

Previously completed (prior entry):
- Phase 0 identity freeze (`src/fabrid/__init__.py`)
- Phase 1 protocol freeze: `src/fabrid/config/protocol.yaml`, `src/fabrid/config/alpha_grid.json`
  (207 values, generated+verified via `src/fabrid/config/alpha_grid.py`), `src/fabrid/config/attack_folds.yaml`,
  `src/fabrid/config/datasets.yaml`
- Repository scaffold per roadmap section 88 architecture (`src/fabrid/{config,data,scoring,calibration,
  frontier,allocation,optimization,evaluation,statistics,audit,schemas}`)
- `data/raw` symlink -> `/home/naslouby/Projects/datp-shared-data/raw` (contains N-BaIoT, CIC_IOT_Dataset2023,
  Edge-IIoTset; CIC IoT-DIAD 2024 NOT present — external replication dataset acquisition is a known blocker,
  see failures.md)
- Audit matrix skeleton created at `docs/FABRID_IDS_Audit_Implementation_Matrix.md`

last verified major implementation chunk: none yet (no batched verification cycle run; only alpha_grid
generation script executed and count checked = 207)

next implementation chunk:
- Phase 2 continued: wire `datp_core.data.nbaiot.NBaIoTReader` output (per-client LazyFrame with
  `source_row_index`, canonical client/attack columns) through `compute_benign_split_boundaries`/
  `compute_attack_split_boundary` to assign partition membership per row; persist a split manifest;
  add `fabrid/data/provenance.py` for the score-artifact provenance columns (section 89) and
  `fabrid/data/eligibility.py` for utility eligibility guardrails (section 36).
- Reuse assessment: `/home/naslouby/Projects/datp-core` has a mature `datp_core.data.nbaiot` reader/
  materializer and `datp_core.detector` training/scoring/checkpoint stack, plus
  `datp_core.thresholds.calibration`. Per roadmap section 18 ("Where FABRID is implemented on the
  existing DATP experimental stack, inherit its frozen preprocessing/FedAvg/..."), FABRID-IDS should
  depend on datp-core for detector training + score generation rather than reimplementing federated
  training. FABRID-IDS's own `src/fabrid/` package owns only the allocation/calibration/optimization/
  statistics decision layer that is the actual roadmap contribution. This must be confirmed by reading
  datp-core's nbaiot reader/schema and detector/scoring contracts before Phase 2 code is written.

known blockers:
- CIC IoT-DIAD 2024 raw data not present under `datp-shared-data/raw`. External replication (Phase 19-20)
  cannot proceed until acquired. Not blocking primary N-BaIoT work (Phases 2-18, 21-25 minus event/external
  branches). Will mark EXTERNAL-* / GATE-G15 rows BLOCKED_EXTERNAL if acquisition remains impossible.
- Gotham 2025 / CICIoMT2024 (event-level dataset candidates, Phase 21) also not present under raw data path.
  Event-level claims (EVENT-*, GATE-G16) likely BLOCKED_EXTERNAL pending data acquisition; record decision
  once Phase 21 is reached.

known stale/incomplete areas: whole `src/fabrid/` decision-layer implementation is new; nothing yet verified
against actual N-BaIoT data.

important pending test/audit runs:
- alpha_grid uniqueness/count check: DONE (passed, 207).
- T01-T18 mandatory scientific software tests: NOT YET WRITTEN.
- ruff/pyright/pytest batched cycle: NOT YET RUN (no substantial code beyond config/alpha_grid yet).
