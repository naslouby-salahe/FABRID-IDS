# State

canonical roadmap path: `docs/FABRID-IDS Roadmap.md` (v2.0, protocol date 2026-08-12; section 18 rewritten
under decision D003 to make FABRID detector-agnostic/standalone — no other section changed)
current git commit (as of last update): 449a0ef, plus uncommitted work on top (MILP optimizer,
FABRID_MACRO/MINIMAX, standalone-decoupling doc correction)
current roadmap phase: Phase 7 (baselines) and Phase 8-9 (FABRID_MACRO/FABRID_MINIMAX) implemented and
tested; Phase 2 (dataset provenance/ingestion) still not wired to real raw data; Phase 3 (detector
training) now explicitly standalone per D003, not started
current requirement/group: OPTIMIZATION-*, FABRID-MACRO-*, FABRID-MINIMAX-* DONE (brute-force parity +
100x determinism verified); MODEL-003/SCORE-003/ARCH-004/ARCH-005 (standalone decoupling) added and
largely satisfied by construction; NEXT is either (a) finish pyright cleanup on the optimizer chunk, or
(b) start the standalone detector/scoring substrate (Phase 2 ingestion + Phase 3 detector) since that is
the next genuinely blocking gap before any real-data experiment can run

## Course correction applied (decision D003)

User explicitly required FABRID-IDS to be fully standalone with no scientific or runtime dependency on
DATP. Applied:
- `docs/FABRID-IDS Roadmap.md` section 18 rewritten: detector-agnostic pipeline diagram, explicit
  "must not be presented as an extension/variant of any external codebase", detector substrate to be
  implemented directly in FABRID-IDS.
- Audit matrix: `PREPROCESS-001` reworded, `ARCH-004` (no external dependency, VERIFIED via grep),
  `MODEL-003`, `SCORE-003`, `ARCH-005` added, `TRAIN-001/002` reworded, new "Standalone/decoupling audit"
  cross-reference table added.
- `decisions.md`: D001 marked superseded, `D003` records the correction and its rationale.
- Verified via `grep -rni datp src tests pyproject.toml`: zero matches. No code changes were needed —
  everything written so far (partitioner, calibration, evaluation metrics, allocation schemas, EQ_FPR,
  GREEDY, MILP optimizer, FABRID_MACRO, FABRID_MINIMAX) was already dependency-free by construction.
- This state.md file itself had drifted stale during the implementation sprint (kept accreting
  "previously completed" blocks without consolidating); rewritten clean below.

## Consolidated implementation status (supersedes all prior entries in this file)

Implemented, tested, and passing as of this update (81/81 tests before the current optimizer chunk's
final pyright cleanup; a few pyright-only findings remain open on the MILP/FABRID_MACRO/MINIMAX files,
tests themselves all green):

- Phase 0 identity freeze — `src/fabrid/__init__.py`
- Phase 1 protocol freeze — `src/fabrid/config/{protocol.yaml,alpha_grid.json,attack_folds.yaml,
  datasets.yaml}`, typed loader `src/fabrid/config/protocol.py` (`Protocol`/`BenignSplitFractions`/
  `AttackSplitFraction`/`SolverSettings`, single canonical constants source)
- Split-boundary arithmetic — `src/fabrid/data/partitioner.py` (`RowCount`/`RowIndex` NewTypes,
  `BenignSplit`/`AttackSplit` StrEnums), verified against all 9 published N-BaIoT client counts
- Finite-sample calibration — `src/fabrid/calibration/order_statistic.py`
- Record-level evaluation metrics — `src/fabrid/evaluation/record_level.py` (MacroRecall,
  WorstClientRecall, federation FPR, BUR/BVR, dispersion, Gini)
- Allocation contracts — `src/fabrid/schemas/allocation.py` (`AllocationPolicy`, `ClientUtilityCurve`,
  `AllocationDecision`, `Allocation`)
- Baselines — `src/fabrid/allocation/{equal_fpr,greedy}.py`
- MILP optimizer — `src/fabrid/optimization/milp.py` (strict accept/reject on success/status/mip_gap)
- Shared MILP formulation helpers — `src/fabrid/allocation/formulation.py`
- FABRID_MACRO / FABRID_MINIMAX — `src/fabrid/allocation/{fabrid_macro,fabrid_minimax}.py`, both with
  brute-force parity tests (3 clients x 4 candidates) and 100x-repeated-solve determinism tests, both
  passing
- `scipy-stubs` installed (`pip install --break-system-packages scipy-stubs`) to get real typing for
  `scipy.optimize`; `tool.pyright.reportMissingTypeStubs = false` added since scipy ships no inline stubs

Not yet implemented: dataset ingestion/provenance wiring to real raw N-BaIoT data, detector training
substrate (now explicitly standalone, D003), score persistence/hashing, POOLED_SHARED, TEST_ORACLE,
EQ_ALERT, frontier/utility curve construction from real scores, T01-T18 mandatory tests, statistics
module (sign-flip/bootstrap/Holm), external/event branches.

next implementation chunk:
1. Finish the open pyright findings on `src/fabrid/optimization/milp.py` /
   `src/fabrid/allocation/fabrid_minimax.py` (scipy `OptimizeResult.fun`/`mip_gap` optionality per
   scipy-stubs) — small, in progress when the course-correction interrupted it.
2. Standalone detector/scoring substrate (Phase 2 ingestion + Phase 3 training + Phase 4 score
   persistence), implemented directly in `fabrid` per D003 — no external research-stack dependency.
   Raw N-BaIoT reading may still reuse ordinary, generic, non-FABRID-specific libraries (e.g. polars/
   pandas CSV reading) but not another project's federated-training/detector code.
3. `fabrid/frontier/{utility,builder,conservative,stability}.py`, `fabrid/allocation/{pooled_shared,
   test_oracle,equal_alert}.py`, `fabrid/audit/*` (T01-T18), `fabrid/statistics/*`.

known blockers:
- CIC IoT-DIAD 2024 raw data not present under `datp-shared-data/raw`. External replication (Phase 19-20)
  cannot proceed until acquired. Not blocking primary N-BaIoT work. Will mark EXTERNAL-*/GATE-G15
  BLOCKED_EXTERNAL if acquisition remains impossible.
- Gotham 2025 / CICIoMT2024 (event-level dataset candidates, Phase 21) also not present. EVENT-*/GATE-G16
  likely BLOCKED_EXTERNAL pending data acquisition; record decision once Phase 21 is reached.

known stale/incomplete areas: no code yet touches real N-BaIoT data; everything verified so far is
synthetic/unit-level (correct and required, but Phase 2 real-data wiring is the next real gap).

important pending test/audit runs:
- alpha_grid uniqueness/count check: DONE (207).
- Brute-force parity (T11-equivalent) and 100x determinism (T12-equivalent): DONE for FABRID_MACRO and
  FABRID_MINIMAX via `tests/allocation/test_fabrid_{macro,minimax}.py`.
- T01, T02-T06 (leakage/perturbation), T07-T08 (score/AUROC identity), T09-T10, T13-T18: NOT YET WRITTEN
  (need real score/allocation pipeline first for most of these to be meaningful).
- Full ruff/pyright/pytest batched cycle: last clean run was 81/81 tests, ruff clean; pyright had a small
  number of open findings on the MILP/minimax files at the point the course correction arrived — resolve
  next, then re-run the full cycle once more before moving on to Phase 2/3.
