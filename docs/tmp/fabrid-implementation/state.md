# State

canonical roadmap path: `docs/FABRID-IDS Roadmap.md` (v2.0, protocol date 2026-08-12; section 18
rewritten under decision D003 to make FABRID detector-agnostic/standalone — no other section changed)
current git commit: 405de32 (37 commits this session)

## Summary

The entire FABRID decision layer, a standalone detector substrate, and the bridge from persisted
scores to the allocation layer are implemented, unit-tested, and validated end-to-end against real
N-BaIoT data — including a genuine full-scale (non-subsampled) training run and a first real
scientific comparison. Fast suite: 242/242 tests (~7s). Integration suite (real data,
`@pytest.mark.integration`, excluded by default): 18/18 tests (~60s). ruff and pyright clean
throughout.

## What is implemented (all with tests, all VERIFIED in the audit matrix where applicable)

- Identity/protocol freeze, 207-point alpha grid, frozen detector hyperparameters (`config/`)
- Typed canonical `Protocol`/`DetectorHyperparameters` loaders — single source for all constants
- Source-order split-boundary arithmetic, exact match to all 9 published N-BaIoT client counts
- Finite-sample order-statistic calibration + independent final calibration (`calibration/`)
- Record-level metrics: MacroRecall, WorstClientRecall, FPR_fed, BUR/BVR, dispersion, Gini, H_U
  heterogeneity, communication-overhead accounting (exact byte-count parity with the roadmap),
  eventization pipeline, weight-heterogeneity gamma sweep (`evaluation/`)
- Score contract (strict `>`, AUROC invariance) + immutable `ScoreArtifact` schema + `ResultRow`
  (`scoring/`, `schemas/`)
- Client utility curves (raw + conservative/LCB), eligibility gate, fallback rate, frontier
  orchestration, allocation-sensitivity analysis (`frontier/`, `data/eligibility.py`)
- All 6 primary policies + conditional EQ_ALERT (`allocation/`) — MACRO/MINIMAX have brute-force
  parity + 100x-determinism tests passing
- Strict `scipy.optimize.milp` wrapper with `SOLVER_INVALID` rejection (`optimization/milp.py`)
- Statistics: exact sign-flip test (1024-enumeration verified), paired bootstrap CI, Holm correction
- Generic audit checks: T01, T07-T10, T12 (`audit/`); T11, T13-T18 covered by allocation/calibration
  tests directly
- **Standalone detector substrate** (per decision D003 — no external stack dependency): real CSV
  ingestion, frozen+hashed feature manifest, TRAIN-only z-score scaler, fixed autoencoder, FedAvg
  training, `generate_score_artifact` wiring it all together
- **`scoring/frontier_inputs.py`**: bridges a persisted `ScoreArtifact` into `ClientFrontierInputs`,
  completing the path from real scores to the allocation layer

## Real-data execution status

- `scripts/run_seed_training.py` / `run_all_seeds.py`: full-scale (non-subsampled), resumable,
  10-seed x 9-client training + score persistence to `results/scores/` (gitignored).
- Seed 0 completed in 298.8s: **7,062,606 total score records**, matching the roadmap's "~7.06
  million observations" claim almost exactly — strong independent validation.
- As of this update: seeds 0-2 persisted, remaining seeds running in the background
  (`/tmp/fabrid_all_seeds_run.log`; a background watcher is armed to notify on completion or error).
- `scripts/run_main_comparison.py` (seed 0, budget=0.01, equal-client weighting) — first real
  scientific result: fallback_rate=0.0 (all 9 clients eligible), EQ_FPR MacroRecall=0.8223,
  FABRID_MINIMAX MacroRecall=0.7110 (average/worst-case tradeoff, directionally as expected).

## F001 — open question, not yet resolved

At full scale (9 clients x 207 candidates = 1863 binary vars), `FABRID_MACRO`'s stage-1 MILP hits
`mip_gap≈1.25e-6` within the 60s time limit — short of the frozen `accept_mip_gap_leq=1e-9` — so it is
correctly rejected as `SOLVER_INVALID` per protocol (roadmap: exclude the coordinate, never accept a
time-limited near-optimum). This is protocol working as intended, not a bug, and no tolerance was
weakened. See `failures.md` F001 for full detail. Needs a decision before Phase 12: accept that some
coordinates will legitimately be `SOLVER_INVALID`, or investigate solver-side tuning (warm-starting
across the 3-4 sequential stages, direct HiGHS options). Full main-experiment timing (10 seeds x 5
budgets x {MACRO: 3 solves, MINIMAX: 4 solves}) has not been measured and could be substantial if many
coordinates need the full 60s per solve.

## What is NOT yet implemented

- Statistics/contrast glue that populates `ResultRow` from real allocation runs across all
  seeds/budgets/policies once the full 10-seed training completes (Phase 12-13).
- Attack-subtype-disjoint / botnet-family-disjoint generalization runs (Phase 14-15).
- Remaining T02-T06 (perturbation invariance — need the real pipeline + multiple perturbed runs to be
  meaningful, not yet automated as tests).
- External replication (CIC IoT-DIAD 2024 — BLOCKED_EXTERNAL, dataset not present) and event-level
  validation (Gotham/CICIoMT2024 — also not present).
- Tables/figures generation (Phase 23), reproduction audit (Phase 24), final novelty search (Phase 25).
- Manuscript-stage items (NOVELTY-*, CLAIM-*, forbidden-claims enforcement) — not applicable until a
  manuscript exists.

## Known blockers

- CIC IoT-DIAD 2024 raw data not present under `datp-shared-data/raw`. External replication cannot
  proceed until acquired. Does not block primary N-BaIoT work.
- Gotham 2025 / CICIoMT2024 (event-level candidates) also not present. Event-level claims blocked
  pending acquisition.
- F001 (see above): full-scale FABRID_MACRO solver timing needs a decision before confirmatory Phase
  12 execution.

## Next implementation chunk (priority order)

1. Let the background 10-seed training run finish; verify all 10 manifests + hashes.
2. Resolve F001 (measure how often/how badly this occurs across seeds/budgets before deciding how to
   handle it — do not weaken the tolerance ad hoc).
3. Statistics/contrast glue: populate `ResultRow` from real runs across all seeds x 5 budgets x
   policies; wire Contrast A (`FABRID_MACRO - EQ_FPR` on MacroRecall) and Contrast B
   (`FABRID_MINIMAX - EQ_FPR` on WorstClientRecall) through the sign-flip test + bootstrap + Holm.
4. T02-T06 perturbation-invariance tests against the real pipeline.
