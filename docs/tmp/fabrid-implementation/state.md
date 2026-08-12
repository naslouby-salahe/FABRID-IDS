# State

canonical roadmap path: `docs/FABRID-IDS Roadmap.md` (v2.0, protocol date 2026-08-12; section 18
rewritten under decision D003 to make FABRID detector-agnostic/standalone — no other section changed)
current git commit: e87a91a (23 commits this session)

## Summary

The entire FABRID decision layer plus a standalone detector substrate are implemented, unit-tested,
and validated end-to-end against real N-BaIoT data. Fast suite: 193/193 tests (~5s). Integration suite
(real data, marked `@pytest.mark.integration`, excluded by default): 18/18 tests (~60s). ruff and
pyright both clean throughout. A smoke run of the complete pipeline (real ingestion -> partitioning ->
per-client scaling -> FedAvg training -> scoring -> AUROC) on 3 real clients (subsampled to 400 rows)
completed in 29.5s with plausible AUROC 0.91-1.00.

## What is implemented (all with tests, all VERIFIED in the audit matrix where applicable)

- Identity/protocol freeze, 207-point alpha grid (`config/`)
- Typed canonical `Protocol` loader (`config/protocol.py`) — single source for all constants
- Source-order split-boundary arithmetic, exact match to all 9 published N-BaIoT client counts
  (`data/partitioner.py`)
- Finite-sample order-statistic calibration + independent final calibration (`calibration/`)
- Record-level metrics: MacroRecall, WorstClientRecall, FPR_fed, BUR/BVR, dispersion, Gini, H_U
  heterogeneity (`evaluation/`)
- Score contract (strict `>`, AUROC invariance) + immutable `ScoreArtifact` schema (`scoring/`,
  `schemas/`)
- Client utility curves, eligibility gate, fallback rate, frontier orchestration (`frontier/`,
  `data/eligibility.py`)
- All 6 primary policies + conditional EQ_ALERT: EQ_FPR, GREEDY, FABRID_MACRO, FABRID_MINIMAX,
  POOLED_SHARED, TEST_ORACLE (`allocation/`) — MACRO/MINIMAX have brute-force parity + 100x-determinism
  tests passing
- Strict `scipy.optimize.milp` wrapper with `SOLVER_INVALID` rejection (`optimization/milp.py`)
- Statistics: exact sign-flip test, paired bootstrap CI, Holm correction (`statistics/`)
- Generic audit checks: T01 (partition exclusivity), T07/T08 (score+AUROC identity), T09/T10 (budget
  invariants), T12 (determinism) (`audit/`)
- **Standalone detector substrate** (new, per decision D003 — not inherited from any external stack):
  - `data/nbaiot_reader.py`: real CSV ingestion, verified against all 9 real devices
  - `data/feature_manifest.py`: frozen+hashed 115-feature manifest, verified identical across all 9
    real devices
  - `data/preprocessing.py`: TRAIN-only z-score `FeatureScaler`
  - `detector/model.py`: fixed autoencoder, `reconstruction_error_scores`
  - `detector/training.py`: FedAvg (local epochs + row-count-weighted averaging)
  - `scoring/score_generation.py`: wires all of the above into `generate_score_artifact`
  - `scripts/smoke_pipeline.py`: real-data end-to-end validation (not a confirmatory run)

## What is NOT yet implemented

- Full-scale confirmatory run: all 10 seeds x 9 clients x full (non-subsampled) row counts, producing
  real persisted `ScoreArtifact`s with hashes for every dataset x seed x client x split coordinate.
  This is the next concrete blocking step and is a genuinely long-running compute task (the benign-only
  read across all 9 clients alone took ~70-80s; full attack data is several times larger; FedAvg
  training across 9 clients x 10 seeds at full scale has not been timed yet). Should be launched
  deliberately (likely as a background task) rather than casually, and its results reviewed before
  being treated as confirmatory.
- Main N-BaIoT experiment execution (Phase 12): running all policies across 10 seeds x 5 budgets and
  persisting the 37-field result schema (`schemas/result.py` not yet written).
- Attack-subtype-disjoint / botnet-family-disjoint generalization runs (Phase 14-15).
- Allocation-sensitivity (500 resamples) and conservative-utility (LCB) analyses (Phase 16-17).
- Remaining T02-T06 (perturbation invariance — need the real pipeline to be meaningful), T13-T18
  (some covered inline in FABRID_MACRO/MINIMAX tests already; not yet promoted to the generic `audit/`
  module).
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

## Next implementation chunk (priority order)

1. Decide and freeze exact detector hyperparameters (architecture, learning rate, local epochs,
   rounds, batch size) as a persisted config, analogous to `protocol.yaml` — currently only exercised
   ad hoc in tests/smoke script. Roadmap does not prescribe exact values (architecture is explicitly
   "not the contribution"), so this is an engineering decision to record, not derive from the roadmap.
2. Build the full (non-subsampled) orchestration script/module that trains all 10 seeds and persists
   `ScoreArtifact`s + hashes to a results directory, gated behind an explicit run command (not
   auto-executed). Time a single full seed first before committing to all 10.
3. `schemas/result.py` (37-field result row) and the evaluation/statistics glue that populates it from
   persisted score artifacts + allocation runs.
4. Main N-BaIoT experiment execution once (2) and (3) exist.
