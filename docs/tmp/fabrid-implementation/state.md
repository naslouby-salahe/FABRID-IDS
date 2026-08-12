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

## F001/F002 — two open findings needing a decision before Phase 12 confirmatory execution

**F001 (confirmed systemic across all 10 real seeds, budget=0.01)**: `FABRID_MACRO` hit
`SOLVER_INVALID` in 9/10 seeds (only seed 6 solved to `mip_gap<=1e-9` within 60s); `FABRID_MINIMAX`
hit it in 6/10. This is the protocol working exactly as specified — never accept a time-limited
near-optimum — not a bug, and no tolerance was weakened. But it leaves Contrast A with only n=1 usable
seed at this budget, far short of the paired-10-seed design the roadmap's statistics assume. See
`failures.md` F001. Needs a decision: accept a much smaller effective n and report it honestly, or
investigate solver-side tuning (warm-starting across stages, direct HiGHS options — an implementation
question, not a protocol change).

**F002 (root-caused against real data, not a bug)**: on the seeds where `FABRID_MINIMAX` did solve,
its `WorstClientRecall` was *worse* than `EQ_FPR`'s (mean_diff=-0.45). Verified against seed 6's real
per-client results: `SimpleHome_XCS7_1003` gets alpha=0.01/recall=1.0 under `EQ_FPR` but
alpha≈0.00026/recall=0.0 under `FABRID_MINIMAX`. Mechanism: the roadmap's own stage-3 tie-break
(minimize total budget once `z*` and mean utility are fixed) pushes any non-bottleneck client's alpha
toward zero, since it doesn't affect either binding constraint at *validation* time — but that
client's *test*-time recall can still crash. This is the minimax formulation behaving exactly as
specified; not a defect. Per roadmap section 99 (Negative Result Policy), must be reported as-is if it
persists in the full confirmatory run — do not adjust budgets/folds/objective in response. See
`failures.md` F002.

Full main-experiment timing (10 seeds x 5 budgets x {MACRO: 3 solves, MINIMAX: 4 solves}) has not been
measured and could be substantial if many coordinates need the full 60s per solve.

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

1. Decide how to handle F001/F002 before treating any FABRID_MACRO/MINIMAX result as confirmatory —
   this is a decision, not further implementation; needs explicit sign-off given it may narrow what
   can be claimed.
2. If proceeding: measure whether F001 (SOLVER_INVALID rate) is budget-dependent by running the
   5-budget sweep (currently only budget=0.01 has been run for real). Larger budgets may have fewer
   candidates bound simultaneously and could solve faster/more reliably — untested assumption.
3. Evaluate the roadmap's practical success gates (section 72) honestly against whatever n survives.
4. Tables/figures generation, reproduction audit, once (1)-(3) are settled.

## Blocked external dataset acquisition — download info (2026-08-12)

**CIC IoT-DIAD 2024** (EXTERNAL-001, §74-79): 105-device IoT topology, 33 attacks, 7 categories
(DDoS/DoS/Recon/Web-based/Brute Force/Spoofing/Mirai), CSV format.
- Official page: https://www.unb.ca/cic/datasets/iot-diad-2024.html
- UNB CIC dataset index: https://www.unb.ca/cic/datasets/index.html
- Associated paper: "Device Identification and Anomaly Detection in IoT Environments", IEEE Internet
  of Things Journal, Dec 2024.
- UNB CIC datasets typically require a request form (name/institution/email) before a download link
  is issued — not a direct anonymous download.

**Gotham Dataset 2025** (EVENT-001, §81): 78 emulated IoT devices, MQTT/CoAP/RTSP, PCAP format,
Mirai full kill-chain (scan/brute-force/infect/flood) + DoS/Telnet brute force/CoAP amplification.
- Zenodo (direct download): https://zenodo.org/records/14502760 (DOI 10.5281/zenodo.14502760)
- Paper: arXiv:2502.03134 — https://arxiv.org/abs/2502.03134
- Packet labeller tool: https://github.com/othmbela/gotham-network-packet-labeller

**CICIoMT2024** (alternative/companion event-level candidate, healthcare IoMT, multi-protocol
WiFi/MQTT): 
- Official page: https://www.unb.ca/cic/datasets/iomt-dataset-2024.html
- IEEE DataPort mirror: https://ieee-dataport.org/documents/ciciomt2024wifimqtt (DOI
  10.21227/tq0p-ag21)
- Paper (ScienceDirect): https://www.sciencedirect.com/science/article/pii/S2542660524002920

Not yet verified: exact file sizes, whether IoT-DIAD/IoMT UNB downloads are gated behind a request
form (likely, per CIC's usual process) vs. direct link. User will acquire manually.
