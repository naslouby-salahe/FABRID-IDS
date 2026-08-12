# Decisions

## D001 — SUPERSEDED by D003. Originally: reuse datp-core for detector/scoring, fabrid-ids owns the decision layer only

Roadmap section 18: "Where FABRID is implemented on the existing DATP experimental stack, inherit its
frozen preprocessing; FedAvg implementation; local training rule; architecture; optimization
hyperparameters; training rounds; anomaly-score function." `/home/naslouby/Projects/datp-core` is that
stack (federated detector training, checkpoints, scoring, thresholds already implemented and presumably
verified in its own project). Decision: `fabrid-ids` will depend on `datp-core` as a library for detector
training + immutable score generation (Phases 3-4), and will not reimplement federated training,
preprocessing, or the autoencoder. `src/fabrid/` implements only the FABRID-specific decision layer: dataset
partitioning contracts specific to the FABRID protocol (roadmap section 24-26 exact split rule, which is
FABRID-specific and NOT necessarily the same as datp-core's existing splits), calibration, utility
frontier, allocation (EQ_FPR/GREEDY/FABRID_MACRO/FABRID_MINIMAX/POOLED_SHARED/TEST_ORACLE), optimization,
evaluation, statistics, and audit modules per section 88's architecture.

This must be re-verified once datp-core's nbaiot reader/schema and detector/scoring contracts are read in
Phase 2/3: if datp-core's existing N-BaIoT splits or score contract conflict with the FABRID roadmap's
exact partitioning rule (section 24-26) or score contract (section 20, strict `>`), the FABRID roadmap
wins per prompt.md section 0 ("roadmap overrides existing code"), and fabrid-ids will need to either
(a) call datp-core's raw feature/preprocessing pipeline but apply FABRID's own partitioner and calibration
rule, or (b) reimplement the minimal preprocessing needed if datp-core's is not reusable as-is.

## D003 — FABRID-IDS is fully standalone; no scientific or runtime dependency on DATP

Course correction (explicit user instruction, supersedes D001): FABRID-IDS must not be presented as a
DATP extension/variant/derivative and must not scientifically or at runtime depend on
`datp-core`/`datp`. The detector is only the fixed experimental substrate; FABRID's contribution is the
post-training decision layer (frontier/allocation/calibration/statistics), which was already being built
standalone in `src/fabrid/` and never imported `datp_core` (verified: `grep -rni datp src tests
pyproject.toml` returns no matches as of this correction).

What changes: the detector-training and score-generation substrate (Phase 3-4) will be implemented
directly inside `fabrid` (or a vendored/reimplemented minimal detector), not obtained via a `datp-core`
library dependency. `docs/FABRID-IDS Roadmap.md` section 18 was rewritten to describe FABRID as
detector-agnostic and standalone; the audit matrix's `PREPROCESS-001`/`TRAIN-001` rows and a new
`ARCH-004` (no external scientific/runtime dependency) row were added/updated accordingly. No code
written so far needs to change — the decision layer (partitioner, calibration, evaluation metrics,
EQ_FPR/GREEDY baselines, MILP optimizer, FABRID_MACRO/MINIMAX) is dependency-free by construction and is
kept as-is.

## D002 — CIC IoT-DIAD 2024 not available; external replication provisionally BLOCKED_EXTERNAL

`datp-shared-data/raw` contains `CIC_IOT_Dataset2023` (a different, earlier CIC dataset) but not
`CIC IoT-DIAD 2024`. Per prompt.md section 11 ("If an external requirement is genuinely impossible, mark
it BLOCKED_EXTERNAL... continue all other work"), primary N-BaIoT work proceeds independently. Will
attempt to locate/acquire the correct dataset before finalizing this as BLOCKED_EXTERNAL — do not
substitute CIC_IOT_Dataset2023 for CIC IoT-DIAD 2024, they are not the same dataset and the roadmap
names the latter specifically (105 devices, 33 attacks, 7 categories, packet+flow representations).
