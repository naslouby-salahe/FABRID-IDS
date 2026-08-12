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

## Cycle 2 (MILP optimizer + FABRID_MACRO/MINIMAX)

`ruff format`, `ruff check`, `pyright`, `pytest -q` after landing the optimizer/allocation chunk plus the
standalone-decoupling doc correction (decision D003). Result: 81/81 tests, 0 ruff findings, 0 pyright
errors (after installing `scipy-stubs` and setting `reportMissingTypeStubs = false` — scipy ships no
inline stub package, so without this every module touching `scipy.optimize` cascades into
`Unknown`-typed noise unrelated to actual code correctness). Brute-force parity and 100x-determinism
tests both pass for `FABRID_MACRO` and `FABRID_MINIMAX`.

## Cycle 3 (score contract/artifact, frontier utility, eligibility)

`pytest -q`, `ruff format`, `ruff check --fix`, `pyright` after landing `scoring/score_contract.py`,
`schemas/score_artifact.py`, `frontier/utility.py`, `data/eligibility.py` (+ `UtilityEligibilityGuardrails`
added to the canonical `Protocol` loader). Result: 109/109 tests, 0 ruff findings, 0 pyright errors.

## Cycle 4 (EQ_ALERT, POOLED_SHARED, TEST_ORACLE)

`pytest -q`, `ruff format`, `ruff check`, `pyright` after landing `allocation/equal_alert.py`
(rejects equal-weight misuse), `allocation/pooled_shared.py` (centralized diagnostic, non-federated),
`allocation/test_oracle.py` (isolated via `OracleAccessToken`, wraps `allocate_fabrid_macro` with
test-attack utility curves). Result: 120/120 tests, 0 ruff findings, 0 pyright errors. All six primary
policies (EQ_FPR, GREEDY, FABRID_MACRO, FABRID_MINIMAX, POOLED_SHARED, TEST_ORACLE) plus the conditional
EQ_ALERT baseline are now implemented.

## Cycle 5 (frontier builder orchestration)

`pytest -q`, `ruff format`, `ruff check --fix`, `pyright` after landing `frontier/builder.py`
(`ClientFrontierInputs`/`ClientFrontier`/`FederationFrontier`, eligible/fallback partitioning). Result:
125/125 tests, 0 ruff findings, 0 pyright errors.

## Cycle 6 (audit module: T01, T07-T10, T12 generic checks)

`pytest -q`, `ruff format`, `ruff check --fix`, `pyright` after landing `audit/{split_leakage,
determinism,score_identity,budget_invariants}.py`. Renamed exceptions to the `*Error` suffix per ruff
N818; converted `assert_deterministic` to PEP 695 generic syntax (`def f[T](...)`) per ruff UP047.
Result: 142/142 tests, 0 ruff findings, 0 pyright errors.

## Cycle 7 (statistics: sign-flip, bootstrap, Holm)

`pytest -q`, `ruff format`, `ruff check --fix`, `pyright` after landing `statistics/{sign_flip,
bootstrap,holm}.py`. Found and fixed a genuine test-expectation bug in the sign-flip test (two-sided
extremeness matches both the all-positive AND all-negative sign assignment when the observed mean is a
unique extremum, not just one — fixed 1/1024 -> 2/1024). Result: 159/159 tests, 0 ruff findings, 0
pyright errors.

## Cycle 8 (final calibration, heterogeneity diagnostic)

`pytest -q`, `ruff format`, `ruff check --fix`, `pyright` after landing `calibration/
final_calibration.py` and `evaluation/heterogeneity.py`. Result: 168/168 tests, 0 ruff findings, 0
pyright errors.

## Cycle 9 (real N-BaIoT ingestion — first real-data-touching code)

`pytest -q`, `ruff format`, `ruff check --fix`, `pyright` after landing `data/nbaiot_reader.py`
(standalone pandas-based CSV reader, no external stack dependency). Installed `pandas-stubs` for
typing. Manual smoke check against the real symlinked raw data
(`data/raw/N-BaIoT/Danmini_Doorbell`): benign shape (49548, 115) and 115 features exactly match the
roadmap-published table and section 21's 115-feature claim; all 10 attack subtype files (5 bashlite +
5 mirai) read successfully with plausible row counts. Result: 172/172 tests, 0 ruff findings, 0 pyright
errors.

## Cycle 10 (real-data integration test, all 9 clients)

Added `tests/data/test_nbaiot_reader_integration.py`: reads all 9 real N-BaIoT devices'
`benign_traffic.csv` and asserts shape `(published_row_count, 115)` for every one — all 9 pass,
confirming `CLIENT-001`/`DATASET-001` exactly against real data (not just the config table). Marked
`@pytest.mark.integration` (real I/O, ~70s) and excluded from the default `pytest` run via
`addopts = "-m 'not integration'"` in `pyproject.toml`, keeping the fast unit suite fast (172 tests,
~3s) while the integration check remains available on demand or at major checkpoints. Result: fast
suite 172/172 in 3s; integration suite 9/9 in ~69s; ruff/pyright clean.

## Cycle 11 (feature manifest freeze)

`pytest -q` (both default and `-m integration`), `ruff format`, `ruff check --fix`, `pyright` after
landing `data/feature_manifest.py` (`FeatureManifest`, `build_feature_manifest_from_csv_header`) plus
a real-data integration test confirming all 9 N-BaIoT devices share an identical 115-column feature
manifest (same names, same order, same sha256). Result: fast suite 176/176 (3s), integration suite
18/18 (~57s), ruff/pyright clean.

## Cycle 12 (standalone detector substrate: preprocessing, autoencoder, FedAvg training)

`pytest -q`, `ruff format`, `ruff check --fix`, `pyright` after landing `data/preprocessing.py`
(z-score `FeatureScaler` fit on TRAIN only), `detector/model.py` (fixed autoencoder,
`reconstruction_error_scores`), `detector/training.py` (FedAvg: local training + row-count-weighted
parameter averaging). Fixed a `zip(seq, seq[1:], strict=True)` bug (always raises since the two
sequences necessarily differ in length by one — same class of bug caught earlier in the calibration
tests) in `Autoencoder.__init__`'s layer-pair construction. Narrow `pyright: ignore[
reportUnknownMemberType]` on two lines where torch's own stubs are incomplete (`optimizer.step()`,
`torch.manual_seed`) — not a blanket suppression, torch simply doesn't fully type these. Result:
189/189 fast tests, 18/18 integration tests unaffected, 0 ruff findings, 0 pyright errors. Training
verified to reduce reconstruction error and to be deterministic given a fixed seed on tiny synthetic
data; full real 10-seed x 9-client training against actual N-BaIoT is the next step and will take real
wall-clock time.

## Cycle 13 (score generation: wires ingestion -> partitioning -> preprocessing -> model -> ScoreArtifact)

`pytest -q`, `ruff format`, `ruff check --fix`, `pyright` after landing `scoring/score_generation.py`
(`generate_score_artifact`), the first module that ties together `nbaiot_reader`, `partitioner`,
`preprocessing`, `detector.model`, and `schemas.score_artifact` end to end. Result: 193/193 fast tests,
18/18 integration tests unaffected, 0 ruff findings, 0 pyright errors.

## Cycle 14 (end-to-end smoke pipeline against real data)

`scripts/smoke_pipeline.py`: runs the FULL pipeline (real N-BaIoT ingestion -> partitioning ->
per-client TRAIN-only scaler fitting -> FedAvg training -> score generation -> AUROC) against 3 real
clients (Danmini, Ennio, Ecobee), subsampled to 400 rows/file for speed. Completed in 29.5s. Results:
score artifacts generated with correct record counts and distinct hashes per client; smoke AUROC
0.910-1.000 across the three clients — plausible and encouraging (benign-trained autoencoder separates
attack traffic well), a scientific sanity signal beyond mere structural correctness. `pytest -q`,
`ruff check`, `pyright` re-run after: 193/193 fast tests, 0 ruff findings, 0 pyright errors — the
smoke script did not touch any tested module's behavior, only exercised it.

## Cycle 15 (frozen detector hyperparameter config)

`pytest -q`, `ruff format`, `ruff check --fix`, `pyright` after landing `config/detector.yaml` +
`config/detector.py` (`DetectorHyperparameters`, `load_detector_hyperparameters`,
`load_detector_seeds`). Renamed `protocol.py`'s private `_read_yaml` to public `read_yaml_mapping`
since `detector.py` now needs to share it (avoiding the private-cross-module-import smell caught
earlier in the MILP optimizer work). Result: 196/196 tests, 0 ruff findings, 0 pyright errors.

## Cycle 16 (result row schema)

`pytest -q`, `ruff format`, `ruff check --fix`, `pyright` after landing `schemas/result.py`
(`ResultRow`, `WeightMode`, `SolverStatus` — the primary per-client result schema). Result: 200/200
tests, 0 ruff findings, 0 pyright errors.

## Cycle 17 (communication overhead accounting)

`pytest -q`, `ruff format`, `ruff check --fix`, `pyright` after landing `evaluation/workload.py`
(`ClientUploadPayload`, `federation_upload_bytes`, `candidate_index_bits/bytes`). Tests reproduce the
roadmap's exact published byte counts (896 bytes/client, 8,064 for 9 clients, 94,080 for 105 clients,
8-bit candidate index for 207 candidates) exactly. Result: 206/206 tests, 0 ruff findings, 0 pyright
errors.

## Cycle 18 (T01/SPLIT-004 closed end-to-end)

Added `tests/audit/test_split_leakage_integration.py`: generates a real `ScoreArtifact` via the full
scoring pipeline and confirms `check_partition_exclusivity` finds zero cross-partition `sample_id`
collisions across all 6 splits (4 benign + 2 attack). Closes `SPLIT-004`/`TEST-T01` from
boundary-arithmetic-only evidence to full pipeline evidence. `pytest -q`, `ruff format`,
`ruff check --fix`, `pyright`: 207/207 tests, 0 ruff findings, 0 pyright errors.

## Cycle 19 (real full-scale seed-0 training run — major scientific validation)

Ran `scripts/run_seed_training.py 0` at FULL scale (no subsampling) against all 9 real N-BaIoT clients:
completed in 298.8s (~5 min). Total generated score records across all 9 clients: 7,062,606 — matches
the roadmap's published "~7.06 million sequential observations" (section 21) almost exactly, an
independent real-data confirmation of the dataset identity claim (DATASET-001) beyond the per-file
counts already verified. Persisted 9 pickled `ScoreArtifact`s (616MB total) + `manifest.json` with
per-client sha256 hashes to `results/scores/seed_0/`.

Launched `scripts/run_all_seeds.py` (resumable: skips seeds with an existing manifest) in the
background to complete the remaining 9 seeds, estimated ~45 additional minutes based on the seed-0
timing. `pytest -q` (207/207), `ruff`, `pyright` all still clean — the orchestration scripts are
outside the package (`scripts/`) and don't affect the tested library surface.

## Cycle 20 (conservative/LCB utility curves)

`pytest -q`, `ruff format`, `ruff check --fix`, `pyright` after landing `frontier/conservative.py`
(one-sided Clopper-Pearson LCB, `conservative_client_utility`, `build_conservative_utility_curve`).
Generalized `frontier/utility.py:client_utility` to accept a `SubtypeRecallSource` Protocol
(`true_positive_rate() -> float`) instead of the concrete `SubtypeConfusionCounts` type, so both raw
and LCB-adjusted recall sources reuse the same averaging function rather than duplicating it. Result:
214/214 tests, 0 ruff findings, 0 pyright errors.

## Cycle 21 (allocation-sensitivity analysis)

`pytest -q`, `ruff format`, `ruff check --fix`, `pyright` after landing `frontier/stability.py`
(`summarize_client_stability` — modal/median/5th/95th percentile + `Instability_k`;
`run_allocation_sensitivity` — generic replicate-driven aggregator, caller supplies the
resample-and-reallocate closure). Result: 220/220 tests, 0 ruff findings, 0 pyright errors.

Follow-up from user feedback mid-session: renamed opaque `i1/i2/i3/n` boundary fields to descriptive
names (`train_end`/`frontier_end`/`final_cal_end`/`total_rows`), replaced hardcoded split-fraction
module constants with a typed `Protocol`/`BenignSplitFractions`/`AttackSplitFraction` loader reading
the single canonical `protocol.yaml`, added `RowCount`/`RowIndex` `NewType`s to avoid raw-int primitive
leakage, and removed all "roadmap section N" references from code/test docstrings and comments
(kept only in `docs/` where such references belong).
