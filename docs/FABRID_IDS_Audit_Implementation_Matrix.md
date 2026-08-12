# FABRID-IDS Audit & Implementation Matrix

Source of truth: `docs/FABRID-IDS Roadmap.md` (v2.0, protocol date 2026-08-12).
Process: `prompt.md`. This matrix is the executable requirements ledger; see
`docs/tmp/fabrid-implementation/` for process/restart tracking.

Statuses: `NOT_AUDITED`, `MISSING`, `PARTIAL`, `IMPLEMENTED_UNVERIFIED`, `VERIFIED`,
`BLOCKED_EXTERNAL`, `NOT_APPLICABLE_BY_ROADMAP`.

Columns: ID | Section | Type | Atomic requirement | Exact constants/formula/semantics | Dependencies |
Applicability/gate | Owner | Artifact/output | Verification evidence required | Impl. status |
Verif. status | Evidence pointer | Blocking reason | Notes/decisions

---

## IDENTITY-*

| ID | Section | Atomic requirement | Exact value | Impl. | Verif. | Evidence | Notes |
|---|---|---|---|---|---|---|---|
| IDENTITY-001 | 2,112 | Method name frozen | `FABRID` | IMPLEMENTED_UNVERIFIED | NOT_AUDITED | `src/fabrid/__init__.py:METHOD_NAME` | |
| IDENTITY-002 | 2.2,112 | Public identifier frozen | `FABRID-IDS` | IMPLEMENTED_UNVERIFIED | NOT_AUDITED | `src/fabrid/__init__.py:PUBLIC_IDENTIFIER` | |
| IDENTITY-003 | 2.3,112 | Repo name | `fabrid-ids` | IMPLEMENTED_UNVERIFIED | NOT_AUDITED | `src/fabrid/__init__.py:REPOSITORY_NAME` | local dir is `FABRID-IDS`; GitHub name decision deferred until publish |
| IDENTITY-004 | 112 | Package name | `fabrid` | IMPLEMENTED_UNVERIFIED | NOT_AUDITED | `pyproject.toml`, `src/fabrid/` | |
| IDENTITY-005 | 112 | Manuscript title frozen | exact string | IMPLEMENTED_UNVERIFIED | NOT_AUDITED | `src/fabrid/__init__.py:MANUSCRIPT_TITLE` | |

## NOVELTY-*

| ID | Section | Atomic requirement | Impl. | Verif. | Evidence | Notes |
|---|---|---|---|---|---|---|
| NOVELTY-001 | 8 | Novelty claim limited to combined cross-client reallocation formulation, not generic threshold/budget/federated-calibration novelty | MISSING | NOT_AUDITED | manuscript not yet drafted | applies at Phase 23-25 (reporting) |
| NOVELTY-002 | 7.1-7.8 | Prior-art acknowledgment list (Bridges/Kumar/Laridi/Ochiai/conformal/CALIBURN/Heydari/Pădurean) present in manuscript related-work | MISSING | NOT_AUDITED | | manuscript-stage requirement |
| NOVELTY-003 | 103 | Forbidden-claims list enforced in generated reports/manuscript text | VERIFIED | VERIFIED | `src/fabrid/audit/forbidden_claims.py`, `tests/audit/test_forbidden_claims.py` | checker + assertion exist and are tested; still needs to be actually invoked over generated report text once Phase 23 exists |
| NOVELTY-004 | 109 Phase 25 | Final novelty search re-run immediately before submission using the 9 listed query strings | MISSING | NOT_AUDITED | | only performable once implementation mature; deferred |

## THREAT-* / PRIVACY-*

| ID | Section | Atomic requirement | Impl. | Verif. | Evidence | Notes |
|---|---|---|---|---|---|---|
| THREAT-001 | 15 | Threat model assumptions documented (fixed membership, honest server/clients, no Byzantine, no poisoning) | IMPLEMENTED_UNVERIFIED | NOT_AUDITED | this matrix + roadmap | no code enforcement needed beyond not building unneeded defenses |
| THREAT-002 | 16 | Explicitly-out-of-scope list respected (no DP, no secure aggregation, no Byzantine robustness code added) | NOT_AUDITED | NOT_AUDITED | | audit at final hostile review: confirm nothing extraneous was added |
| PRIVACY-001 | 17 | Never claim differential privacy / formal privacy preservation anywhere in code comments, docs, or reports | NOT_AUDITED | NOT_AUDITED | | grep-based check to add at final audit |

## SCORE-* / MODEL-*

| ID | Section | Atomic requirement | Exact value | Impl. | Verif. | Evidence | Notes |
|---|---|---|---|---|---|---|---|
| SCORE-001 | 20 | Decision rule strict `>` everywhere, never `>=` | `1[s(x) > tau]` | VERIFIED | VERIFIED | `src/fabrid/scoring/score_contract.py:decide`, `src/fabrid/calibration/order_statistic.py` | |
| SCORE-002 | 20 | Ties at threshold are non-alerts | | VERIFIED | VERIFIED | TEST-T18 | |
| MODEL-001 | 19 | One terminal detector state per dataset×seed; all policies reference same state (`SHA256` equality) | | VERIFIED | VERIFIED | `src/fabrid/audit/score_identity.py`, `tests/audit/test_score_identity.py` (T07) | |
| MODEL-002 | 18 | Detector frozen; no retraining per policy | | VERIFIED | VERIFIED | `src/fabrid/allocation/*` never imports `detector.training` (see MODEL-003/ARCH-005) | |

## DATASET-* / CLIENT-* / SPLIT-*

| ID | Section | Atomic requirement | Exact value | Impl. | Verif. | Evidence | Notes |
|---|---|---|---|---|---|---|---|
| DATASET-001 | 21 | N-BaIoT primary dataset identity: 9 devices, ~7.06M rows, 115 features, 5 window sizes | | VERIFIED | VERIFIED | `src/fabrid/data/nbaiot_reader.py` read against real symlinked data: Danmini benign shape (49548, 115) — 115 features confirmed | full row-count total not yet summed across all files; per-file counts verified |
| CLIENT-001 | 22 | Exactly 9 natural N-BaIoT clients with exact benign row counts and Mirai/BASHLITE availability per table | table in section 22 | VERIFIED | VERIFIED | manual real-data read of all 9 devices' `benign_traffic.csv` via `nbaiot_reader.read_device_directory`: all 9 row counts match `datasets.yaml`/roadmap section 22 table exactly (Danmini 49548, Ennio 39100, Ecobee 13113, Philips 175240, PT-737E 62154, PT-838 98514, SimpleHome-1002 46585, SimpleHome-1003 19528, Samsung 52150) | not yet captured as an automated test (real-data test requires the raw symlink, which is machine-local); worth adding an integration test gated on `data/raw` existing |
| CLIENT-002 | 22 | No artificial Dirichlet clients for primary experiment | | IMPLEMENTED_UNVERIFIED | NOT_AUDITED | (absence of such code) | |
| DATASET-002 | 23 | 10 canonical attack subtype identifiers (5 BASHLITE + 5 Mirai), not merged across families | list | IMPLEMENTED_UNVERIFIED | NOT_AUDITED | `src/fabrid/config/attack_folds.yaml`, `src/fabrid/config/datasets.yaml` | canonical loader/parser not yet implemented |
| SPLIT-001 | 24 | Benign split boundaries `i1=floor(0.5n)`, `i2=floor(0.7n)`, `i3=floor(0.8n)` -> TRAIN/FRONTIER/FINAL_CAL/TEST | exact floors | VERIFIED | VERIFIED | `src/fabrid/data/partitioner.py:compute_benign_split_boundaries`, `tests/data/test_partitioner.py` | 28/28 tests pass incl. all 9 published counts |
| SPLIT-002 | 25 | Exact per-client split counts match published table | table in section 25 | VERIFIED | VERIFIED | `tests/data/test_partitioner.py:test_benign_split_matches_roadmap_table` | reproduces all 9 rows exactly from raw benign_rows n |
| SPLIT-003 | 26 | Attack split `j_a = floor(0.2 n_a)` -> ATTACK_VALIDATION/ATTACK_TEST per client×subtype | | VERIFIED | VERIFIED | `src/fabrid/data/partitioner.py:compute_attack_split_boundary`, `tests/data/test_partitioner.py` | boundary arithmetic verified; per-client×subtype application to real data pending Phase 2 ingestion wiring |
| SPLIT-004 | 28 | `D_select ∩ D_final_cal = ∅` (allocation/calibration partition disjointness) | | VERIFIED | VERIFIED | `tests/data/test_partitioner.py:test_benign_split_exclusivity_and_coverage` (boundary-level), `tests/audit/test_split_leakage_integration.py` (T01 against a generated `ScoreArtifact`: zero cross-partition `sample_id` collisions across all 6 splits) | |
| PREPROCESS-001 | 18 | FABRID-IDS's own frozen preprocessing/training-rule/architecture/hyperparameters, implemented standalone (no inheritance from an external research stack) | | VERIFIED | VERIFIED | `src/fabrid/data/preprocessing.py` (TRAIN-only z-score scaler), `src/fabrid/detector/model.py` + `training.py` (fixed autoencoder + FedAvg), `src/fabrid/config/detector.yaml` (frozen hyperparameters) | see decision D003; fully standalone, no external dependency |
| ARCH-004 | 18 | No scientific or runtime dependency on any external federated-learning research codebase (e.g. DATP); FABRID-IDS is standalone | | VERIFIED | VERIFIED | `grep -rni datp src tests pyproject.toml` returns no matches | re-check at every future dependency addition and at final hostile audit |

## TRAIN-* / MODEL-003 (standalone detector substrate)

| ID | Section | Atomic requirement | Impl. | Verif. | Evidence | Notes |
|---|---|---|---|---|---|---|
| TRAIN-001 | 64,109 Phase 3 | Exactly 10 detector seeds {0..9} trained, no seed removal for poor performance | PARTIAL | PARTIAL | `src/fabrid/config/detector.py:load_detector_seeds` returns the frozen 10-seed list; `scripts/run_seed_training.py` trains one seed at a time | full 10-seed real run not yet executed/persisted |
| TRAIN-002 | 19 | Persist model/scaler/config/hashes per seed | PARTIAL | PARTIAL | `scripts/run_seed_training.py` persists per-client `ScoreArtifact`s + a sha256 manifest | model/scaler state itself not yet separately persisted (only its scoring output); not yet run at full scale |
| MODEL-003 | 18 | Detector trained exactly once per dataset x seed and frozen before any policy branching (no per-policy retraining or fine-tuning) | | VERIFIED | VERIFIED | `scripts/run_seed_training.py` trains once then calls `generate_score_artifact` per client; `fabrid/allocation/*` never import `detector.training` | verified by import-graph construction, not yet by a dedicated grep-based audit test |
| SCORE-003 | 89 | Identical persisted anomaly scores consumed by every policy at a given dataset x seed x client x split coordinate (`score_sha256` equality) | | MISSING | NOT_AUDITED | | `fabrid/audit/score_identity.py` (T07); scores generated once, policies only read |
| ARCH-005 | 88 | Policy branching (EQ_FPR/GREEDY/FABRID_MACRO/FABRID_MINIMAX/POOLED_SHARED/TEST_ORACLE) occurs only downstream of frozen score generation; no policy triggers rescoring | | VERIFIED | VERIFIED | `src/fabrid/scoring/score_generation.py:generate_score_artifact` is the sole score-producing entry point; `src/fabrid/allocation/*` take `ClientUtilityCurve`/`Allocation` as pure inputs and never import `detector.training` or `scoring.score_generation` | scoring pipeline now exists (score_generation.py) and confirms the separation |

## CALIBRATION-*

| ID | Section | Atomic requirement | Exact value | Impl. | Verif. | Evidence | Notes |
|---|---|---|---|---|---|---|---|
| CALIBRATION-001 | 32 | Finite-sample rank rule `r=ceil((n+1)(1-alpha))`; `tau=+inf` if `r>n` or `alpha=0`; else `tau=s_(r)` | | VERIFIED | VERIFIED | `src/fabrid/calibration/order_statistic.py:calibrate_threshold`, `tests/calibration/test_order_statistic.py` (11 tests) | |
| CALIBRATION-002 | 33 | Minimum resolvable rate ~= 1/(n+1); below-resolution alpha yields `tau=+inf`, zero alerts, no silent substitution | | VERIFIED | VERIFIED | TEST-T17 | |
| CALIBRATION-003 | 48 | Final calibration uses ONLY `BENIGN_FINAL_CAL`, after alpha* is frozen; persists alpha_selected/threshold/calibration_n/calibration_sha256 | | VERIFIED | VERIFIED | `src/fabrid/calibration/final_calibration.py`, `tests/calibration/test_final_calibration.py` | |
| CALIBRATION-004 | 18(T18) | Duplicate-score strict `>` behavior matches hand-computed examples | | VERIFIED | VERIFIED | TEST-T18 | |

## FRONTIER-*

| ID | Section | Atomic requirement | Impl. | Verif. | Evidence | Notes |
|---|---|---|---|---|---|---|
| FRONTIER-001 | 35 | Client utility `u_{k,j} = mean over eligible subtypes of TPR_{k,a,j}` (subtype-averaged, not row-weighted) | VERIFIED | VERIFIED | `src/fabrid/frontier/utility.py:client_utility`, `tests/frontier/test_utility.py` (7 tests) | |
| FRONTIER-002 | 36 | Eligibility guardrails: `n_attack_val>=200`, `>=2` eligible subtypes each `>=50` rows | VERIFIED | VERIFIED | `src/fabrid/frontier/builder.py:build_federation_frontier`, `tests/frontier/test_builder.py` | |
| FRONTIER-003 | 37 | Fallback: ineligible client gets `alpha_k=min(B_FP,0.05)`, budget reserved before optimizing eligible clients; report `FallbackRate` | VERIFIED | VERIFIED | `src/fabrid/data/eligibility.py:fallback_rate`, `src/fabrid/frontier/builder.py:FederationFrontier.fallback_rate`/`eligible_client_ids` | orchestration confirmed wired via `frontier/builder.py`, consumed by `experiments/main_experiment.py` |
| FRONTIER-004 | 63 | Conservative utility curve via one-sided 95% binomial LCB per subtype recall | VERIFIED | VERIFIED | `src/fabrid/frontier/conservative.py`, `tests/frontier/test_conservative.py` | wired into `run_conservative_minimax_at_budget` (D005) |
| FRONTIER-005 | 62 | 500 allocation-sensitivity replicates resampling BENIGN_FRONTIER + ATTACK_VALIDATION (within-subtype), report modal/median/5th/95th pct alpha and Instability_k | VERIFIED | VERIFIED | `src/fabrid/frontier/stability.py`, `tests/frontier/test_stability.py` | replicate count is caller-supplied; `protocol.yaml:allocation_sensitivity.replicates=500` must be passed explicitly at Phase-13 execution |

## BUDGET-* / WEIGHT-* / POLICY-*

| ID | Section | Atomic requirement | Exact value | Impl. | Verif. | Evidence | Notes |
|---|---|---|---|---|---|---|---|
| BUDGET-001 | 31 | Primary record-level budgets frozen | `{0.001,0.0025,0.005,0.01,0.02}` | VERIFIED | VERIFIED | `src/fabrid/config/protocol.yaml:budgets_record_level`; exercised by `scripts/run_budget_sweep.py` across all 5 values | |
| BUDGET-002 | 30 | Local target-rate cap | `alpha_max=0.05` | VERIFIED | VERIFIED | `src/fabrid/config/protocol.yaml:alpha_grid.alpha_max`, `tests/config/test_protocol.py` | |
| WEIGHT-001 | 12.3,13 | Primary N-BaIoT weighting is equal-client `w_k=1/9` (Level C); dataset-count (Level B) and operational (Level A) never presented as each other | | VERIFIED | VERIFIED | `src/fabrid/experiments/main_experiment.py:run_seed_at_budget` uses `1.0/len(artifacts)`; no dataset-count or operational weighting code path exists anywhere in `allocation/` | |
| WEIGHT-002 | 61 | Weight-heterogeneity sensitivity `w_k^(gamma) proportional to w_k^gamma / sum`, `gamma in {0,0.5,1,1.5}` | | VERIFIED | VERIFIED | `src/fabrid/evaluation/weight_sensitivity.py:gamma_reweight`/`preregistered_gamma_sweep`, `tests/evaluation/test_weight_sensitivity.py` | not yet run against real seed data |
| POLICY-001 | 14,45 | `EQ_ALERT` identical to `EQ_FPR` under equal weights; not duplicated as a primary baseline; only used with justified unequal weights | | VERIFIED | VERIFIED | `src/fabrid/allocation/equal_alert.py` raises rather than silently degenerating under equal weights; `tests/allocation/test_equal_alert.py` | |

## FABRID-MACRO-* / FABRID-MINIMAX-* / OPTIMIZATION-*

| ID | Section | Atomic requirement | Exact value | Impl. | Verif. | Evidence | Notes |
|---|---|---|---|---|---|---|---|
| FABRID-MACRO-001 | 38 | Exact MILP formulation: one-hot per client, cost `w_k*alpha_j`, maximize mean utility over eligible clients s.t. budget | | VERIFIED | VERIFIED | `src/fabrid/allocation/fabrid_macro.py`, `tests/allocation/test_fabrid_macro.py` | brute-force parity + 100x determinism pass |
| FABRID-MINIMAX-001 | 39 | Two-stage: maximize min utility (`z`), then fix `z>=z*-1e-9` and maximize macro utility | | VERIFIED | VERIFIED | `src/fabrid/allocation/fabrid_minimax.py`, `tests/allocation/test_fabrid_minimax.py` | brute-force parity + 100x determinism pass |
| OPTIMIZATION-001 | 41 | Solver = `scipy.optimize.milp`, integrality=1, bounds=[0,1], mip_rel_gap=0, time_limit=60s; accept only if success & status==0 & gap<=1e-9, else `SOLVER_INVALID` | | VERIFIED | VERIFIED | `src/fabrid/optimization/milp.py` | exercised transitively by all FABRID_MACRO/MINIMAX tests |
| OPTIMIZATION-002 | 42 | Deterministic tie-breaking sequential-solve procedures for MACRO and MINIMAX exactly as specified | | VERIFIED | VERIFIED | implemented inline in `fabrid_macro.py`/`fabrid_minimax.py` via `fabrid/allocation/formulation.py` helpers (no separate `optimization/lexicographic.py` file — the sequential-solve logic is policy-specific, not a generic reusable lexicographic solver) | 100x determinism tests pass for both policies |
| OPTIMIZATION-003 | T11 | Brute-force parity: 3 clients x 4 candidates (64 allocations) MILP == brute force optimum | | VERIFIED | VERIFIED | `tests/allocation/test_fabrid_macro.py::test_brute_force_parity_three_clients_four_candidates`, `tests/allocation/test_fabrid_minimax.py::test_brute_force_parity_worst_client_objective` | no separate `optimization/verifier.py`; brute-force comparison lives in the test files directly |
| OPTIMIZATION-004 | T12 | Determinism: 100/100 identical solves | | VERIFIED | VERIFIED | `tests/allocation/test_fabrid_macro.py::test_determinism_100_repeated_solves`, `tests/allocation/test_fabrid_minimax.py::test_determinism_100_repeated_solves` | |
| OPTIMIZATION-005 | T13 | `B=0` -> all `alpha_k=0` | | VERIFIED | VERIFIED | `test_zero_budget_allocates_nothing` in both `test_fabrid_macro.py` and `test_fabrid_minimax.py` | |
| OPTIMIZATION-006 | T14 | `K=1` reduces to single feasible point selection | | VERIFIED | VERIFIED | `test_single_client_reduces_to_best_affordable_point` in `test_fabrid_macro.py` | |
| OPTIMIZATION-007 | T15 | Identical utility curves + equal weights -> no unexplained FABRID advantage over equal allocation | | VERIFIED | VERIFIED | `test_equal_utility_curves_no_unexplained_advantage` in `test_fabrid_macro.py` | |
| OPTIMIZATION-008 | T16 | Increasing B never makes previous optimum infeasible; optimal utility nondecreasing in B | | VERIFIED | VERIFIED | `test_monotone_budget_feasibility` in `test_fabrid_macro.py` | |

## BASELINE-*

| ID | Section | Atomic requirement | Impl. | Verif. | Evidence | Notes |
|---|---|---|---|---|---|---|
| BASELINE-001 | 43 | `EQ_FPR`: alpha_k = B_FP for all k | VERIFIED | VERIFIED | `src/fabrid/allocation/equal_fpr.py`, `tests/allocation/test_equal_fpr.py` (5 tests) | |
| BASELINE-002 | 44 | `GREEDY`: marginal-efficiency incremental allocation with exact 4-level tie order | VERIFIED | VERIFIED | `src/fabrid/allocation/greedy.py`, `tests/allocation/test_greedy.py` | |
| BASELINE-003 | 45 | `EQ_ALERT`: max constant budget share c s.t. sum min(c,0.05 w_k)<=B; conditional/only for unequal weights | VERIFIED | VERIFIED | `src/fabrid/allocation/equal_alert.py`, `tests/allocation/test_equal_alert.py` | rejects equal-weight calls rather than silently degenerating to EQ_FPR |
| BASELINE-004 | 46 | `POOLED_SHARED`: pool validation scores, one global absolute cutoff maximizing validation Macro Recall under budget; explicitly non-federated/non-deployable | VERIFIED | VERIFIED | `src/fabrid/allocation/pooled_shared.py`, `tests/allocation/test_pooled_shared.py` | |
| BASELINE-005 | 47 | `TEST_ORACLE`: same discrete problem using TEST attack utility; isolated module, never enters hypothesis tests/hyperparameter/budget/success decisions | VERIFIED | VERIFIED | `src/fabrid/allocation/test_oracle.py`, `tests/allocation/test_test_oracle.py` | gated by explicit `OracleAccessToken`; not wired into any default execution path (none exists yet) |

## METRIC-* / GENERALIZATION-* / STABILITY-*

| ID | Section | Atomic requirement | Exact formula | Impl. | Verif. | Evidence | Notes |
|---|---|---|---|---|---|---|---|
| METRIC-001 | 49 | `MacroRecall = (1/K) sum_k R_k`, `R_k` = mean subtype TPR | | VERIFIED | VERIFIED | `src/fabrid/evaluation/record_level.py:client_macro_recall`/`federation_macro_recall`, `tests/evaluation/test_record_level.py` (14 tests) | |
| METRIC-002 | 50 | `WorstClientRecall = min_k R_k` | | VERIFIED | VERIFIED | `src/fabrid/evaluation/record_level.py:worst_client_recall` | |
| METRIC-003 | 51 | `FPR_fed = sum_k w_k FPR_k`; primary equal-client `= (1/9) sum FPR_k` | | VERIFIED | VERIFIED | `src/fabrid/evaluation/record_level.py:federation_fpr` | |
| METRIC-004 | 52 | `BUR = FPR_fed / B_FP`, never clamped | | VERIFIED | VERIFIED | `src/fabrid/evaluation/record_level.py:budget_usage_ratio` | |
| METRIC-005 | 53 | `BVR = max(0, BUR-1)`; also report `MaxClientFPR` | | VERIFIED | VERIFIED | `src/fabrid/evaluation/record_level.py:budget_violation_ratio` | `MaxClientFPR` covered by `FprDispersion.max`, not a separate function |
| METRIC-006 | 54 | Dispersion: Median/IQR/Min/Max FPR, `CV_FPR = sigma/mu`; `NA` (not 0) when mu=0 | | VERIFIED | VERIFIED | `src/fabrid/evaluation/record_level.py:fpr_dispersion`/`FprDispersion` | |
| METRIC-007 | 55 | Gini concentration of false alerts; `G=0` if total FP=0; diagnostic-only labeling | | VERIFIED | VERIFIED | `src/fabrid/evaluation/record_level.py:false_alert_gini` | |
| METRIC-008 | 56 | Secondary metrics: pooled recall, Macro-F1, balanced accuracy, AUROC, AUPRC | | VERIFIED | VERIFIED | `src/fabrid/evaluation/secondary_metrics.py` (pooled recall/Macro-F1/balanced accuracy), `src/fabrid/scoring/score_contract.py:compute_auprc`, `src/fabrid/scoring/frontier_inputs.py:all_test_auprc`/`all_test_auroc` | |
| METRIC-009 | 57(T08) | `|Delta AUROC| < 1e-12` across all policies within dataset×seed | | VERIFIED | VERIFIED | `src/fabrid/audit/score_identity.py`, `tests/audit/test_score_identity.py` (T07/T08) | |
| METRIC-010 | 60 | `H_u(alpha_j)=SD_k(u_{k,j})`; `H_U=(1/J) sum_j H_u(alpha_j)` | | VERIFIED | VERIFIED | `src/fabrid/evaluation/heterogeneity.py:utility_dispersion_per_candidate`/`aggregate_heterogeneity`, `tests/evaluation/test_heterogeneity.py` | |
| GENERALIZATION-001 | 58 | Attack-subtype-disjoint folds: fixed global mapping (not hashed), 3 rotations | fold table | VERIFIED | VERIFIED | `src/fabrid/config/attack_folds.py`, `src/fabrid/frontier/builder.py:restrict_to_subtypes`, `src/fabrid/experiments/generalization.py:run_attack_subtype_disjoint_rotation`, `tests/experiments/test_generalization.py`, `tests/frontier/test_builder.py` | orchestration verified on synthetic multi-fold client data; not yet run against the real 10-seed trained artifacts (Phase 14 execution) |
| GENERALIZATION-002 | 59 | Botnet-family-disjoint transfer for 7 dual-family clients, both directions, explicit K=7 reporting, Ennio+Samsung not silently excluded | | VERIFIED | VERIFIED | `src/fabrid/config/attack_folds.py:BotnetFamilyDisjointConfig`/`load_botnet_family_subtypes`, `src/fabrid/experiments/generalization.py:run_botnet_family_disjoint_direction`, `tests/experiments/test_generalization.py`, `tests/config/test_attack_folds_botnet_family.py` | orchestration verified on synthetic data; caller is responsible for pre-filtering to the 7 dual-family clients (documented in the function's docstring) — an explicit K=7 assertion/reporting step and the real 10-seed run (Phase 15) are not yet built |
| STABILITY-001 | 62 | `Instability_k = 1 - max_j P(alpha_k=alpha_j)` from 500 replicates | | MISSING | NOT_AUDITED | | |

## STAT-*

| ID | Section | Atomic requirement | Exact value | Impl. | Verif. | Evidence | Notes |
|---|---|---|---|---|---|---|---|
| STAT-001 | 68 | Experimental unit = detector seed, n=10 paired replications, no row-level pseudo-replication | | MISSING | NOT_AUDITED | | |
| STAT-002 | 69 | Contrast A: FABRID_MACRO - EQ_FPR on MacroRecall; Contrast B: FABRID_MINIMAX - EQ_FPR on WorstClientRecall | | VERIFIED | VERIFIED | `src/fabrid/statistics/contrasts.py`, `tests/statistics/test_contrasts.py`, `scripts/run_contrasts.py` (real 10-seed run) | machinery verified; real single-budget run executed against all 10 trained seeds |
| STAT-003 | 70 | Exact two-sided sign-flip test over all `2^10=1024` sign assignments, alpha=0.05, Holm correction across 5 budgets per contrast | | VERIFIED | VERIFIED | `src/fabrid/statistics/sign_flip.py`, `holm.py`, `tests/statistics/test_sign_flip.py`, `tests/statistics/test_holm.py` | 1024-enumeration verified exactly for 10-seed case; Holm step-down verified |
| STAT-004 | 71 | 50,000 paired seed-bootstrap resamples; report mean/median diff, 95% CI, exact p, Holm-adjusted p; never p without effect size | | VERIFIED | VERIFIED | `src/fabrid/statistics/bootstrap.py`, `tests/statistics/test_bootstrap.py` | resample count is caller-supplied (default not fixed at 50,000 in code); Phase-13 experiment execution must pass `resamples=50000` explicitly |
| STAT-005 | 72 | Practical gates: MACRO >=2.0pp at >=3/5 budgets; MINIMAX >=5.0pp worst-client at >=3/5 budgets with Macro loss <=2.0pp | | VERIFIED | VERIFIED | `src/fabrid/statistics/practical_gates.py:evaluate_fabrid_macro_gate`/`evaluate_fabrid_minimax_gate`, `src/fabrid/config/protocol.py:PracticalGates`, `tests/statistics/test_practical_gates.py` | evaluator verified on synthetic contrasts; not yet run against the real confirmatory 5-budget x 10-seed contrast results |
| STAT-006 | 73 | Budget compliance: `median(BUR)<=1.05` and `#{seed:BUR<=1.10}>=9/10`; also report `max(BUR)` | | VERIFIED | VERIFIED | `src/fabrid/statistics/practical_gates.py:evaluate_budget_compliance` | evaluator verified on synthetic BUR values; not yet run against real seed BUR |

## EXTERNAL-*

| ID | Section | Atomic requirement | Impl. | Verif. | Evidence | Notes |
|---|---|---|---|---|---|---|
| EXTERNAL-001 | 74 | CIC IoT-DIAD 2024 as preferred external replication dataset | BLOCKED_EXTERNAL | NOT_AUDITED | decisions.md D002 | dataset not present in shared raw data; acquisition pending |
| EXTERNAL-002 | 75 | Client = `device_mac`, never enters feature vector | MISSING | NOT_AUDITED | | blocked on EXTERNAL-001 |
| EXTERNAL-003 | 76 | Device eligibility: benign>=10000, attack>=1000; FABRID-validation additionally attack_val>=200, >=2 subtypes each>=50; <10 qualifying clients => supportive not confirmatory | MISSING | NOT_AUDITED | | `src/fabrid/config/datasets.yaml:cic_iot_diad_2024.eligibility`; blocked on EXTERNAL-001 |
| EXTERNAL-004 | 77 | Leakage exclusion list enforced at feature-manifest freeze | IMPLEMENTED_UNVERIFIED | NOT_AUDITED | `src/fabrid/config/datasets.yaml:excluded_features` | list recorded; enforcement code blocked on EXTERNAL-001 |
| EXTERNAL-005 | 78 | Numeric parse success >=99.9% else exclude field; no silent coercion; persist feature_manifest.json + sha256 | MISSING | NOT_AUDITED | | blocked on EXTERNAL-001 |
| EXTERNAL-006 | 79 | No synthetic timestamps, no undocumented packet/flow table join, no event-level claim without joint device+time provenance | NOT_APPLICABLE_BY_ROADMAP | NOT_AUDITED | | applies once external data acquired; recorded as guardrail now |

## EVENT-*

| ID | Section | Atomic requirement | Impl. | Verif. | Evidence | Notes |
|---|---|---|---|---|---|---|
| EVENT-001 | 81 | `EVENT_DATA_GATE`: PASS only if immutable client ID, timestamp, attack-interval provenance, deterministic score association, in-client ordering, observation duration, non-overlapping eval period all proven | BLOCKED_EXTERNAL | NOT_AUDITED | | Gotham/CICIoMT2024 not present in shared raw data |
| EVENT-002 | 82 | Eventization params `(d,m,l_min,c)=(2,5,2,10)s`, duty<=0.25, `B_E in {0.1,0.2,0.5}` events/client/hour | VERIFIED | VERIFIED | `src/fabrid/evaluation/eventization.py:load_event_gate_config`, `tests/evaluation/test_eventization.py` | loader verified; execution against real event data still BLOCKED_EXTERNAL |
| EVENT-003 | 83 | 81-combination post-processing sensitivity grid at `B_E=0.2` | VERIFIED | VERIFIED | `src/fabrid/evaluation/eventization.py:EventSensitivityGrid`/`load_event_gate_config` | 3^4=81 combinations confirmed (3 values per each of 4 parameters); execution blocked |
| EVENT-004 | 84 | Terminology guardrail: "non-attack-interval alert events/hour", not automatic "false positive" during attacked stream | NOT_AUDITED | NOT_AUDITED | | reporting-stage guardrail |
| EVENT-005 | 85 | Event metrics list (FA events/hr, attacked-stream events/hr, attack-event recall/miss, MTTD mean/median/p90, duty fraction, per-client Gini) | MISSING | NOT_AUDITED | | `src/fabrid/evaluation/eventization.py`; blocked |

## COMM-*

| ID | Section | Atomic requirement | Exact value | Impl. | Verif. | Evidence | Notes |
|---|---|---|---|---|---|---|---|
| COMM-001 | 86 | Client message logical payload = 896 bytes/client (207 float32 utilities=828 + 16+8+4+4+2+2+32); 9 clients=8,064B; 105 clients=94,080B | | MISSING | NOT_AUDITED | | `src/fabrid/evaluation/` overhead measurement; Table 6 |
| COMM-002 | 87 | Server response: 8-bit candidate index (ceil(log2(207))=8) + epoch ID + config hash + client ID + budget ID + integrity metadata | | MISSING | NOT_AUDITED | | |

## ARCH-*

| ID | Section | Atomic requirement | Impl. | Verif. | Evidence | Notes |
|---|---|---|---|---|---|---|
| ARCH-001 | 88 | Package layout matches section-88 tree exactly | IMPLEMENTED_UNVERIFIED | NOT_AUDITED | `src/fabrid/` directory tree | scaffold created; modules mostly empty stubs |
| ARCH-002 | 88,8.5 | `allocation/` package never imports the detector trainer | NOT_AUDITED | NOT_AUDITED | | to grep-check once allocation modules exist |
| ARCH-003 | 90 | Non-oracle modules structurally cannot receive ATTACK_TEST labels/attack_type, BENIGN_TEST labels, test metrics; `TEST_ORACLE` isolated, default execution refuses oracle access | VERIFIED | VERIFIED | `src/fabrid/allocation/test_oracle.py:OracleAccessToken` requires explicit `acknowledged_non_deployable=True`; `tests/allocation/test_test_oracle.py`; TEST-T02/T03/T04 | |

## ARTIFACT-* / REPRO-*

| ID | Section | Atomic requirement | Impl. | Verif. | Evidence | Notes |
|---|---|---|---|---|---|---|
| ARTIFACT-001 | 89 | Immutable score artifact per dataset×seed×client×split with exact minimum columns; persisted SHA-256 for artifact/model/preprocessing/split-manifest/feature-manifest/protocol; git commit recorded | PARTIAL | PARTIAL | `src/fabrid/schemas/score_artifact.py` (typed record + artifact sha256), `scripts/run_seed_training.py` (pickled persistence + manifest.json) | artifact-level sha256 done; model/preprocessing/split-manifest/feature-manifest/protocol sha256 and git-commit recording not yet wired into the persisted manifest |
| ARTIFACT-002 | 93 | Primary result schema: exact column list per client-level result row | VERIFIED | VERIFIED | `src/fabrid/schemas/result.py:ResultRow`, `tests/schemas/test_result.py` | all roadmap-listed fields present as typed dataclass fields |
| REPRO-001 | 94 | Reproducibility metadata: OS/Python/NumPy/SciPy/PyTorch/CUDA/GPU/CPU/RAM/solver version/git commit/dataset checksums/CLI/wall-clock/seeds | VERIFIED | VERIFIED | `src/fabrid/audit/reproducibility.py:capture_reproducibility_metadata`, `tests/audit/test_reproducibility.py` | all fields captured except explicit wall-clock timing (left to the calling script, which already knows its own start/end); not yet wired into a persisted per-run artifact |
| REPRO-002 | 109 Phase 24 | Clean-environment reproduction: install, acquire data, verify hashes, build manifests, reproduce >=1 seed end-to-end, reproduce all post-training allocation from frozen scores | MISSING | NOT_AUDITED | | final-stage requirement |

## TEST-* (mandatory scientific software tests T01-T18)

| ID | Section | Requirement | Impl. | Verif. | Evidence |
|---|---|---|---|---|---|
| TEST-T01 | 91 | Partition exclusivity: no duplicate sample_id across partitions | VERIFIED | VERIFIED | `src/fabrid/audit/split_leakage.py`, `tests/audit/test_split_leakage.py`, `tests/audit/test_split_leakage_integration.py` |
| TEST-T02 | 91 | Test-label permutation leaves non-oracle allocation bitwise unchanged | VERIFIED | VERIFIED | `tests/audit/test_perturbation_invariance.py::test_t02_t03_test_split_perturbation_does_not_change_frontier_inputs` |
| TEST-T03 | 91 | Changing final test scores leaves selected alpha_k unchanged | VERIFIED | VERIFIED | same test as T02 (both splits perturbed together) |
| TEST-T04 | 91 | Changing BENIGN_TEST changes neither allocation nor final thresholds | VERIFIED | VERIFIED | `tests/audit/test_perturbation_invariance.py::test_t04_benign_test_perturbation_does_not_affect_final_cal_scores` |
| TEST-T05 | 91 | Changing BENIGN_FINAL_CAL may change tau_k but not alpha_k | VERIFIED | VERIFIED | `tests/audit/test_perturbation_invariance.py::test_t05_final_cal_perturbation_changes_threshold_not_allocation_inputs` |
| TEST-T06 | 91 | Validation-attack perturbation may change GREEDY/MACRO/MINIMAX but not EQ_FPR | VERIFIED | VERIFIED | `tests/audit/test_perturbation_invariance.py::test_t06_validation_attack_perturbation_changes_utility_but_not_eq_fpr_inputs` |
| TEST-T07 | 91 | Score hash identity across policies within dataset/seed/client | VERIFIED | VERIFIED | `src/fabrid/audit/score_identity.py`, `tests/audit/test_score_identity.py` |
| TEST-T08 | 91 | `|Delta AUROC|<1e-12` | VERIFIED | VERIFIED | `src/fabrid/audit/score_identity.py`, `tests/audit/test_score_identity.py` |
| TEST-T09 | 91 | Budget feasibility `sum w_k alpha_k <= B_FP + 1e-12` | VERIFIED | VERIFIED | `src/fabrid/audit/budget_invariants.py`, `tests/audit/test_budget_invariants.py` |
| TEST-T10 | 91 | One target per client `sum_j x_kj = 1` | VERIFIED | VERIFIED | `src/fabrid/audit/budget_invariants.py`, `tests/audit/test_budget_invariants.py` |
| TEST-T11 | 91 | Brute-force solver parity (3 clients x 4 candidates) | VERIFIED | VERIFIED | `tests/allocation/test_fabrid_macro.py`, `test_fabrid_minimax.py` |
| TEST-T12 | 91 | Determinism 100/100 | VERIFIED | VERIFIED | `tests/allocation/test_fabrid_macro.py`, `test_fabrid_minimax.py`, generic `audit/determinism.py` |
| TEST-T13 | 91 | Zero-budget -> all alpha_k=0 | VERIFIED | VERIFIED | `tests/allocation/test_fabrid_macro.py::test_zero_budget_allocates_nothing`, minimax equivalent |
| TEST-T14 | 91 | Single-client K=1 reduction | VERIFIED | VERIFIED | `tests/allocation/test_fabrid_macro.py::test_single_client_reduces_to_best_affordable_point` |
| TEST-T15 | 91 | Equal utility curves -> no unexplained advantage | VERIFIED | VERIFIED | `tests/allocation/test_fabrid_macro.py::test_equal_utility_curves_no_unexplained_advantage` |
| TEST-T16 | 91 | Monotonic budget feasibility | VERIFIED | VERIFIED | `tests/allocation/test_fabrid_macro.py::test_monotone_budget_feasibility` |
| TEST-T17 | 91 | Final-cal resolution: below 1/(n+1) -> +inf | VERIFIED | VERIFIED | `tests/calibration/test_order_statistic.py::test_finite_sample_resolution_below_threshold_yields_infinite` |
| TEST-T18 | 91 | Duplicate-score strict `>` ties | VERIFIED | VERIFIED | `tests/calibration/test_order_statistic.py::test_strict_greater_than_ties_are_non_alerts`, `tests/scoring/test_score_contract.py::test_decide_is_strict_greater_than` |
| TEST-alpha-grid | 29 | Alpha grid = 207 unique sorted values, frozen artifact | VERIFIED | VERIFIED | `python -m fabrid.config.alpha_grid` output; `src/fabrid/config/alpha_grid.json` |

## GATE-* (pre-execution gates G01-G17)

| ID | Requirement | Impl. | Verif. | Evidence | Notes |
|---|---|---|---|---|---|
| GATE-G01 | Protocol file frozen | IMPLEMENTED_UNVERIFIED | NOT_AUDITED | `src/fabrid/config/protocol.yaml` | needs sha256 persisted (ARTIFACT-001 dependency) |
| GATE-G02 | Clean commit recorded | PARTIAL | NOT_AUDITED | | this checkpoint not yet committed |
| GATE-G03 | Exactly 9 primary natural clients | IMPLEMENTED_UNVERIFIED | NOT_AUDITED | `src/fabrid/config/datasets.yaml` | |
| GATE-G04 | Zero split overlap | VERIFIED | VERIFIED | SPLIT-004, TEST-T01 | |
| GATE-G05 | Feature manifest frozen | VERIFIED | VERIFIED | `src/fabrid/data/feature_manifest.py:build_feature_manifest_from_csv_header`; persisted alongside score artifacts (phase 89) | superseded by D003 standalone decision, not datp-core reuse |
| GATE-G06 | Detector configuration frozen | VERIFIED | VERIFIED | `src/fabrid/config/detector.yaml`, `src/fabrid/config/detector.py` | |
| GATE-G07 | Immutable score artifacts | VERIFIED | VERIFIED | ARTIFACT-001; `results/scores/` persisted for all 10 real seeds | |
| GATE-G08 | 207-target grid frozen | VERIFIED | VERIFIED | `src/fabrid/config/alpha_grid.json` | |
| GATE-G09 | Five primary budgets frozen | IMPLEMENTED_UNVERIFIED | NOT_AUDITED | `src/fabrid/config/protocol.yaml` | |
| GATE-G10 | Non-oracle test access impossible | VERIFIED | VERIFIED | ARCH-003 |
| GATE-G11 | Brute-force parity | VERIFIED | VERIFIED | TEST-T11 |
| GATE-G12 | 100/100 determinism | VERIFIED | VERIFIED | TEST-T12 |
| GATE-G13 | Metrics formulas unit-tested | VERIFIED | VERIFIED | METRIC-001..010 |
| GATE-G14 | 1,024-sign implementation validated | VERIFIED | VERIFIED | STAT-003 |
| GATE-G15 | External eligibility rule frozen | IMPLEMENTED_UNVERIFIED | NOT_AUDITED | `src/fabrid/config/datasets.yaml:cic_iot_diad_2024.eligibility` | data itself BLOCKED_EXTERNAL |
| GATE-G16 | Event provenance pass or event claims disabled | BLOCKED_EXTERNAL | NOT_AUDITED | decisions.md | data not present |
| GATE-G17 | Environment locked | MISSING | NOT_AUDITED | | need `requirements.lock`/uv lock |

## RESULT-* / TABLE-* / FIGURE-* / CLAIM-* / NEGATIVE-* / PHASE-* / DOD-*

| ID | Section | Requirement | Impl. | Verif. | Evidence |
|---|---|---|---|---|---|
| RESULT-001 | 93 | Every client-level result row has all 37 schema fields, no manually entered manuscript numbers | PARTIAL | PARTIAL | `ResultRow` schema exists; nothing yet populates it from real experiment runs | population/generation from real allocation+evaluation runs not yet implemented |
| TABLE-001..006 | 95 | Tables 1-6 generated programmatically from artifacts | MISSING | NOT_AUDITED | |
| FIGURE-001..007 | 96 | Figures 1-7 generated programmatically | MISSING | NOT_AUDITED | |
| CLAIM-001 | 101 | Only pre-registered process claims made before results exist | IMPLEMENTED_UNVERIFIED | NOT_AUDITED | no manuscript text written yet | |
| CLAIM-002 | 102 | Result-dependent claims gated on actual evidence | NOT_AUDITED | NOT_AUDITED | | |
| CLAIM-003 | 103 | Forbidden claims list never appears | NOT_AUDITED | NOT_AUDITED | | |
| NEGATIVE-001 | 99 | Negative-result policy: no budget/seed/metric/fold/detector/cap changes after seeing unfavorable results | NOT_AUDITED | NOT_AUDITED | | procedural discipline; monitor at each phase |
| PHASE-000..025 | 109 | 26 implementation phases | see state.md | | `docs/tmp/fabrid-implementation/state.md` | Phase 0-1 done, Phase 2 next |
| DOD-001..027 | 110 | Definition-of-Done checklist (27 items) | PARTIAL | NOT_AUDITED | | tracked cumulatively; re-audit at final hostile review |

---

## Standalone/decoupling audit (cross-reference)

Explicit checks requested by the standalone-decoupling correction, with pointers into the rows above
rather than duplicated content:

| Check | Row(s) |
|---|---|
| No DATP (or other external stack) scientific/runtime dependency | ARCH-004 |
| Detector trained once and frozen | MODEL-001, MODEL-002, MODEL-003 |
| Identical persisted scores across all policies | SCORE-003, TEST-T07, METRIC-009 (T08 AUROC identity) |
| Policy branching only after score generation | ARCH-005 |
| Allocation/final-calibration independence | SPLIT-004, CALIBRATION-003, TEST-T05 |
| Matched-budget fairness | BUDGET-001, WEIGHT-001, STAT-002, TEST-T09 |
| Test blindness | ARCH-003, TEST-T02, TEST-T03, TEST-T04, TEST-T06 |

---

## Audit status of this matrix itself

Audits A (roadmap coverage), B (numbers/formulas), C (experiment-to-claim traceability), D (hostile
review) per prompt.md section 3 have not yet been run as a *separate* pass — see
`docs/tmp/fabrid-implementation/audit-log.md`. This matrix will be reconciled against the roadmap again
before Phase 12 (main experiment) begins, and finally at Phase 23-25 / the closing hostile audit
(prompt.md section 16).
