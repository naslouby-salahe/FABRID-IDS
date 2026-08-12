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
| IDENTITY-001 | 2,112 | Method name frozen | `FABRID` | IMPLEMENTED_UNVERIFIED | NOT_AUDITED | `fabrid/__init__.py:METHOD_NAME` | |
| IDENTITY-002 | 2.2,112 | Public identifier frozen | `FABRID-IDS` | IMPLEMENTED_UNVERIFIED | NOT_AUDITED | `fabrid/__init__.py:PUBLIC_IDENTIFIER` | |
| IDENTITY-003 | 2.3,112 | Repo name | `fabrid-ids` | IMPLEMENTED_UNVERIFIED | NOT_AUDITED | `fabrid/__init__.py:REPOSITORY_NAME` | local dir is `FABRID-IDS`; GitHub name decision deferred until publish |
| IDENTITY-004 | 112 | Package name | `fabrid` | IMPLEMENTED_UNVERIFIED | NOT_AUDITED | `pyproject.toml`, `fabrid/` | |
| IDENTITY-005 | 112 | Manuscript title frozen | exact string | IMPLEMENTED_UNVERIFIED | NOT_AUDITED | `fabrid/__init__.py:MANUSCRIPT_TITLE` | |

## NOVELTY-*

| ID | Section | Atomic requirement | Impl. | Verif. | Evidence | Notes |
|---|---|---|---|---|---|---|
| NOVELTY-001 | 8 | Novelty claim limited to combined cross-client reallocation formulation, not generic threshold/budget/federated-calibration novelty | MISSING | NOT_AUDITED | manuscript not yet drafted | applies at Phase 23-25 (reporting) |
| NOVELTY-002 | 7.1-7.8 | Prior-art acknowledgment list (Bridges/Kumar/Laridi/Ochiai/conformal/CALIBURN/Heydari/Pădurean) present in manuscript related-work | MISSING | NOT_AUDITED | | manuscript-stage requirement |
| NOVELTY-003 | 103 | Forbidden-claims list enforced in generated reports/manuscript text | MISSING | NOT_AUDITED | | should be a lint/check over generated report text at Phase 23 |
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
| SCORE-001 | 20 | Decision rule strict `>` everywhere, never `>=` | `1[s(x) > tau]` | MISSING | NOT_AUDITED | | to enforce in `fabrid/calibration/order_statistic.py` and `fabrid/allocation/*` |
| SCORE-002 | 20 | Ties at threshold are non-alerts | | MISSING | NOT_AUDITED | | test T18 |
| MODEL-001 | 19 | One terminal detector state per dataset×seed; all policies reference same state (`SHA256` equality) | | MISSING | NOT_AUDITED | | requires `fabrid/audit/score_identity.py` (T07) |
| MODEL-002 | 18 | Detector frozen; no retraining per policy | | MISSING | NOT_AUDITED | | architectural constraint: allocation package must not import trainer (section 88) |

## DATASET-* / CLIENT-* / SPLIT-*

| ID | Section | Atomic requirement | Exact value | Impl. | Verif. | Evidence | Notes |
|---|---|---|---|---|---|---|---|
| DATASET-001 | 21 | N-BaIoT primary dataset identity: 9 devices, ~7.06M rows, 115 features, 5 window sizes | | IMPLEMENTED_UNVERIFIED | NOT_AUDITED | `fabrid/config/datasets.yaml` | descriptive only so far |
| CLIENT-001 | 22 | Exactly 9 natural N-BaIoT clients with exact benign row counts and Mirai/BASHLITE availability per table | table in section 22 | IMPLEMENTED_UNVERIFIED | NOT_AUDITED | `fabrid/config/datasets.yaml:nbaiot.clients` | |
| CLIENT-002 | 22 | No artificial Dirichlet clients for primary experiment | | IMPLEMENTED_UNVERIFIED | NOT_AUDITED | (absence of such code) | |
| DATASET-002 | 23 | 10 canonical attack subtype identifiers (5 BASHLITE + 5 Mirai), not merged across families | list | IMPLEMENTED_UNVERIFIED | NOT_AUDITED | `fabrid/config/attack_folds.yaml`, `fabrid/config/datasets.yaml` | canonical loader/parser not yet implemented |
| SPLIT-001 | 24 | Benign split boundaries `i1=floor(0.5n)`, `i2=floor(0.7n)`, `i3=floor(0.8n)` -> TRAIN/FRONTIER/FINAL_CAL/TEST | exact floors | MISSING | NOT_AUDITED | | `fabrid/data/partitioner.py` not yet written |
| SPLIT-002 | 25 | Exact per-client split counts match published table | table in section 25 | IMPLEMENTED_UNVERIFIED | NOT_AUDITED | `fabrid/config/datasets.yaml:split_counts` | must be verified against computed partitioner output once raw data is read (T-level check) |
| SPLIT-003 | 26 | Attack split `j_a = floor(0.2 n_a)` -> ATTACK_VALIDATION/ATTACK_TEST per client×subtype | | MISSING | NOT_AUDITED | | |
| SPLIT-004 | 28 | `D_select ∩ D_final_cal = ∅` (allocation/calibration partition disjointness) | | MISSING | NOT_AUDITED | | enforced by construction if partitioner is correct; needs explicit test (T01) |
| PREPROCESS-001 | 18 | Inherit frozen preprocessing/FedAvg/training rule/architecture/hyperparameters from existing DATP stack | | NOT_AUDITED | NOT_AUDITED | | see decision D001; must confirm datp-core contract before coding Phase 3 |

## TRAIN-* / (detector via datp-core dependency)

| ID | Section | Atomic requirement | Impl. | Verif. | Evidence | Notes |
|---|---|---|---|---|---|---|
| TRAIN-001 | 64,109 Phase 3 | Exactly 10 detector seeds {0..9} trained, no seed removal for poor performance | MISSING | NOT_AUDITED | | via datp-core, invoked from fabrid orchestration (not yet written) |
| TRAIN-002 | 19 | Persist model/scaler/config/hashes per seed | MISSING | NOT_AUDITED | | |

## CALIBRATION-*

| ID | Section | Atomic requirement | Exact value | Impl. | Verif. | Evidence | Notes |
|---|---|---|---|---|---|---|---|
| CALIBRATION-001 | 32 | Finite-sample rank rule `r=ceil((n+1)(1-alpha))`; `tau=+inf` if `r>n` or `alpha=0`; else `tau=s_(r)` | | MISSING | NOT_AUDITED | | `fabrid/calibration/order_statistic.py` |
| CALIBRATION-002 | 33 | Minimum resolvable rate ~= 1/(n+1); below-resolution alpha yields `tau=+inf`, zero alerts, no silent substitution | | MISSING | NOT_AUDITED | | test T17 |
| CALIBRATION-003 | 48 | Final calibration uses ONLY `BENIGN_FINAL_CAL`, after alpha* is frozen; persists alpha_selected/threshold/calibration_n/calibration_sha256 | | MISSING | NOT_AUDITED | | `fabrid/calibration/final_calibration.py` |
| CALIBRATION-004 | 18(T18) | Duplicate-score strict `>` behavior matches hand-computed examples | | MISSING | NOT_AUDITED | | test T18 |

## FRONTIER-*

| ID | Section | Atomic requirement | Impl. | Verif. | Evidence | Notes |
|---|---|---|---|---|---|---|
| FRONTIER-001 | 35 | Client utility `u_{k,j} = mean over eligible subtypes of TPR_{k,a,j}` (subtype-averaged, not row-weighted) | MISSING | NOT_AUDITED | | `fabrid/frontier/utility.py` |
| FRONTIER-002 | 36 | Eligibility guardrails: `n_attack_val>=200`, `>=2` eligible subtypes each `>=50` rows | MISSING | NOT_AUDITED | | `fabrid/frontier/builder.py` |
| FRONTIER-003 | 37 | Fallback: ineligible client gets `alpha_k=min(B_FP,0.05)`, budget reserved before optimizing eligible clients; report `FallbackRate` | MISSING | NOT_AUDITED | | |
| FRONTIER-004 | 63 | Conservative utility curve via one-sided 95% binomial LCB per subtype recall | MISSING | NOT_AUDITED | | `fabrid/frontier/conservative.py` |
| FRONTIER-005 | 62 | 500 allocation-sensitivity replicates resampling BENIGN_FRONTIER + ATTACK_VALIDATION (within-subtype), report modal/median/5th/95th pct alpha and Instability_k | MISSING | NOT_AUDITED | | `fabrid/frontier/stability.py` |

## BUDGET-* / WEIGHT-* / POLICY-*

| ID | Section | Atomic requirement | Exact value | Impl. | Verif. | Evidence | Notes |
|---|---|---|---|---|---|---|---|
| BUDGET-001 | 31 | Primary record-level budgets frozen | `{0.001,0.0025,0.005,0.01,0.02}` | IMPLEMENTED_UNVERIFIED | NOT_AUDITED | `fabrid/config/protocol.yaml:budgets_record_level` | |
| BUDGET-002 | 30 | Local target-rate cap | `alpha_max=0.05` | IMPLEMENTED_UNVERIFIED | NOT_AUDITED | `fabrid/config/protocol.yaml:alpha_grid.alpha_max` | |
| WEIGHT-001 | 12.3,13 | Primary N-BaIoT weighting is equal-client `w_k=1/9` (Level C); dataset-count (Level B) and operational (Level A) never presented as each other | | MISSING | NOT_AUDITED | | `fabrid/allocation/` weight handling |
| WEIGHT-002 | 61 | Weight-heterogeneity sensitivity `w_k^(gamma) proportional to w_k^gamma / sum`, `gamma in {0,0.5,1,1.5}` | | MISSING | NOT_AUDITED | | |
| POLICY-001 | 14,45 | `EQ_ALERT` identical to `EQ_FPR` under equal weights; not duplicated as a primary baseline; only used with justified unequal weights | | MISSING | NOT_AUDITED | | `fabrid/allocation/equal_alert.py` must guard against equal-weight misuse |

## FABRID-MACRO-* / FABRID-MINIMAX-* / OPTIMIZATION-*

| ID | Section | Atomic requirement | Exact value | Impl. | Verif. | Evidence | Notes |
|---|---|---|---|---|---|---|---|
| FABRID-MACRO-001 | 38 | Exact MILP formulation: one-hot per client, cost `w_k*alpha_j`, maximize mean utility over eligible clients s.t. budget | | MISSING | NOT_AUDITED | | `fabrid/allocation/fabrid_macro.py` |
| FABRID-MINIMAX-001 | 39 | Two-stage: maximize min utility (`z`), then fix `z>=z*-1e-9` and maximize macro utility | | MISSING | NOT_AUDITED | | `fabrid/allocation/fabrid_minimax.py` |
| OPTIMIZATION-001 | 41 | Solver = `scipy.optimize.milp`, integrality=1, bounds=[0,1], mip_rel_gap=0, time_limit=60s; accept only if success & status==0 & gap<=1e-9, else `SOLVER_INVALID` | | MISSING | NOT_AUDITED | | `fabrid/optimization/milp.py` |
| OPTIMIZATION-002 | 42 | Deterministic tie-breaking sequential-solve procedures for MACRO and MINIMAX exactly as specified | | MISSING | NOT_AUDITED | | `fabrid/optimization/lexicographic.py` |
| OPTIMIZATION-003 | T11 | Brute-force parity: 3 clients x 4 candidates (64 allocations) MILP == brute force optimum | | MISSING | NOT_AUDITED | | `fabrid/optimization/verifier.py` + test |
| OPTIMIZATION-004 | T12 | Determinism: 100/100 identical solves | | MISSING | NOT_AUDITED | | |
| OPTIMIZATION-005 | T13 | `B=0` -> all `alpha_k=0` | | MISSING | NOT_AUDITED | | |
| OPTIMIZATION-006 | T14 | `K=1` reduces to single feasible point selection | | MISSING | NOT_AUDITED | | |
| OPTIMIZATION-007 | T15 | Identical utility curves + equal weights -> no unexplained FABRID advantage over equal allocation | | MISSING | NOT_AUDITED | | |
| OPTIMIZATION-008 | T16 | Increasing B never makes previous optimum infeasible; optimal utility nondecreasing in B | | MISSING | NOT_AUDITED | | |

## BASELINE-*

| ID | Section | Atomic requirement | Impl. | Verif. | Evidence | Notes |
|---|---|---|---|---|---|---|
| BASELINE-001 | 43 | `EQ_FPR`: alpha_k = B_FP for all k | MISSING | NOT_AUDITED | | `fabrid/allocation/equal_fpr.py` |
| BASELINE-002 | 44 | `GREEDY`: marginal-efficiency incremental allocation with exact 4-level tie order | MISSING | NOT_AUDITED | | `fabrid/allocation/greedy.py` |
| BASELINE-003 | 45 | `EQ_ALERT`: max constant budget share c s.t. sum min(c,0.05 w_k)<=B; conditional/only for unequal weights | MISSING | NOT_AUDITED | | `fabrid/allocation/equal_alert.py` |
| BASELINE-004 | 46 | `POOLED_SHARED`: pool validation scores, one global absolute cutoff maximizing validation Macro Recall under budget; explicitly non-federated/non-deployable | MISSING | NOT_AUDITED | | `fabrid/allocation/pooled_shared.py` |
| BASELINE-005 | 47 | `TEST_ORACLE`: same discrete problem using TEST attack utility; isolated module, never enters hypothesis tests/hyperparameter/budget/success decisions | MISSING | NOT_AUDITED | | `fabrid/allocation/test_oracle.py`; must be structurally isolated per GATE T02/T-blind |

## METRIC-* / GENERALIZATION-* / STABILITY-*

| ID | Section | Atomic requirement | Exact formula | Impl. | Verif. | Evidence | Notes |
|---|---|---|---|---|---|---|---|
| METRIC-001 | 49 | `MacroRecall = (1/K) sum_k R_k`, `R_k` = mean subtype TPR | | MISSING | NOT_AUDITED | | `fabrid/evaluation/record_level.py` |
| METRIC-002 | 50 | `WorstClientRecall = min_k R_k` | | MISSING | NOT_AUDITED | | |
| METRIC-003 | 51 | `FPR_fed = sum_k w_k FPR_k`; primary equal-client `= (1/9) sum FPR_k` | | MISSING | NOT_AUDITED | | |
| METRIC-004 | 52 | `BUR = FPR_fed / B_FP`, never clamped | | MISSING | NOT_AUDITED | | |
| METRIC-005 | 53 | `BVR = max(0, BUR-1)`; also report `MaxClientFPR` | | MISSING | NOT_AUDITED | | |
| METRIC-006 | 54 | Dispersion: Median/IQR/Min/Max FPR, `CV_FPR = sigma/mu`; `NA` (not 0) when mu=0 | | MISSING | NOT_AUDITED | | |
| METRIC-007 | 55 | Gini concentration of false alerts; `G=0` if total FP=0; diagnostic-only labeling | | MISSING | NOT_AUDITED | | |
| METRIC-008 | 56 | Secondary metrics: pooled recall, Macro-F1, balanced accuracy, AUROC, AUPRC | | MISSING | NOT_AUDITED | | |
| METRIC-009 | 57(T08) | `|Delta AUROC| < 1e-12` across all policies within dataset×seed | | MISSING | NOT_AUDITED | | `fabrid/audit/score_identity.py` |
| METRIC-010 | 60 | `H_u(alpha_j)=SD_k(u_{k,j})`; `H_U=(1/J) sum_j H_u(alpha_j)` | | MISSING | NOT_AUDITED | | `fabrid/evaluation/heterogeneity.py` |
| GENERALIZATION-001 | 58 | Attack-subtype-disjoint folds: fixed global mapping (not hashed), 3 rotations | fold table | IMPLEMENTED_UNVERIFIED | NOT_AUDITED | `fabrid/config/attack_folds.yaml` | orchestration not yet built |
| GENERALIZATION-002 | 59 | Botnet-family-disjoint transfer for 7 dual-family clients, both directions, explicit K=7 reporting, Ennio+Samsung not silently excluded | | IMPLEMENTED_UNVERIFIED | NOT_AUDITED | `fabrid/config/attack_folds.yaml:botnet_family_disjoint` | orchestration not yet built |
| STABILITY-001 | 62 | `Instability_k = 1 - max_j P(alpha_k=alpha_j)` from 500 replicates | | MISSING | NOT_AUDITED | | |

## STAT-*

| ID | Section | Atomic requirement | Exact value | Impl. | Verif. | Evidence | Notes |
|---|---|---|---|---|---|---|---|
| STAT-001 | 68 | Experimental unit = detector seed, n=10 paired replications, no row-level pseudo-replication | | MISSING | NOT_AUDITED | | |
| STAT-002 | 69 | Contrast A: FABRID_MACRO - EQ_FPR on MacroRecall; Contrast B: FABRID_MINIMAX - EQ_FPR on WorstClientRecall | | MISSING | NOT_AUDITED | | |
| STAT-003 | 70 | Exact two-sided sign-flip test over all `2^10=1024` sign assignments, alpha=0.05, Holm correction across 5 budgets per contrast | | MISSING | NOT_AUDITED | | `fabrid/statistics/sign_flip.py`, `holm.py`; GATE G14 |
| STAT-004 | 71 | 50,000 paired seed-bootstrap resamples; report mean/median diff, 95% CI, exact p, Holm-adjusted p; never p without effect size | | MISSING | NOT_AUDITED | | `fabrid/statistics/bootstrap.py` |
| STAT-005 | 72 | Practical gates: MACRO >=2.0pp at >=3/5 budgets; MINIMAX >=5.0pp worst-client at >=3/5 budgets with Macro loss <=2.0pp | | MISSING | NOT_AUDITED | | evaluated post-experiment |
| STAT-006 | 73 | Budget compliance: `median(BUR)<=1.05` and `#{seed:BUR<=1.10}>=9/10`; also report `max(BUR)` | | MISSING | NOT_AUDITED | | |

## EXTERNAL-*

| ID | Section | Atomic requirement | Impl. | Verif. | Evidence | Notes |
|---|---|---|---|---|---|---|
| EXTERNAL-001 | 74 | CIC IoT-DIAD 2024 as preferred external replication dataset | BLOCKED_EXTERNAL | NOT_AUDITED | decisions.md D002 | dataset not present in shared raw data; acquisition pending |
| EXTERNAL-002 | 75 | Client = `device_mac`, never enters feature vector | MISSING | NOT_AUDITED | | blocked on EXTERNAL-001 |
| EXTERNAL-003 | 76 | Device eligibility: benign>=10000, attack>=1000; FABRID-validation additionally attack_val>=200, >=2 subtypes each>=50; <10 qualifying clients => supportive not confirmatory | MISSING | NOT_AUDITED | | `fabrid/config/datasets.yaml:cic_iot_diad_2024.eligibility`; blocked on EXTERNAL-001 |
| EXTERNAL-004 | 77 | Leakage exclusion list enforced at feature-manifest freeze | IMPLEMENTED_UNVERIFIED | NOT_AUDITED | `fabrid/config/datasets.yaml:excluded_features` | list recorded; enforcement code blocked on EXTERNAL-001 |
| EXTERNAL-005 | 78 | Numeric parse success >=99.9% else exclude field; no silent coercion; persist feature_manifest.json + sha256 | MISSING | NOT_AUDITED | | blocked on EXTERNAL-001 |
| EXTERNAL-006 | 79 | No synthetic timestamps, no undocumented packet/flow table join, no event-level claim without joint device+time provenance | NOT_APPLICABLE_BY_ROADMAP | NOT_AUDITED | | applies once external data acquired; recorded as guardrail now |

## EVENT-*

| ID | Section | Atomic requirement | Impl. | Verif. | Evidence | Notes |
|---|---|---|---|---|---|---|
| EVENT-001 | 81 | `EVENT_DATA_GATE`: PASS only if immutable client ID, timestamp, attack-interval provenance, deterministic score association, in-client ordering, observation duration, non-overlapping eval period all proven | BLOCKED_EXTERNAL | NOT_AUDITED | | Gotham/CICIoMT2024 not present in shared raw data |
| EVENT-002 | 82 | Eventization params `(d,m,l_min,c)=(2,5,2,10)s`, duty<=0.25, `B_E in {0.1,0.2,0.5}` events/client/hour | IMPLEMENTED_UNVERIFIED | NOT_AUDITED | `fabrid/config/protocol.yaml:event_gate` | params frozen; code + data blocked |
| EVENT-003 | 83 | 81-combination post-processing sensitivity grid at `B_E=0.2` | IMPLEMENTED_UNVERIFIED | NOT_AUDITED | `fabrid/config/protocol.yaml:event_gate.sensitivity_grid` | params frozen; execution blocked |
| EVENT-004 | 84 | Terminology guardrail: "non-attack-interval alert events/hour", not automatic "false positive" during attacked stream | NOT_AUDITED | NOT_AUDITED | | reporting-stage guardrail |
| EVENT-005 | 85 | Event metrics list (FA events/hr, attacked-stream events/hr, attack-event recall/miss, MTTD mean/median/p90, duty fraction, per-client Gini) | MISSING | NOT_AUDITED | | `fabrid/evaluation/eventization.py`; blocked |

## COMM-*

| ID | Section | Atomic requirement | Exact value | Impl. | Verif. | Evidence | Notes |
|---|---|---|---|---|---|---|---|
| COMM-001 | 86 | Client message logical payload = 896 bytes/client (207 float32 utilities=828 + 16+8+4+4+2+2+32); 9 clients=8,064B; 105 clients=94,080B | | MISSING | NOT_AUDITED | | `fabrid/evaluation/` overhead measurement; Table 6 |
| COMM-002 | 87 | Server response: 8-bit candidate index (ceil(log2(207))=8) + epoch ID + config hash + client ID + budget ID + integrity metadata | | MISSING | NOT_AUDITED | | |

## ARCH-*

| ID | Section | Atomic requirement | Impl. | Verif. | Evidence | Notes |
|---|---|---|---|---|---|---|
| ARCH-001 | 88 | Package layout matches section-88 tree exactly | IMPLEMENTED_UNVERIFIED | NOT_AUDITED | `fabrid/` directory tree | scaffold created; modules mostly empty stubs |
| ARCH-002 | 88,8.5 | `allocation/` package never imports the detector trainer | NOT_AUDITED | NOT_AUDITED | | to grep-check once allocation modules exist |
| ARCH-003 | 90 | Non-oracle modules structurally cannot receive ATTACK_TEST labels/attack_type, BENIGN_TEST labels, test metrics; `TEST_ORACLE` isolated, default execution refuses oracle access | MISSING | NOT_AUDITED | | typed API boundary, not just discipline; tests T02 |

## ARTIFACT-* / REPRO-*

| ID | Section | Atomic requirement | Impl. | Verif. | Evidence | Notes |
|---|---|---|---|---|---|---|
| ARTIFACT-001 | 89 | Immutable score artifact per dataset×seed×client×split with exact minimum columns; persisted SHA-256 for artifact/model/preprocessing/split-manifest/feature-manifest/protocol; git commit recorded | MISSING | NOT_AUDITED | | `fabrid/schemas/score_artifact.py` |
| ARTIFACT-002 | 93 | Primary result schema: exact column list (37 fields) per client-level result row | MISSING | NOT_AUDITED | | `fabrid/schemas/result.py` |
| REPRO-001 | 94 | Reproducibility metadata: OS/Python/NumPy/SciPy/PyTorch/CUDA/GPU/CPU/RAM/solver version/git commit/dataset checksums/CLI/wall-clock/seeds | MISSING | NOT_AUDITED | | |
| REPRO-002 | 109 Phase 24 | Clean-environment reproduction: install, acquire data, verify hashes, build manifests, reproduce >=1 seed end-to-end, reproduce all post-training allocation from frozen scores | MISSING | NOT_AUDITED | | final-stage requirement |

## TEST-* (mandatory scientific software tests T01-T18)

| ID | Section | Requirement | Impl. | Verif. | Evidence |
|---|---|---|---|---|---|
| TEST-T01 | 91 | Partition exclusivity: no duplicate sample_id across partitions | MISSING | NOT_AUDITED | |
| TEST-T02 | 91 | Test-label permutation leaves non-oracle allocation bitwise unchanged | MISSING | NOT_AUDITED | |
| TEST-T03 | 91 | Changing final test scores leaves selected alpha_k unchanged | MISSING | NOT_AUDITED | |
| TEST-T04 | 91 | Changing BENIGN_TEST changes neither allocation nor final thresholds | MISSING | NOT_AUDITED | |
| TEST-T05 | 91 | Changing BENIGN_FINAL_CAL may change tau_k but not alpha_k | MISSING | NOT_AUDITED | |
| TEST-T06 | 91 | Validation-attack perturbation may change GREEDY/MACRO/MINIMAX but not EQ_FPR | MISSING | NOT_AUDITED | |
| TEST-T07 | 91 | Score hash identity across policies within dataset/seed/client | MISSING | NOT_AUDITED | |
| TEST-T08 | 91 | `|Delta AUROC|<1e-12` | MISSING | NOT_AUDITED | |
| TEST-T09 | 91 | Budget feasibility `sum w_k alpha_k <= B_FP + 1e-12` | MISSING | NOT_AUDITED | |
| TEST-T10 | 91 | One target per client `sum_j x_kj = 1` | MISSING | NOT_AUDITED | |
| TEST-T11 | 91 | Brute-force solver parity (3 clients x 4 candidates) | MISSING | NOT_AUDITED | |
| TEST-T12 | 91 | Determinism 100/100 | MISSING | NOT_AUDITED | |
| TEST-T13 | 91 | Zero-budget -> all alpha_k=0 | MISSING | NOT_AUDITED | |
| TEST-T14 | 91 | Single-client K=1 reduction | MISSING | NOT_AUDITED | |
| TEST-T15 | 91 | Equal utility curves -> no unexplained advantage | MISSING | NOT_AUDITED | |
| TEST-T16 | 91 | Monotonic budget feasibility | MISSING | NOT_AUDITED | |
| TEST-T17 | 91 | Final-cal resolution: below 1/(n+1) -> +inf | MISSING | NOT_AUDITED | |
| TEST-T18 | 91 | Duplicate-score strict `>` ties | MISSING | NOT_AUDITED | |
| TEST-alpha-grid | 29 | Alpha grid = 207 unique sorted values, frozen artifact | VERIFIED | VERIFIED | `python -m fabrid.config.alpha_grid` output; `fabrid/config/alpha_grid.json` |

## GATE-* (pre-execution gates G01-G17)

| ID | Requirement | Impl. | Verif. | Evidence | Notes |
|---|---|---|---|---|---|
| GATE-G01 | Protocol file frozen | IMPLEMENTED_UNVERIFIED | NOT_AUDITED | `fabrid/config/protocol.yaml` | needs sha256 persisted (ARTIFACT-001 dependency) |
| GATE-G02 | Clean commit recorded | PARTIAL | NOT_AUDITED | | this checkpoint not yet committed |
| GATE-G03 | Exactly 9 primary natural clients | IMPLEMENTED_UNVERIFIED | NOT_AUDITED | `fabrid/config/datasets.yaml` | |
| GATE-G04 | Zero split overlap | MISSING | NOT_AUDITED | | depends on SPLIT-* + TEST-T01 |
| GATE-G05 | Feature manifest frozen | MISSING | NOT_AUDITED | | depends on datp-core preprocessing reuse decision |
| GATE-G06 | Detector configuration frozen | MISSING | NOT_AUDITED | | |
| GATE-G07 | Immutable score artifacts | MISSING | NOT_AUDITED | | |
| GATE-G08 | 207-target grid frozen | VERIFIED | VERIFIED | `fabrid/config/alpha_grid.json` | |
| GATE-G09 | Five primary budgets frozen | IMPLEMENTED_UNVERIFIED | NOT_AUDITED | `fabrid/config/protocol.yaml` | |
| GATE-G10 | Non-oracle test access impossible | MISSING | NOT_AUDITED | | ARCH-003 |
| GATE-G11 | Brute-force parity | MISSING | NOT_AUDITED | | TEST-T11 |
| GATE-G12 | 100/100 determinism | MISSING | NOT_AUDITED | | TEST-T12 |
| GATE-G13 | Metrics formulas unit-tested | MISSING | NOT_AUDITED | | METRIC-* |
| GATE-G14 | 1,024-sign implementation validated | MISSING | NOT_AUDITED | | STAT-003 |
| GATE-G15 | External eligibility rule frozen | IMPLEMENTED_UNVERIFIED | NOT_AUDITED | `fabrid/config/datasets.yaml:cic_iot_diad_2024.eligibility` | data itself BLOCKED_EXTERNAL |
| GATE-G16 | Event provenance pass or event claims disabled | BLOCKED_EXTERNAL | NOT_AUDITED | decisions.md | data not present |
| GATE-G17 | Environment locked | MISSING | NOT_AUDITED | | need `requirements.lock`/uv lock |

## RESULT-* / TABLE-* / FIGURE-* / CLAIM-* / NEGATIVE-* / PHASE-* / DOD-*

| ID | Section | Requirement | Impl. | Verif. | Evidence |
|---|---|---|---|---|---|
| RESULT-001 | 93 | Every client-level result row has all 37 schema fields, no manually entered manuscript numbers | MISSING | NOT_AUDITED | |
| TABLE-001..006 | 95 | Tables 1-6 generated programmatically from artifacts | MISSING | NOT_AUDITED | |
| FIGURE-001..007 | 96 | Figures 1-7 generated programmatically | MISSING | NOT_AUDITED | |
| CLAIM-001 | 101 | Only pre-registered process claims made before results exist | IMPLEMENTED_UNVERIFIED | NOT_AUDITED | no manuscript text written yet | |
| CLAIM-002 | 102 | Result-dependent claims gated on actual evidence | NOT_AUDITED | NOT_AUDITED | | |
| CLAIM-003 | 103 | Forbidden claims list never appears | NOT_AUDITED | NOT_AUDITED | | |
| NEGATIVE-001 | 99 | Negative-result policy: no budget/seed/metric/fold/detector/cap changes after seeing unfavorable results | NOT_AUDITED | NOT_AUDITED | | procedural discipline; monitor at each phase |
| PHASE-000..025 | 109 | 26 implementation phases | see state.md | | `docs/tmp/fabrid-implementation/state.md` | Phase 0-1 done, Phase 2 next |
| DOD-001..027 | 110 | Definition-of-Done checklist (27 items) | PARTIAL | NOT_AUDITED | | tracked cumulatively; re-audit at final hostile review |

---

## Audit status of this matrix itself

Audits A (roadmap coverage), B (numbers/formulas), C (experiment-to-claim traceability), D (hostile
review) per prompt.md section 3 have not yet been run as a *separate* pass — see
`docs/tmp/fabrid-implementation/audit-log.md`. This matrix will be reconciled against the roadmap again
before Phase 12 (main experiment) begins, and finally at Phase 23-25 / the closing hostile audit
(prompt.md section 16).
