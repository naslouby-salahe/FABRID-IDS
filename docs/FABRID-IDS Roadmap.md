# FABRID-IDS: Federated Alert-Budget Reallocation for Heterogeneous IoT Intrusion Detection

## Scientific Specification, Implementation Protocol, Adversarial Audit, and Publication Roadmap

**Method name:** FABRID
**Method expansion:** **F**ederated **A**lert-**B**udget **R**eallocation for **I**ntrusion **D**etection
**Public/search-facing identifier:** **FABRID-IDS**
**Manuscript title:** **FABRID-IDS: Federated Alert-Budget Reallocation for Heterogeneous IoT Intrusion Detection**
**Recommended GitHub repository:** `fabrid-ids`
**Recommended Python package:** `fabrid`
**Recommended CLI:** `fabrid`
**Artifact prefix:** `fabrid_`
**Document status:** Pre-registration-grade scientific and engineering specification
**Version:** 2.0
**Protocol date:** 12 August 2026
**Research type:** Federated IoT anomaly-detection decision-layer methodology
**Primary contribution class:** Cross-client operating-point allocation under a federation-wide nominal false-alert constraint
**Primary experimental paradigm:** Frozen detector → frozen scores → local utility estimation → cross-client budget allocation → independent final calibration → matched-budget evaluation
**Publication objective:** Journal-quality research article, with a protocol sufficiently complete for reproducible implementation before confirmatory experiments begin

---

# 1. Document Purpose and Revision Basis

This document replaces the previous research specification and incorporates a new hostile review of:

1. naming and acronym collisions;
2. novelty against current literature;
3. mathematical formulation;
4. calibration validity;
5. budget semantics;
6. traffic-weight validity;
7. baseline fairness;
8. dataset feasibility;
9. attack-validation leakage;
10. temporal dependence;
11. event-level workload claims;
12. optimization determinism;
13. statistical inference;
14. software architecture;
15. reproducibility;
16. publication positioning.

The previous specification correctly established the central concept of allocating a common false-alert allowance across heterogeneous federated clients rather than treating every local threshold independently.

Several details are deliberately changed here because a deeper implementation audit exposed loopholes that could otherwise create reviewer objections or incorrect scientific claims.

---

# 2. Final Naming Decision

## 2.1 Core name

The research method shall be called:

> **FABRID — Federated Alert-Budget Reallocation for Intrusion Detection**

The word **reallocation** is deliberate.

A federation begins with a finite common alert allowance. FABRID reallocates that allowance across clients instead of giving every client the same operating rate.

---

## 2.2 Public disambiguation

The exact acronym `FABRID` is already used by an unrelated USENIX Security 2023 system:

> *FABRID: Flexible Attestation-Based Routing for Inter-Domain Networks.*

That work concerns SCION inter-domain routing and has no methodological relationship to this research, but the acronym collision is real.

Therefore the research shall use:

> **FABRID-IDS**

in:

* the manuscript title;
* GitHub repository name;
* README title;
* software releases;
* Zenodo/software archives;
* search keywords;
* presentation titles.

Within equations and method descriptions, the algorithm family may still be called **FABRID**.

---

## 2.3 Exact repository naming

Use:

```text
fabrid-ids
```

Do not use:

```text
fabrid
```

as the public repository name because it would collide semantically with the existing FABRID networking system.

Recommended repository description:

> **Reference implementation of FABRID-IDS: federated cross-client false-alert budget allocation for heterogeneous IoT anomaly detection.**

---

# 3. Executive Scientific Decision

**Decision: GO.**

However, the research is publishable only under the narrow novelty boundary defined in this document.

FABRID is **not**:

* a new anomaly detector;
* a new autoencoder;
* a new FL aggregation rule;
* a new federated-training algorithm;
* a new generic threshold estimator;
* the first alert-budget-aware IDS;
* the first constrained threshold optimizer;
* the first federated thresholding method;
* the first multi-detector alert-control mechanism;
* the first IDS resource-allocation formulation.

Its defensible contribution is:

> **A federated decision layer that reallocates one common nominal false-alert allowance across heterogeneous clients by selecting client-specific target operating rates from locally estimated detection–budget utility curves, while keeping the underlying anomaly detector fixed and calibrating final thresholds independently after the allocation has been selected.**

That formulation survives the literature audit substantially better than a generic “alert-budget allocation” claim.

---

# 4. Central Research Problem

Consider a federation of:

[
K
]

heterogeneous IoT clients.

A fixed trained anomaly detector produces a local anomaly score:

[
s_k(x).
]

Each client can operate at different false-positive-rate targets:

[
\alpha_k.
]

Increasing (\alpha_k) generally lowers its decision threshold and may increase attack detection, but also consumes more false-alert capacity.

The federation has one finite nominal allowance:

[
B_{\mathrm{FP}}.
]

The research problem is:

[
\boxed{
\text{How should } B_{\mathrm{FP}}
\text{ be distributed across clients?}
}
]

The generic constrained form is:

[
\boxed{
\sum_{k=1}^{K} w_k \alpha_k
\le
B_{\mathrm{FP}}
}
]

subject to:

[
0\le\alpha_k\le\alpha_{\max}.
]

FABRID chooses the vector:

[
\boldsymbol{\alpha}
===================

(\alpha_1,\ldots,\alpha_K)
]

to optimize a detection objective.

---

# 5. Primary Research Question

> **When federated IoT clients have heterogeneous score distributions and different marginal detection benefits from increased alerting, does reallocating a common nominal false-alert allowance across clients improve detection utility relative to uniform operating-rate policies, without modifying the underlying federated detector?**

---

# 6. Secondary Research Questions

### RQ1 — Average utility

Does FABRID improve federation-level attack-subtype Macro Recall at a fixed common false-alert allowance?

### RQ2 — Worst-client protection

Can a minimax FABRID policy improve the least-protected client without materially sacrificing federation-average detection?

### RQ3 — Heterogeneity mechanism

Does the value of FABRID increase as client-level detection–budget curves become more heterogeneous?

### RQ4 — Traffic weighting

How sensitive are conclusions to equal-client versus traffic-weighted definitions of the federation budget?

### RQ5 — Attack generalization

Does an allocation estimated from historical attack types transfer to attack types excluded from allocation design?

### RQ6 — Allocation stability

Are selected client operating rates stable to finite validation-data uncertainty?

### RQ7 — External validity

Does the effect reproduce on a second independently structured physical-device IoT population?

### RQ8 — Operational workload

Where timestamp and device provenance permit event-based analysis, do conclusions survive when alert points are converted into investigation-level alert events?

---

# 7. Hostile Literature Audit

## 7.1 Heterogeneous alert-rate regulation predates FABRID

Bridges, Jamieson, and Reed developed a mathematical threshold-setting procedure for ensembles of heterogeneous and potentially dynamic anomaly detectors. Their experiments regulated approximately **2,500 adaptive detectors processing more than 1.5 million events over five hours**.

Therefore FABRID shall never claim:

> “We are the first to allocate thresholds across heterogeneous anomaly detectors.”

The distinction is instead:

> FABRID optimizes **cross-client target operating rates inside a federated IDS under a shared federation-level constraint**, using locally estimated detection utility and independent final client calibration.

---

## 7.2 Constrained threshold optimization already exists

Kumar, Narasimhan, and Cotter study rate-constrained optimization of thresholded metrics, including optimizing false-negative rate under a fixed false-positive rate.

Therefore:

> The constrained optimizer itself is not FABRID's novelty.

---

## 7.3 Federated anomaly threshold calculation already exists

Laridi, Palmer, and Tam introduced a federated threshold calculation method for autoencoder anomaly detection using aggregated client summary statistics under non-IID conditions.

Therefore FABRID shall not claim:

> “First federated anomaly threshold method.”

FABRID allocates **different client target rates under one shared budget** rather than estimating a single federated anomaly threshold.

---

## 7.4 Distributed collaborative IoT threshold finding already exists

Ochiai et al. proposed WAFL-Autoencoder together with a distributed threshold-finding method for global anomaly detection across communicating IoT edges.

Therefore collaborative distributed thresholding is established prior art.

---

## 7.5 Federated finite-sample calibration already exists

Federated Conformal Prediction explicitly studies calibration and uncertainty quantification under heterogeneous federated clients and addresses violations of conventional exchangeability assumptions.

Any order-statistic or conformal-style calibration used in FABRID is therefore:

> **a statistical building block, not a novelty claim.**

---

## 7.6 Alert-budget-aware IDS already exists

CALIBURN explicitly incorporates operator-defined alert budgets into a streaming intrusion-detection pipeline and studies operational calibration under different attack-prevalence regimes.

Therefore FABRID shall never claim:

> “First IDS to introduce an alert budget.”

---

## 7.7 Event-level alarm-budget evaluation already exists

Heydari and Nyarko evaluate IDSs at matched alarm-event budgets rather than unconstrained threshold-free metrics. Their study uses explicit event-rate budgets, attack-event miss rates, detection delay, and workload diagnostics, and demonstrates that nominal calibration can fail to predict attacked-stream workload.

Therefore FABRID's contribution is not event-budgeted evaluation itself.

---

## 7.8 Cross-device IDS resource allocation already exists

Pădurean, Genge, and Bolboacă formulate a resource-aware IDS allocation problem as an integer linear program over heterogeneous devices, assigning different protocol-layer monitoring depths subject to a global resource constraint.

Therefore FABRID shall not claim:

> “First IDS resource allocation across devices.”

The allocated resource in FABRID is specifically:

> **false-alert operating capacity represented through client-specific target operating rates.**

---

# 8. Defensible Novelty Boundary

The manuscript's novelty claim should be limited to the combined formulation:

> **FABRID formulates post-training operating-point selection in federated IoT anomaly detection as a cross-client reallocation problem under one shared nominal false-alert allowance. Each client locally estimates the detection utility associated with candidate target false-positive rates, the server allocates those rates under a common constraint, and final thresholds are independently calibrated after allocation. All policies are evaluated using identical underlying anomaly scores and matched federation-level budgets.**

Do **not** use “first” in the manuscript title, abstract, or contributions unless a final search immediately before submission supports it.

---

# 9. Scientific Added Value

FABRID adds value through the combination of the following properties.

## 9.1 Federation-level resource formulation

The resource is defined globally rather than independently:

[
\sum_k w_k\alpha_k\le B_{\mathrm{FP}}.
]

---

## 9.2 Client-specific operating rates

FABRID permits:

[
\alpha_i\neq\alpha_j.
]

A client receives more alert capacity only when its estimated detection utility justifies it.

---

## 9.3 Fixed detector

The federated detector is frozen before any allocation policy is evaluated.

Thus:

[
\text{model weights}
]

are identical across:

* equal-FPR;
* greedy;
* FABRID;
* centralized diagnostic;
* oracle diagnostic.

---

## 9.4 Fixed scores

Every policy uses the exact same persisted anomaly scores.

Therefore any performance difference is attributable to:

[
\text{operating-point policy},
]

not representation learning.

---

## 9.5 Allocation/calibration independence

FABRID selects:

[
\alpha_k
]

using one data partition but estimates the final deployed:

[
\tau_k
]

using an independent benign calibration partition.

This prevents selection-induced calibration contamination.

---

## 9.6 Two explicit objectives

FABRID evaluates:

1. average attack detection;
2. worst-client protection.

No arbitrary weighted combination is used.

---

## 9.7 Matched-budget evaluation

Every meaningful policy comparison is performed under the same:

[
B_{\mathrm{FP}}.
]

---

# 10. Critical Budget Definition

A major methodological distinction must be maintained.

## 10.1 Nominal false-alert budget

The primary budget is:

[
B_{\mathrm{FP}}.
]

It refers to benign false alerts.

It does **not** mean:

> “The SOC will receive at most (B_{\mathrm{FP}}) total tickets.”

---

## 10.2 Total deployment alerts

During deployment:

[
A_{\mathrm{total}}
==================

A_{\mathrm{false}}
+
A_{\mathrm{true}}
+
A_{\mathrm{other}}.
]

Attack prevalence, drift, operational changes, and post-processing may alter total workload.

Recent event-budget work explicitly demonstrates that nominal calibration and attacked-stream alert emission can diverge substantially.

Therefore:

### Permitted claim

> “FABRID allocates a nominal benign false-alert allowance.”

### Forbidden claim

> “FABRID guarantees SOC workload.”

---

# 11. Weight-Provenance Audit

The previous specification implicitly equated dataset row counts with deployment traffic load.

That assumption is not universally defensible.

A dataset may contain unequal numbers of rows because of:

* different recording durations;
* different experimental procedures;
* file construction;
* filtering;
* packet-generation rates;
* feature extraction.

Therefore FABRID defines three weight-evidence levels.

---

# 12. Weight Evidence Levels

## 12.1 Level A — Operational weights

Use:

[
w_k
===

\frac{\lambda_k}{\sum_j\lambda_j}
]

where (\lambda_k) is a verified benign traffic or event rate measured over comparable wall-clock exposure.

Only **Level A** weights support operational workload statements.

---

## 12.2 Level B — Dataset-count proxy

Use:

[
w_k^{count}
===========

\frac{N_k}{\sum_jN_j}.
]

These weights describe the composition of the dataset.

They do **not** automatically represent deployment workload.

Results using them must be labelled:

> **dataset-volume-weighted sensitivity analysis**

rather than operational traffic allocation.

---

## 12.3 Level C — Equal-client weights

Use:

[
\boxed{
w_k=\frac1K.
}
]

This answers:

> “How should one average per-client false-alert allowance be redistributed across clients?”

This is the primary N-BaIoT confirmatory formulation because the published data establish sequential per-device observations but do not establish comparable wall-clock collection durations from which defensible operational rates can be inferred. The original study reports per-device benign counts and chronological data construction, but not a common equal-duration collection contract across devices.

---

# 13. Primary N-BaIoT Budget Contract

With:

[
K=9
]

and equal-client weighting:

[
\boxed{
\frac1{9}
\sum_{k=1}^{9}
\alpha_k
\le
B_{\mathrm{FP}}.
}
]

Interpret this as:

> **average allocated client benign-FPR target**

not:

> percentage of all real-world SOC traffic.

This wording is mandatory.

---

# 14. Consequence for the Equal-Alert Baseline

Under equal weights:

[
w_k=\frac1K,
]

equal absolute budget contribution implies:

[
\alpha_1=\cdots=\alpha_K.
]

Therefore:

[
EQ_ALERT
\equiv
EQ_FPR.
]

They must **not** be shown as two separate primary N-BaIoT baselines.

`EQ_ALERT` is used only in experiments with defensibly unequal (w_k).

This removes a redundant baseline from the earlier protocol.

---

# 15. Threat Model

FABRID version 1 assumes:

* fixed federation membership during an allocation epoch;
* one coordinating server;
* honest protocol execution;
* local client calibration and validation data;
* no malicious utility reports;
* no compromised allocation messages;
* no malicious server;
* no Byzantine clients;
* no adversarial poisoning of final calibration data.

---

# 16. Explicitly Out of Scope

The following are not claims of FABRID v1:

* Byzantine robustness;
* malicious-client budget capture;
* secure aggregation;
* differential privacy;
* server compromise;
* membership churn;
* continual online reallocation;
* concept-drift guarantees;
* calibration-poisoning defense;
* inference-time evasion;
* communication confidentiality;
* raw-packet adversarial robustness.

These are future extensions, not required additions to the present study.

---

# 17. Privacy Boundary

FABRID is:

> **raw-data decentralized**

but not:

> **formally privacy-preserving**.

Clients transmit operating-utility information rather than raw local observations.

However, utility curves can leak information about:

* local attack detectability;
* score geometry;
* local traffic characteristics;
* attack composition.

Therefore never claim differential privacy or privacy preservation without an actual mechanism and proof.

---

# 18. Detector Contract

The detector is not the contribution. FABRID-IDS is a standalone, detector-agnostic decision layer:

[
\boxed{
\text{frozen detector}
\rightarrow
\text{frozen anomaly scores}
\rightarrow
\text{local detection–budget utility estimation}
\rightarrow
\text{cross-client target-rate allocation}
\rightarrow
\text{independent final benign calibration}
\rightarrow
\text{matched-budget evaluation}.
}
]

FABRID-IDS must not be presented, implemented, or described as an extension, variant, or derivative of
any other named federated-learning research codebase or project. It has no scientific or runtime
dependency on any such codebase.

Use one fixed detector family for the confirmatory study, with its own frozen:

* preprocessing;
* federated aggregation rule (e.g. FedAvg) or centralized training rule, as applicable;
* local training rule;
* architecture;
* optimization hyperparameters;
* training rounds;
* anomaly-score function.

These are implemented directly within the FABRID-IDS codebase (or vendored/reimplemented as needed)
rather than imported from an external research stack. Internal code reuse of generic, non-FABRID-specific
utilities is permitted where it does not create a scientific or runtime dependency; FABRID-IDS's own
partitioning, calibration, allocation, optimization, and statistics contracts always take precedence and
are never inherited from elsewhere.

Do not retune the detector for FABRID after allocation results are observed.

---

# 19. Model Isolation Contract

For each:

[
dataset\times seed
]

there shall be exactly one terminal detector state.

All threshold/allocation policies reference that same state.

Required invariant:

[
SHA256(model_{\mathrm{policy\ A}})
==================================

SHA256(model_{\mathrm{policy\ B}}).
]

No policy is allowed to retrain the detector.

---

# 20. Score Contract

The detector produces:

[
s(x)\in\mathbb R
]

where larger values mean greater anomaly evidence.

The deployed decision rule is:

[
\boxed{
\hat y(x)
=========

\mathbf{1}[s(x)>\tau].
}
]

Use strict:

```text
>
```

everywhere.

Do not alternate between:

```text
>
```

and:

```text
>=
```

Ties exactly at the threshold are non-alerts.

---

# 21. N-BaIoT Primary Dataset

N-BaIoT contains real traffic from **nine commercial IoT devices** infected using Mirai and BASHLITE and contains approximately **7.06 million sequential observations with 115 extracted traffic features**.

The original feature construction summarizes network behavior over five overlapping temporal windows:

[
100\text{ ms},;
500\text{ ms},;
1.5\text{ s},;
10\text{ s},;
1\text{ min}.
]

This overlap is important because observations should not be treated as independent IID samples for inferential statistics.

---

# 22. N-BaIoT Client Population

Use exactly the nine natural devices reported in the original study. Their benign counts and botnet availability are:

| ID | Client                                   | Benign rows | Mirai | BASHLITE |
| -: | ---------------------------------------- | ----------: | :---: | :------: |
|  1 | Danmini Doorbell                         |      49,548 |  Yes  |    Yes   |
|  2 | Ennio Doorbell                           |      39,100 |   No  |    Yes   |
|  3 | Ecobee Thermostat                        |      13,113 |  Yes  |    Yes   |
|  4 | Philips B120N/10 Baby Monitor            |     175,240 |  Yes  |    Yes   |
|  5 | Provision PT-737E Security Camera        |      62,154 |  Yes  |    Yes   |
|  6 | Provision PT-838 Security Camera         |      98,514 |  Yes  |    Yes   |
|  7 | SimpleHome XCS7-1002-WHT Security Camera |      46,585 |  Yes  |    Yes   |
|  8 | SimpleHome XCS7-1003-WHT Security Camera |      19,528 |  Yes  |    Yes   |
|  9 | Samsung SNH 1011 N Webcam                |      52,150 |   No  |    Yes   |

Do not generate artificial Dirichlet clients for the primary experiment.

---

# 23. N-BaIoT Attack Taxonomy

The original study contains five BASHLITE attack types:

1. Scan;
2. Junk;
3. UDP;
4. TCP;
5. COMBO.

And five Mirai attack types:

1. Scan;
2. ACK;
3. SYN;
4. UDP;
5. UDPplain.

Use canonical internal identifiers:

```text
bashlite_scan
bashlite_junk
bashlite_udp
bashlite_tcp
bashlite_combo
mirai_scan
mirai_ack
mirai_syn
mirai_udp
mirai_udpplain
```

Do not merge the two `scan` or `udp` categories merely because their names overlap.

---

# 24. Benign Partitioning

Use source-order contiguous splits.

For a client with (n) benign rows:

[
i_1=\lfloor0.50n\rfloor
]

[
i_2=\lfloor0.70n\rfloor
]

[
i_3=\lfloor0.80n\rfloor.
]

Then:

```text
BENIGN_TRAIN      = [0, i1)
BENIGN_FRONTIER   = [i1, i2)
BENIGN_FINAL_CAL  = [i2, i3)
BENIGN_TEST       = [i3, n)
```

Fractions are therefore approximately:

* 50% training;
* 20% allocation/frontier construction;
* 10% independent final calibration;
* 20% final benign test.

No row may belong to two partitions.

---

# 25. Exact N-BaIoT Benign Split Counts

Using the deterministic floor rule:

| Client          |  Train | Frontier | Final Cal |   Test |
| --------------- | -----: | -------: | --------: | -----: |
| Danmini         | 24,774 |    9,909 |     4,955 |  9,910 |
| Ennio           | 19,550 |    7,820 |     3,910 |  7,820 |
| Ecobee          |  6,556 |    2,623 |     1,311 |  2,623 |
| Philips         | 87,620 |   35,047 |    17,525 | 35,048 |
| PT-737E         | 31,077 |   12,430 |     6,216 | 12,431 |
| PT-838          | 49,257 |   19,702 |     9,852 | 19,703 |
| SimpleHome-1002 | 23,292 |    9,317 |     4,659 |  9,317 |
| SimpleHome-1003 |  9,764 |    3,905 |     1,953 |  3,906 |
| Samsung         | 26,075 |   10,430 |     5,215 | 10,430 |

The original N-BaIoT evaluation also used chronological benign partitioning, so preserving source order is more faithful than shuffling these sequential observations.

---

# 26. Attack Partitioning

For every:

[
client\times attack\ subtype
]

with (n_a) rows:

[
j_a=\lfloor0.20n_a\rfloor.
]

Assign:

```text
ATTACK_VALIDATION = [0, j_a)
ATTACK_TEST       = [j_a, n_a)
```

Thus:

* first 20% → allocation utility estimation;
* remaining 80% → final attack test.

No test attack score or label enters FABRID allocation.

---

# 27. Why Attack Validation Is Permitted

The underlying anomaly detector remains benign-trained.

However, FABRID uses historical attack labels to estimate:

[
\text{benefit of allocating more alert capacity}.
]

Therefore the correct description is:

> **benign-trained anomaly detector with validation-informed operating-point allocation.**

Do not call the complete FABRID decision layer “fully unsupervised.”

---

# 28. Allocation/Calibration Separation

The data roles are strictly separated:

### `BENIGN_FRONTIER`

Used to map candidate target rates to provisional thresholds during allocation design.

### `ATTACK_VALIDATION`

Used to estimate detection utility.

### Server allocation

Chooses:

[
\alpha_k^*.
]

### `BENIGN_FINAL_CAL`

Used only **after** (\alpha_k^*) has been selected to produce the final deployed threshold.

### Final tests

Used only for evaluation.

Therefore:

[
\boxed{
D_{\mathrm{select}}
\cap
D_{\mathrm{final-cal}}
======================

\emptyset.
}
]

This is a core internal-validity requirement.

---

# 29. Candidate Target-Rate Grid

The previous 0.001 minimum was too coarse.

At the smallest federation budget:

[
B_{\mathrm{FP}}=0.001,
]

FABRID may need to assign some clients less than 0.1% while reallocating capacity to others.

Use:

[
\boxed{
\mathcal A
==========

{0}
\cup
\operatorname{LogSpace}_{201}(10^{-4},0.05)
\cup
{0.001,;0.0025,;0.005,;0.01,;0.02}.
}
]

The logarithmic component is:

[
\alpha_j
========

10^{
-4+
j
\frac{
\log_{10}(0.05)+4
}{200}
},
\qquad
j=0,\ldots,200.
]

After sorting and duplicate removal at:

[
10^{-12}
]

absolute tolerance, the target grid contains:

[
\boxed{207}
]

unique values.

Persist the exact array to:

```text
config/alpha_grid.json
```

and never regenerate it independently.

---

# 30. Local Target-Rate Cap

Set:

[
\boxed{
\alpha_{\max}=0.05.
}
]

Thus no client may receive a nominal target FPR above:

[
5%.
]

This is a preregistered engineering guardrail.

It is **not** claimed as a universal SOC standard.

---

# 31. Primary Record-Level Budgets

Use exactly:

[
\boxed{
B_{\mathrm{FP}}
\in
{
0.001,;
0.0025,;
0.005,;
0.010,;
0.020
}.
}
]

Equivalent average rates:

| Budget | Percent |
| -----: | ------: |
|  0.001 |   0.10% |
| 0.0025 |   0.25% |
|  0.005 |   0.50% |
|  0.010 |   1.00% |
|  0.020 |   2.00% |

No new confirmatory budget value may be introduced after results are inspected.

---

# 32. Finite-Sample Threshold Rule

For a benign calibration score set:

[
S_k
===

{s_1,\ldots,s_n}
]

sort:

[
s_{(1)}
\le
\cdots
\le
s_{(n)}.
]

For target rate:

[
\alpha>0,
]

calculate:

[
r
=

\left\lceil
(n+1)(1-\alpha)
\right\rceil.
]

If:

[
r>n,
]

set:

[
\tau=+\infty.
]

Otherwise:

[
\tau=s_{(r)}.
]

For:

[
\alpha=0,
]

always set:

[
\tau=+\infty.
]

Decision:

[
alert
\iff
s>\tau.
]

---

# 33. Calibration Resolution

The smallest nonzero rate that can produce a finite threshold is approximately:

[
\frac1{n+1}.
]

For the N-BaIoT final calibration partitions:

| Client          | Final-cal (n) | Approx. minimum resolvable rate |
| --------------- | ------------: | ------------------------------: |
| Danmini         |         4,955 |                        0.02018% |
| Ennio           |         3,910 |                        0.02557% |
| Ecobee          |         1,311 |                        0.07622% |
| Philips         |        17,525 |                        0.00571% |
| PT-737E         |         6,216 |                        0.01608% |
| PT-838          |         9,852 |                        0.01015% |
| SimpleHome-1002 |         4,659 |                        0.02146% |
| SimpleHome-1003 |         1,953 |                        0.05118% |
| Samsung         |         5,215 |                        0.01917% |

A target below a client's finite-sample resolution is legal, but it yields:

[
\tau=+\infty
]

and therefore zero alerts.

Do not silently substitute another target.

---

# 34. Statistical Guarantee Boundary

Order-statistic calibration has meaningful finite-sample interpretations under exchangeability assumptions.

However, N-BaIoT is explicitly a **sequential** dataset, and its 115 features are constructed from overlapping time windows.

Therefore do **not** state:

> “FABRID guarantees an exact future client FPR on N-BaIoT.”

Use:

> “FABRID uses a finite-sample order-statistic calibration rule; theoretical marginal calibration interpretations require the stated exchangeability assumptions, while empirical held-out FPR is reported without assuming row independence.”

This is substantially safer.

---

# 35. Local Utility Curve

For client (k), attack subtype (a), and candidate target rate (j):

[
TPR_{k,a,j}
===========

\frac{
TP_{k,a,j}
}{
TP_{k,a,j}+FN_{k,a,j}
}.
]

Define client utility:

[
\boxed{
u_{k,j}
=======

\frac1{|\mathcal T_k|}
\sum_{a\in\mathcal T_k}
TPR_{k,a,j}.
}
]

Thus attacks are first averaged within client by subtype.

A large UDP file cannot dominate simply because it contains many more rows than a smaller subtype.

---

# 36. Utility Eligibility

For the standard validation-informed experiment, require a client to have:

[
n_{\mathrm{attack,val}}\ge200
]

and at least:

[
2
]

eligible attack subtypes, each with:

[
n_{a,\mathrm{val}}\ge50.
]

These are preregistered data-sufficiency guardrails, not universal theoretical thresholds.

---

# 37. Utility Fallback

If a client fails utility eligibility:

[
\boxed{
\alpha_k=B_{\mathrm{FP}}
}
]

subject to:

[
\alpha_k\le0.05.
]

This is the fallback equal-FPR allocation.

Its budget cost is reserved before the optimization problem for eligible clients is solved.

Report:

[
FallbackRate
============

\frac{#\text{fallback clients}}{K}.
]

If more than:

[
20%
]

of clients in the external replication require fallback, the validation-informed external analysis is classified as **supportive**, not confirmatory.

---

# 38. FABRID-Macro Formulation

For eligible client (k) and candidate (j), define:

[
x_{k,j}\in{0,1}.
]

Exactly one target rate is chosen:

[
\sum_jx_{k,j}=1.
]

Candidate cost:

[
c_{k,j}
=======

w_k\alpha_j.
]

Let (B_R) be the remaining budget after fallback reservation.

Solve:

[
\boxed{
\max_x
\frac1{K_e}
\sum_{k\in E}
\sum_j
u_{k,j}x_{k,j}
}
]

subject to:

[
\sum_{k\in E}\sum_j
c_{k,j}x_{k,j}
\le B_R.
]

This policy is named:

```text
FABRID_MACRO
```

---

# 39. FABRID-Minimax Formulation

Introduce:

[
z.
]

Stage 1:

[
\boxed{
\max z
}
]

subject to:

[
\sum_j
u_{k,j}x_{k,j}
\ge z,
\qquad
\forall k\in E,
]

plus all FABRID budget and one-choice constraints.

Let optimum be:

[
z^*.
]

Stage 2:

constrain:

[
z\ge z^*-10^{-9}
]

and maximize:

[
\frac1{K_e}
\sum_{k,j}
u_{k,j}x_{k,j}.
]

This policy is:

```text
FABRID_MINIMAX
```

---

# 40. Why No Weighted Macro/Minimax Objective Is Used

Do not invent:

[
\lambda MacroRecall
+
(1-\lambda)WorstRecall.
]

Such a formulation introduces an arbitrary policy preference.

Instead:

* `FABRID_MACRO` answers the efficiency question;
* `FABRID_MINIMAX` answers the worst-client protection question.

The two objectives remain scientifically interpretable.

---

# 41. Optimization Solver

Use:

```text
scipy.optimize.milp
```

which wraps HiGHS and supports integer variables, time limits, and MIP relative-gap control. SciPy documents solver status `0` as an optimal solution and notes that the implementation is deterministic.

Set:

```text
integrality = 1 for all x variables
bounds      = [0, 1]
mip_rel_gap = 0
time_limit  = 60 seconds
```

Accept a FABRID optimization result only if:

```text
res.success == True
res.status == 0
res.mip_gap <= 1e-9
```

Otherwise mark:

```text
SOLVER_INVALID
```

and exclude the coordinate pending investigation.

Never silently use a time-limited feasible solution as an optimum.

---

# 42. Deterministic FABRID Tie-Breaking

## `FABRID_MACRO`

Sequential solves:

1. maximize Macro utility;
2. constrain utility to within (10^{-9}) of the optimum;
3. minimize total budget consumption;
4. constrain budget consumption to within (10^{-12}) of that optimum;
5. lexicographically minimize the selected (\alpha) vector ordered by immutable client ID.

---

## `FABRID_MINIMAX`

Sequential solves:

1. maximize minimum utility;
2. fix minimum utility;
3. maximize Macro utility;
4. fix Macro utility;
5. minimize total budget;
6. lexicographically minimize (\alpha).

This makes the scientific output independent of arbitrary equivalent MILP solutions.

---

# 43. Baseline 1 — `EQ_FPR`

Every client receives:

[
\boxed{
\alpha_k=B_{\mathrm{FP}}.
}
]

For equal-client N-BaIoT:

[
\frac1K\sum_k\alpha_k
=====================

B_{\mathrm{FP}}.
]

This is the principal baseline.

---

# 44. Baseline 2 — `GREEDY`

Initialize:

[
\alpha_k=0.
]

For every client, consider the next larger candidate rate.

Calculate marginal utility efficiency:

[
\rho_{k}
========

\frac{
\Delta u_k
}{
w_k\Delta\alpha_k
}.
]

Select the feasible increment with largest (\rho_k).

Tie order:

1. larger (\Delta u);
2. lower incremental budget;
3. lower immutable client ID;
4. lower resulting (\alpha).

Repeat until no increment fits.

This asks whether exact FABRID optimization materially improves over an intuitive marginal-gain heuristic.

---

# 45. Conditional Baseline — `EQ_ALERT`

Use only where:

[
w_i\neq w_j
]

is justified by valid weight provenance.

Find the maximal constant budget share (c) satisfying:

[
\sum_k
\min(c,0.05w_k)
\le B_{\mathrm{FP}}.
]

Then:

[
\alpha_k
========

\min
\left(
\frac{c}{w_k},
0.05
\right).
]

Under equal-client weighting this baseline is identical to `EQ_FPR` and shall not be duplicated.

---

# 46. Centralized Diagnostic — `POOLED_SHARED`

This is explicitly not federated.

Pool validation-stage scores and select one common absolute score threshold that optimizes validation Macro Recall subject to the corresponding nominal false-alert constraint.

Purpose:

> Determine whether client-specific rate allocation adds value beyond one centrally selected global score cutoff.

Call it:

```text
POOLED_SHARED
```

Do not call it deployable in the federated setting.

---

# 47. Test Oracle

Define:

```text
TEST_ORACLE
```

using the same discrete allocation problem but computing (u_{k,j}) using test attacks.

It is strictly a non-deployable upper bound.

It shall never:

* enter primary hypothesis tests;
* determine hyperparameters;
* determine budget values;
* determine success/failure;
* be presented as a competing practical policy.

---

# 48. Final Threshold Calibration

Once FABRID or a baseline has frozen:

[
\alpha_k^*,
]

discard the provisional `BENIGN_FRONTIER` threshold.

Load:

```text
BENIGN_FINAL_CAL
```

and compute:

[
\tau_k^*
]

using the finite-sample rule defined earlier.

No attack validation data are used at this stage.

Persist:

```text
alpha_selected
threshold
calibration_n
calibration_sha256
```

---

# 49. Primary Detection Metric

For each client (k), calculate attack-subtype Macro Recall:

[
R_k
===

\frac1{|\mathcal T_k^{test}|}
\sum_{a\in\mathcal T_k^{test}}
TPR_{k,a}.
]

Federation Macro Recall:

[
\boxed{
MacroRecall
===========

\frac1K\sum_kR_k.
}
]

This is the primary effectiveness endpoint for `FABRID_MACRO`.

---

# 50. Worst-Client Metric

[
\boxed{
WorstClientRecall
=================

\min_kR_k.
}
]

This is the primary endpoint for `FABRID_MINIMAX`.

---

# 51. Federation FPR

Generic weighted definition:

[
\boxed{
FPR_{\mathrm{fed}}
==================

\sum_kw_kFPR_k.
}
]

For primary N-BaIoT equal-client analysis:

[
\boxed{
FPR_{\mathrm{fed}}
==================

\frac1{9}\sum_{k=1}^{9}FPR_k.
}
]

Report this as:

> average client FPR

not traffic-weighted SOC false-alert probability.

---

# 52. Budget Usage Ratio

[
\boxed{
BUR
===

\frac{
FPR_{\mathrm{fed}}
}{
B_{\mathrm{FP}}
}.
}
]

Interpretation:

* (BUR=0.8): 80% of nominal budget;
* (BUR=1.0): exactly the nominal budget;
* (BUR=1.2): empirical test FPR is 20% above it.

Do not clamp the ratio.

---

# 53. Budget Violation Ratio

[
\boxed{
BVR
===

\max(0,BUR-1).
}
]

Also report:

[
MaxClientFPR
============

\max_kFPR_k.
]

Aggregate compliance must not hide an extremely noisy client.

---

# 54. Client FPR Dispersion

Report:

[
MedianFPR
]

[
IQR(FPR_k)
]

[
MinFPR
]

[
MaxFPR.
]

Also report:

[
CV_{FPR}
========

\frac{
\sigma(FPR_k)
}{
\mu(FPR_k)
}.
]

If:

[
\mu(FPR_k)=0,
]

write:

```text
NA
```

rather than zero.

---

# 55. False-Alert Concentration

Where comparable alert counts are meaningful, define:

[
A_k^{FP}.
]

Then calculate Gini:

[
\boxed{
G
=

\frac{
\sum_i\sum_j
|A_i^{FP}-A_j^{FP}|
}{
2K\sum_iA_i^{FP}
}.
}
]

If:

[
\sum_iA_i^{FP}=0,
]

define:

[
G=0.
]

For equal-client N-BaIoT record-level analysis, use Gini only as a **distributional diagnostic**, not a claim about analyst labor.

---

# 56. Secondary Metrics

Report:

* pooled recall;
* Macro-F1;
* balanced accuracy;
* AUROC;
* AUPRC.

AUROC and AUPRC describe the frozen detector.

They cannot establish FABRID superiority because FABRID does not change score rankings.

---

# 57. Mandatory AUROC Invariance

Within:

[
dataset\times seed,
]

all threshold policies must satisfy:

[
\boxed{
|\Delta AUROC|<10^{-12}.
}
]

If this invariant fails, the fixed-score experimental contract has been violated.

---

# 58. N-BaIoT Attack-Subtype-Disjoint Generalization

Do **not** assign attack types to folds using a hash.

That procedure can leave BASHLITE-only devices with an empty or badly unbalanced validation fold.

Use this fixed global mapping.

### Fold 0

```text
bashlite_scan
bashlite_tcp
mirai_ack
mirai_udp
```

### Fold 1

```text
bashlite_junk
bashlite_udp
mirai_scan
```

### Fold 2

```text
bashlite_combo
mirai_syn
mirai_udpplain
```

Run:

| Rotation | Validation attacks | Test attacks |
| -------- | ------------------ | ------------ |
| 0        | Fold 0             | Folds 1 + 2  |
| 1        | Fold 1             | Folds 0 + 2  |
| 2        | Fold 2             | Folds 0 + 1  |

This mapping ensures that BASHLITE-only clients retain a validation attack subtype in every rotation.

Call the experiment:

> **attack-subtype-disjoint generalization**

not zero-day proof.

---

# 59. Optional Stronger Botnet-Family Generalization

Seven N-BaIoT clients contain both Mirai and BASHLITE.

For those seven clients only, add:

### Direction A

```text
utility validation = BASHLITE
final attack test  = Mirai
```

### Direction B

```text
utility validation = Mirai
final attack test  = BASHLITE
```

Report explicitly:

[
K=7.
]

Call this:

> **botnet-family-disjoint transfer**

Do not silently exclude Ennio and Samsung.

---

# 60. Client-Heterogeneity Mechanism Test

Traffic volume is not the only heterogeneity relevant to FABRID.

The more fundamental quantity is heterogeneity of:

[
u_k(\alpha).
]

Define a curve-dispersion diagnostic at every candidate (\alpha_j):

[
H_u(\alpha_j)
=============

SD_k(u_{k,j}).
]

Also report aggregate curve heterogeneity:

[
\boxed{
H_U
===

\frac1J
\sum_j
SD_k(u_{k,j}).
}
]

This directly measures whether clients benefit differently from additional false-alert capacity.

This should be more central than merely transforming traffic weights.

---

# 61. Weight-Heterogeneity Sensitivity

For sensitivity analyses with a nonuniform reference weight vector (w_k), define:

[
w_k^{(\gamma)}
==============

\frac{
w_k^\gamma
}{
\sum_\ell w_\ell^\gamma
}.
]

Use:

[
\boxed{
\gamma\in
{0,;0.5,;1,;1.5}.
}
]

Interpretation:

* (0): equal-client budget;
* (0.5): reduced concentration;
* (1): original proxy/operational weights;
* (1.5): amplified concentration.

If the underlying weights are only dataset-count proxies, label the complete experiment:

> **dataset-composition sensitivity**

not traffic-load sensitivity.

---

# 62. Allocation Stability Audit

The previous audit resampled only attack validation.

That is incomplete because FABRID's utility curve depends on both:

* benign frontier calibration;
* attack validation.

For each:

[
seed\times budget
]

perform:

[
\boxed{500}
]

allocation-sensitivity replicates.

Each replicate resamples:

1. `BENIGN_FRONTIER`;
2. `ATTACK_VALIDATION` independently within attack subtype.

Then recompute:

* provisional thresholds;
* utility curves;
* FABRID allocation.

For each client report:

* modal selected (\alpha);
* modal frequency;
* median (\alpha);
* 5th percentile;
* 95th percentile.

Define:

[
\boxed{
Instability_k
=============

1-
\max_j
P(\alpha_k=\alpha_j).
}
]

Because N-BaIoT observations are temporally dependent, this procedure is described as:

> **allocation sensitivity analysis**

rather than a formal IID bootstrap confidence interval.

---

# 63. Conservative Utility Sensitivity

For every subtype recall estimate, compute a one-sided:

[
95%
]

binomial lower confidence bound.

Construct a second utility curve:

[
u^{LCB}_{k,j}.
]

Resolve FABRID using that curve.

Report whether policy conclusions survive.

This directly tests whether FABRID is exploiting uncertain optimistic validation estimates.

---

# 64. Seeds

Use exactly:

[
\boxed{
{0,1,2,3,4,5,6,7,8,9}.
}
]

Ten detector seeds.

No seeds may be removed because they perform poorly.

Allocation and calibration rules are deterministic once scores and partitions are fixed.

---

# 65. Primary Experimental Policies

For N-BaIoT equal-client weighting:

1. `EQ_FPR`
2. `GREEDY`
3. `FABRID_MACRO`
4. `FABRID_MINIMAX`
5. `POOLED_SHARED` — centralized diagnostic
6. `TEST_ORACLE` — upper bound only

`EQ_ALERT` is omitted because it is identical to `EQ_FPR` under equal weights.

---

# 66. Main Experiment Count

Deployable/fair policy comparison:

[
10\ seeds
\times
5\ budgets
\times
4\ deployable\ policies
=======================

\boxed{200}
]

primary policy evaluations.

Centralized diagnostic:

[
10\times5
=========

50.

]

Oracle diagnostic:

[
10\times5
=========

50.

]

Total recorded main cells:

[
\boxed{300}.
]

Only ten detector trainings are required because score artifacts are reused.

---

# 67. Attack-Subtype-Disjoint Evaluation Count

For four deployable policies:

[
10
\times
3
\times
5
\times
4
=

\boxed{600}
]

primary cells.

Add:

[
150
]

`POOLED_SHARED` diagnostic cells and:

[
150
]

oracle cells if both are recomputed for all rotations.

Grand total:

[
\boxed{900}
]

recorded cells.

---

# 68. Statistical Experimental Unit

The experimental unit is:

> **detector seed**

not an individual network row.

Thus each method contrast has:

[
n=10
]

paired independent training replications.

Millions of highly related network observations shall not be used as pseudo-replicates.

---

# 69. Primary Statistical Contrasts

Pre-register exactly:

### Contrast A

[
FABRID_MACRO
------------

EQ_FPR
]

on:

[
MacroRecall.
]

### Contrast B

[
FABRID_MINIMAX
--------------

EQ_FPR
]

on:

[
WorstClientRecall.
]

Because `EQ_ALERT` is redundant under the primary equal-client formulation, it is not used as Contrast B.

---

# 70. Exact Sign-Flip Test

For ten paired seed differences:

[
d_1,\ldots,d_{10},
]

enumerate all:

[
2^{10}
======

\boxed{1,024}
]

sign assignments.

Calculate the two-sided exact sign-flip test on the paired mean difference.

Use:

[
\alpha=0.05.
]

Apply Holm correction over the five budget-specific tests separately for each primary contrast.

The sign-flip test is reported as seed-level inference and not as proof that arbitrary future datasets share the same distribution.

---

# 71. Confidence Intervals

For effect-size reporting:

[
\boxed{50,000}
]

paired seed-bootstrap resamples.

Report:

* mean paired difference;
* median paired difference;
* 95% bootstrap CI;
* exact sign-flip (p);
* Holm-adjusted (p).

Do not report a (p)-value without the corresponding effect size.

---

# 72. Practical Success Gates

These are **preregistered engineering thresholds**, not universal scientific constants.

## `FABRID_MACRO`

Require:

[
\Delta MacroRecall
\ge2.0
]

percentage points versus `EQ_FPR`

at at least:

[
\boxed{3/5}
]

primary budgets.

---

## `FABRID_MINIMAX`

Require:

[
\Delta WorstClientRecall
\ge5.0
]

percentage points,

while:

[
\Delta MacroRecall
\ge-2.0
]

percentage points,

at at least:

[
3/5
]

budgets.

---

# 73. Budget-Compliance Reporting

Do not use an unstable 95th percentile estimate from only ten seed values as a hard success criterion.

Instead require:

[
median(BUR)\le1.05
]

and:

[
\boxed{
#{seed:BUR\le1.10}\ge9/10.
}
]

Also report:

[
max(BUR).
]

If these conditions fail, FABRID may still improve recall, but the manuscript may not characterize empirical budget transfer as reliable on that dataset.

---

# 74. External Replication — CIC IoT-DIAD 2024

CIC IoT-DIAD 2024 contains a topology of **105 IoT devices**, **33 attacks**, and **seven attack categories**: DDoS, DoS, Recon, Web-based, Brute Force, Spoofing, and Mirai. It provides both packet-based and flow-based processed representations.

Use it as the preferred physical-device external replication candidate.

---

# 75. External Client Definition

For the packet-based representation:

[
client
======

device_mac.
]

`device_mac` is used only to form natural client groups.

It must never enter the model feature vector.

---

# 76. External Device Eligibility

A device qualifies if:

[
n_{\mathrm{benign}}\ge10,000
]

and:

[
n_{\mathrm{attack}}\ge1,000.
]

Validation-informed FABRID additionally requires:

[
n_{\mathrm{attack,val}}\ge200
]

and at least two eligible attack categories/subtypes with:

[
n_{a,val}\ge50.
]

No manual client selection.

If fewer than:

[
10
]

clients qualify, the dataset is classified as supportive rather than confirmatory.

---

# 77. CIC IoT-DIAD Leakage Exclusion

At minimum, exclude from model input:

```text
device_mac
anomaly label
device-identification label
source row id
source file
split id
```

Additionally exclude direct endpoint or textual identifiers unless a frozen preprocessing audit explicitly establishes a legitimate detection rationale, including fields such as:

* direct MAC identifiers;
* explicit IP identifiers;
* vendor/OUI textual identity;
* host strings;
* URI strings;
* user-agent strings;
* DNS host/server identifiers;
* TLS server identifiers.

CIC IoT-DIAD was explicitly designed for both device identification and anomaly detection, so identity-oriented features are particularly dangerous if client identity is also the federated partition variable.

Freeze and hash the exact feature manifest before training.

---

# 78. External Feature Parsing Rule

For a field proposed as numeric:

* successful numeric parsing must be at least **99.9%**;
* otherwise exclude the field from the confirmatory feature manifest.

Do not silently coerce malformed text to zero.

Store:

```text
feature_manifest.json
feature_manifest.sha256
```

---

# 79. Temporal Provenance Loophole

Do not assume the processed CIC IoT-DIAD packet table and flow table can be joined merely because one provides useful device identity and the other useful timing fields.

The official dataset documentation describes packet-based and flow-based representations as separate feature extraction outputs.

Therefore:

* no synthetic timestamp from row number;
* no undocumented cross-table join;
* no cumulative inter-arrival reconstruction;
* no event-level FABRID result unless raw provenance validates device + time jointly.

---

# 80. Operational Event-Level Validation Dataset

Record-level FPR does not equal analyst investigations.

Therefore FABRID should include an event-level validation branch when an appropriate dataset passes a provenance gate.

Two strong candidates are:

### Gotham Dataset 2025

Gotham contains **78 emulated heterogeneous IoT devices**, raw PCAP and labelled CSV data, with traffic captured separately at each IoT device interface. This makes it particularly suitable for distributed device-level temporal reconstruction.

### CICIoMT2024

CICIoMT2024 contains **40 IoMT devices—25 real and 15 simulated—with 18 attacks**, and distributes original PCAP traffic for Wi-Fi/MQTT and Bluetooth experiments.

---

# 81. `EVENT_DATA_GATE`

Event-level claims are allowed only if all of the following are proven from source provenance:

* immutable physical/emulated client ID;
* packet timestamp;
* attack/benign interval provenance;
* deterministic score association;
* ordering within client;
* observation duration;
* non-overlapping final evaluation period.

If one requirement fails:

```text
EVENT_DATA_GATE = FAIL
```

and no event-level workload claim is produced.

---

# 82. Eventization Protocol

If the gate passes, adopt the established primary event-construction values:

[
\boxed{
(d,m,\ell_{\min},c)
===================

(2,5,2,10)\text{ seconds}.
}
]

Use:

* dilation (d=2) s;
* merge gap (m=5) s;
* minimum event length (\ell_{\min}=2) s;
* cooldown (c=10) s.

Apply alarm duty guardrail:

[
\boxed{
Duty\le0.25.
}
]

Evaluate:

[
\boxed{
B_E
\in
{0.1,0.2,0.5}
}
]

alert events/client/hour on average.

These values are borrowed from the 2026 event-budget evaluation literature and are not FABRID contributions.

---

# 83. Eventization Robustness

At representative:

[
B_E=0.2
]

events/client/hour, perform a post-processing sensitivity grid:

[
d\in{1,2,3}
]

[
m\in{3,5,10}
]

[
\ell_{\min}\in{1,2,3}
]

[
c\in{5,10,20}.
]

Total:

[
3^4
===

\boxed{81}
]

post-processing combinations.

This determines whether FABRID's event-level result depends on one arbitrary alarm-merging configuration.

---

# 84. Event-Level Terminology

On nominal attack-free data, report:

> false alert events/hour.

On an attacked stream, alerts occurring outside annotated attack intervals should not automatically be called confirmed false positives unless the dataset's ground truth justifies that interpretation.

Prefer:

> non-attack-interval alert events/hour

or:

> unlabeled alert workload.

This avoids overclaiming ground-truth cleanliness during attacked operation.

---

# 85. Event-Level Metrics

Where `EVENT_DATA_GATE=PASS`, report:

* false alert events/hour on nominal data;
* attacked-stream alert events/hour;
* attack-event recall;
* attack-event miss rate;
* mean time to detect;
* median time to detect;
* 90th-percentile time to detect;
* alarm duty fraction;
* per-client alert-event Gini.

---

# 86. Communication Protocol

The global candidate grid has:

[
207
]

known target rates.

The client need not send local score vectors or provisional thresholds to the server.

A minimal logical message contains:

| Field                       | Type    |   Bytes |
| --------------------------- | ------- | ------: |
| 207 utility values          | float32 |     828 |
| client UUID                 | 128-bit |      16 |
| nominal/predeployment count | uint64  |       8 |
| final-calibration count     | uint32  |       4 |
| validation-attack count     | uint32  |       4 |
| eligible-subtype count      | uint16  |       2 |
| flags                       | uint16  |       2 |
| config SHA-256              | 256-bit |      32 |
| **Total**                   |         | **896** |

Thus logical payload:

[
\boxed{
896\text{ bytes/client}
}
]

before serialization, headers, authentication, TLS, RPC, or transport overhead.

For nine clients:

[
9\times896
==========

\boxed{8,064\text{ bytes}}.
]

For 105 clients:

[
105\times896
============

\boxed{94,080\text{ bytes}}.
]

Measure actual serialized payload independently.

---

# 87. Server Response

There are 207 candidate indices.

Therefore selected index requires:

[
\lceil\log_2(207)\rceil
=======================

8
]

bits.

Thus one unsigned byte is sufficient for the selected candidate index.

Actual server response must also contain:

* allocation epoch ID;
* configuration hash;
* client ID;
* budget ID;
* integrity metadata.

---

# 88. Software Architecture

Use:

```text
fabrid/
    config/
        protocol.yaml
        alpha_grid.json
        datasets.yaml
        attack_folds.yaml
    data/
        partitioner.py
        eligibility.py
        provenance.py
        feature_manifest.py
    scoring/
        score_contract.py
        score_reader.py
    calibration/
        order_statistic.py
        final_calibration.py
    frontier/
        utility.py
        builder.py
        conservative.py
        stability.py
    allocation/
        equal_fpr.py
        equal_alert.py
        greedy.py
        fabrid_macro.py
        fabrid_minimax.py
        pooled_shared.py
        test_oracle.py
    optimization/
        milp.py
        lexicographic.py
        verifier.py
    evaluation/
        record_level.py
        workload.py
        heterogeneity.py
        eventization.py
    statistics/
        sign_flip.py
        bootstrap.py
        holm.py
    audit/
        split_leakage.py
        test_blindness.py
        score_identity.py
        budget_invariants.py
        determinism.py
        provenance.py
    schemas/
        score_artifact.py
        frontier.py
        allocation.py
        result.py
```

The allocation package must never import the model trainer.

---

# 89. Immutable Score Artifact

Persist exactly one artifact for every:

```text
dataset
× detector_seed
× client
× split
```

Minimum columns:

```text
sample_id
dataset_id
client_id
source_file
source_row
split_id
score
label
attack_type
timestamp
```

`timestamp` may be null only where the source lacks verified temporal provenance.

Persist:

* artifact SHA-256;
* model SHA-256;
* preprocessing SHA-256;
* split-manifest SHA-256;
* feature-manifest SHA-256;
* protocol SHA-256;
* Git commit.

---

# 90. Test-Blind Software Boundary

Non-oracle policy modules must never receive:

```text
ATTACK_TEST labels
ATTACK_TEST attack_type
BENIGN_TEST labels
test metrics
```

The API should make this impossible rather than merely relying on researcher discipline.

`TEST_ORACLE` must live in an isolated module.

Default execution must refuse oracle access.

---

# 91. Mandatory Scientific Software Tests

## T01 — Partition exclusivity

Any duplicate `sample_id` across partitions:

```text
FAIL
```

---

## T02 — Test-label perturbation

Randomly permuting test labels must leave every non-oracle allocation bitwise unchanged.

---

## T03 — Test-score perturbation

Changing final test scores must leave selected (\alpha_k) unchanged.

---

## T04 — Benign-test perturbation

Changing `BENIGN_TEST` must change neither allocation nor final thresholds.

---

## T05 — Final-calibration perturbation

Changing `BENIGN_FINAL_CAL`:

* may change (\tau_k);
* must not change selected (\alpha_k).

---

## T06 — Validation-attack perturbation

May change:

* `GREEDY`;
* `FABRID_MACRO`;
* `FABRID_MINIMAX`.

Must not change:

* `EQ_FPR`.

---

## T07 — Score hash identity

All policies within the same dataset/seed/client coordinate must reference the same score artifact hash.

---

## T08 — AUROC identity

[
|\Delta AUROC|<10^{-12}.
]

---

## T09 — Budget feasibility

[
\sum_kw_k\alpha_k
\le
B_{\mathrm{FP}}+10^{-12}.
]

---

## T10 — One target per client

[
\sum_jx_{k,j}=1.
]

---

## T11 — Brute-force solver parity

Synthetic case:

* 3 clients;
* 4 candidate points/client.

Enumerate:

[
4^3=64
]

possible allocations.

FABRID MILP must equal exact brute-force optimum.

---

## T12 — Determinism

Solve the identical allocation:

[
100
]

times.

Require:

[
100/100
]

identical selected vectors.

---

## T13 — Zero-budget behavior

For:

[
B=0,
]

all:

[
\alpha_k=0.
]

---

## T14 — Single-client behavior

For:

[
K=1,
]

FABRID reduces correctly to selection of one feasible operating point.

---

## T15 — Equal utility curves

When all clients have identical utility curves and equal weights, FABRID must not create an unexplained performance advantage over equal allocation.

---

## T16 — Monotonic budget feasibility

Increasing (B) must never make the previous optimum infeasible.

The optimal utility should therefore be nondecreasing up to numerical tolerance.

---

## T17 — Final-calibration resolution

Targets below:

[
1/(n+1)
]

must yield:

[
+\infty.
]

---

## T18 — Duplicate score ties

Strict `>` behavior must match hand-computed examples.

---

# 92. Pre-Execution Audit Gates

No confirmatory results may be viewed until these gates pass.

| Gate               | Requirement                              |
| ------------------ | ---------------------------------------- |
| `G01_PROTOCOL`     | protocol file frozen                     |
| `G02_REPO`         | clean commit recorded                    |
| `G03_CLIENTS`      | exactly 9 primary natural clients        |
| `G04_SPLITS`       | zero overlap                             |
| `G05_FEATURES`     | feature manifest frozen                  |
| `G06_MODEL`        | detector configuration frozen            |
| `G07_SCORE_LOCK`   | immutable score artifacts                |
| `G08_ALPHA_GRID`   | 207-target grid frozen                   |
| `G09_BUDGETS`      | five primary budgets frozen              |
| `G10_TEST_BLIND`   | non-oracle test access impossible        |
| `G11_SOLVER`       | brute-force parity                       |
| `G12_DETERMINISM`  | 100/100 identical                        |
| `G13_METRICS`      | formulas unit-tested                     |
| `G14_STATS`        | 1,024-sign implementation validated      |
| `G15_EXTERNAL`     | external eligibility rule frozen         |
| `G16_EVENT`        | provenance pass or event claims disabled |
| `G17_DEPENDENCIES` | environment locked                       |

---

# 93. Primary Result Schema

Every client-level result row must contain:

```text
experiment_id
dataset_id
seed
budget_id
budget_value
weight_mode
policy
client_id
alpha_selected
threshold
calibration_n
nominal_weight
realized_weight
n_benign_test
n_attack_test
attack_subtype
tp
fn
fp
tn
fpr
tpr
macro_attack_recall
false_alert_count
solver_status
solver_objective
solver_gap
solver_runtime_ms
model_sha256
score_sha256
split_sha256
feature_sha256
protocol_sha256
git_commit
```

No manuscript number is manually entered.

---

# 94. Reproducibility Metadata

Record:

* OS;
* Python version;
* NumPy version;
* SciPy version;
* PyTorch version;
* CUDA version;
* GPU model;
* CPU;
* RAM;
* HiGHS/SciPy solver version;
* Git commit;
* dataset checksums;
* exact CLI;
* wall-clock start/end;
* random seeds.

---

# 95. Primary Main Tables

## Table 1 — Literature boundary

Columns:

```text
Study
Multiple detectors/clients
Federated
Client-specific operating points
Shared alert constraint
Cross-client allocation
Detection utility used for allocation
Independent final calibration
Event-level evaluation
```

---

## Table 2 — Dataset populations

```text
Dataset
Natural clients
Eligible clients
Benign rows
Attack rows
Attack types
Physical/emulated
Timestamp provenance
Weight evidence level
```

---

## Table 3 — Matched-budget N-BaIoT

```text
Budget
Policy
MacroRecall
WorstClientRecall
MeanClientFPR
BUR
MaxClientFPR
CV_FPR
```

---

## Table 4 — Attack-subtype-disjoint

```text
Rotation
Policy
MacroRecall
WorstClientRecall
BUR
```

---

## Table 5 — External replication

Same primary metrics.

No dataset-specific metric replacement.

---

## Table 6 — System overhead

```text
K
utility payload
serialized upload
allocation runtime
peak memory
response bytes
```

---

# 96. Primary Figures

## Figure 1 — FABRID architecture

```text
frozen federated detector
          ↓
    local scores
          ↓
 BENIGN_FRONTIER ───────┐
                        ├─ local utility curve
ATTACK_VALIDATION ──────┘
          ↓
       server
          ↓
  FABRID allocation
          ↓
      alpha_k*
          ↓
BENIGN_FINAL_CAL
          ↓
       tau_k*
          ↓
       alerts
```

---

## Figure 2

[
B_{\mathrm{FP}}
\rightarrow MacroRecall.
]

---

## Figure 3

[
B_{\mathrm{FP}}
\rightarrow WorstClientRecall.
]

---

## Figure 4

At preregistered:

[
B_{\mathrm{FP}}=0.005,
]

show per client:

* selected (\alpha_k);
* utility;
* final FPR;
* recall.

---

## Figure 5

Allocation-sensitivity distribution of:

[
\alpha_k.
]

---

## Figure 6

Client utility curves:

[
u_k(\alpha).
]

This is scientifically important because it visually demonstrates the mechanism FABRID exploits.

---

## Figure 7

FABRID gain versus:

[
H_U.
]

---

# 97. Hostile Reviewer Audit

## Attack 1 — “This is just threshold tuning.”

**Response:** FABRID does not propose a new threshold estimator. It allocates target operating rates between federated clients under a shared constraint and calibrates final thresholds independently.

---

## Attack 2 — “Alert budgets already exist.”

**Response:** Correct; CALIBURN and event-level IDS budget evaluation are acknowledged directly. FABRID studies **cross-client redistribution of one budget**, not invention of budgeted IDS.

---

## Attack 3 — “Multiple-detector threshold control already exists.”

**Response:** Correct; Bridges et al. are explicit prior art. FABRID's remaining contribution is the federated client-specific utility-allocation formulation.

---

## Attack 4 — “Federated anomaly thresholds already exist.”

**Response:** Correct; Laridi et al. and WAFL threshold work are cited. FABRID does not claim threshold-estimation novelty.

---

## Attack 5 — “This is just MILP.”

**Response:** MILP is standard. The contribution is the security decision problem, protocol, separation of stages, matched-budget experiment, and empirical evidence.

---

## Attack 6 — “Your traffic weights are fake.”

**Response:** Primary N-BaIoT uses equal-client weights because common-duration operational rates are not established. Dataset-row weighting is explicitly sensitivity-only.

---

## Attack 7 — “Equal alert and equal FPR are the same.”

**Response:** Correct under equal weights; therefore they are not duplicated in the primary experiment.

---

## Attack 8 — “Your unsupervised detector uses attack labels.”

**Response:** Detector training is benign-only; FABRID allocation is explicitly described as validation-informed.

---

## Attack 9 — “You optimize and evaluate on the same attacks.”

**Response:** 20% attack-validation and 80% final attack-test partitions are disjoint; an additional attack-subtype-disjoint protocol holds out complete attack types.

---

## Attack 10 — “Your threshold is selected and calibrated on the same benign data.”

**Response:** Impossible by design. `BENIGN_FRONTIER` and `BENIGN_FINAL_CAL` are disjoint.

---

## Attack 11 — “Finite-sample guarantee assumes IID traffic.”

**Response:** The manuscript explicitly does not claim unconditional IID guarantees on N-BaIoT; the data are sequential and use overlapping temporal features.

---

## Attack 12 — “The optimizer exploits validation noise.”

**Response:** 500-resample allocation sensitivity and lower-confidence-bound utility analysis quantify the issue.

---

## Attack 13 — “A tiny client receives a ridiculous FPR.”

**Response:**

[
\alpha_k\le5%.
]

---

## Attack 14 — “Average recall sacrifices one client.”

**Response:** `FABRID_MINIMAX` and WorstClientRecall directly address that failure mode.

---

## Attack 15 — “AUROC does not improve.”

**Response:** It should not improve. FABRID changes the decision operating point, not the score representation.

---

## Attack 16 — “Record FPR is not analyst workload.”

**Response:** Correct. Record-level results are labelled accordingly; operational event claims require the separate event-provenance gate.

---

## Attack 17 — “Your event timestamps are fabricated.”

**Response:** Explicitly prohibited. No event analysis without raw/provenance-supported client time.

---

## Attack 18 — “Utility sharing is privacy-preserving.”

**Response:** No formal privacy claim is made.

---

## Attack 19 — “A malicious client can lie about its utility.”

**Response:** Honest execution is part of the threat model; Byzantine allocation manipulation is out of scope.

---

## Attack 20 — “FABRID already exists.”

**Response:** The acronym already exists in unrelated inter-domain routing research, which is why the public research identity is **FABRID-IDS**.

---

# 98. Mandatory Ablations

The final confirmatory study shall include:

1. `FABRID_MACRO` vs `EQ_FPR`;
2. `FABRID_MINIMAX` vs `EQ_FPR`;
3. `FABRID_MACRO` vs `GREEDY`;
4. five-budget sweep;
5. client utility-curve visualization;
6. utility-heterogeneity analysis;
7. attack-subtype-disjoint validation;
8. optional botnet-family-disjoint validation;
9. raw-utility vs conservative-utility FABRID;
10. allocation-sensitivity analysis;
11. equal-client vs justified alternative weight sensitivity;
12. centralized pooled-score diagnostic;
13. external physical-device replication;
14. event-level workload validation where provenance passes.

Do not add unrelated poisoning attacks or FL defenses to enlarge the experiment.

---

# 99. Negative Result Policy

If FABRID does not win:

* do not change budgets;
* do not remove seeds;
* do not replace the primary metric;
* do not eliminate difficult clients;
* do not alter attack folds;
* do not retune the detector;
* do not change the local FPR cap;
* do not select a different external dataset because it gives better results.

A valid conclusion could be:

> **Under the measured client utility heterogeneity, equal-FPR allocation is already near-optimal and cross-client reallocation provides limited additional value.**

That remains publishable evidence if the study is rigorous.

---

# 100. Publication Success Conditions

A strong positive paper requires all of the following:

### Internal validity

* fixed detector;
* identical score artifacts;
* no test leakage;
* independent final calibration;
* deterministic optimizer;
* valid budget constraints.

### Primary effect

`FABRID_MACRO` satisfies its 2-percentage-point practical gate at ≥3/5 budgets.

### Worst-client evidence

`FABRID_MINIMAX` satisfies its 5-percentage-point worst-client gate at ≥3/5 budgets without >2-point Macro Recall degradation.

### Budget behavior

≥9/10 seeds satisfy:

[
BUR\le1.10.
]

### Generalization

Effect direction survives attack-subtype-disjoint testing.

### External evidence

Direction reproduces on a second physical-device population.

---

# 101. Claims Permitted Before Results

The paper may state:

> **FABRID formulates the allocation of client operating points in federated IoT anomaly detection as a shared nominal false-alert budget problem.**

It may state:

> **FABRID selects client-specific target operating rates from locally estimated detection-utility curves while keeping the underlying detector fixed.**

It may state:

> **FABRID separates operating-rate allocation from independent final local threshold calibration.**

---

# 102. Claims Requiring Positive Results

Only after evidence exists:

> “FABRID improves Macro Recall under matched false-alert budgets.”

Only after minimax evidence exists:

> “FABRID improves worst-client detection without material average-performance loss.”

Only after external replication:

> “The observed benefit generalizes across independently structured IoT device populations.”

---

# 103. Forbidden Claims

Do not claim:

* “first alert-budget IDS”;
* “first federated thresholding method”;
* “first heterogeneous detector threshold allocator”;
* “first IDS resource allocator”;
* “novel MILP optimizer”;
* “privacy-preserving”;
* “differentially private”;
* “guaranteed SOC workload”;
* “guaranteed future FPR under arbitrary traffic”;
* “fully unsupervised FABRID”;
* “zero-day guarantee”;
* “real-world deployment validated”;
* “end-to-end optimal”;
* “globally optimal security”;
* “robust to malicious clients”;
* “concept-drift proof.”

---

# 104. Recommended Manuscript Contribution Statement

> **FABRID-IDS addresses a post-training decision problem in federated IoT anomaly detection: how a finite nominal false-alert allowance should be distributed across heterogeneous clients. Rather than assigning every client a common false-positive-rate target, FABRID uses locally estimated detection–budget utility curves to select client-specific target operating rates subject to one federation-level constraint. Allocation is performed without modifying the underlying detector, and the selected target rate is subsequently converted into a deployed threshold using an independent local benign calibration partition. This design permits matched-budget comparisons on identical anomaly-score artifacts and enables explicit evaluation of average detection utility, worst-client protection, allocation stability, attack-type generalization, client heterogeneity, and operational event workload where timestamp provenance permits.**

---

# 105. Recommended Abstract-Level One-Sentence Gap

> **Existing research addresses anomaly threshold selection, federated calibration, alert budgeting, and heterogeneous detector management; FABRID instead asks how one common false-alert allowance should be redistributed across heterogeneous federated IDS clients whose marginal detection benefit from additional alert capacity differs.**

---

# 106. Publication Positioning

The recommended first target is:

> **Internet of Things — Elsevier**

because the manuscript naturally combines IoT security, intelligent edge/federated operation, and system-level security decision making.

The recommended second target is:

> **Computer Networks — Elsevier**

particularly if the final manuscript emphasizes network-security operations and constrained resource allocation.

Venue scopes must be rechecked immediately before submission because editorial policies can change.

---

# 107. Repository Identity

Use:

```text
Repository:
fabrid-ids
```

Recommended top-level repository structure:

```text
fabrid-ids/
    README.md
    LICENSE
    CITATION.cff
    pyproject.toml
    requirements.lock
    protocol/
    fabrid/
    tests/
    configs/
    manifests/
    scripts/
    results/
    figures/
    docs/
```

---

# 108. Recommended README Heading

```markdown
# FABRID-IDS

Federated Alert-Budget Reallocation for Heterogeneous IoT Intrusion Detection
```

First sentence:

> FABRID-IDS is a research framework for allocating a shared nominal false-alert allowance across heterogeneous federated anomaly-detection clients while keeping the underlying detector fixed.

---

# 109. Implementation Roadmap

## Phase 0 — Freeze research identity

Commit:

```text
method_name = FABRID
public_name = FABRID-IDS
repository  = fabrid-ids
paper_title = FABRID-IDS: Federated Alert-Budget Reallocation for Heterogeneous IoT Intrusion Detection
```

---

## Phase 1 — Freeze protocol

Create:

```text
protocol/protocol_v2.yaml
protocol/alpha_grid.json
protocol/attack_folds.yaml
protocol/metrics.yaml
```

Record SHA-256 hashes.

No confirmatory experiment before this commit.

---

## Phase 2 — Dataset provenance

Implement:

* nine-client N-BaIoT inventory;
* canonical device IDs;
* canonical attack IDs;
* source-row provenance;
* deterministic source-order partitioning;
* overlap validation.

---

## Phase 3 — Detector freeze

Train exactly:

[
10
]

detector seeds.

Persist:

* model;
* scaler/preprocessor;
* configuration;
* hashes.

---

## Phase 4 — Generate immutable scores

Score every:

```text
client × split × seed
```

exactly once.

After hashing, policies can only read score files.

---

## Phase 5 — Calibration implementation

Implement:

* order-statistic index;
* (+\infty) behavior;
* strict `>`;
* duplicate-score behavior;
* target-resolution checks.

---

## Phase 6 — Utility frontier implementation

Generate the fixed:

[
207
]

target-rate points.

For every:

```text
seed × client × alpha
```

calculate:

* provisional threshold;
* subtype TPR;
* Macro subtype utility;
* target budget cost.

---

## Phase 7 — Implement baselines first

Freeze:

1. `EQ_FPR`;
2. `GREEDY`;
3. `POOLED_SHARED`.

Do this before viewing FABRID performance.

---

## Phase 8 — Implement `FABRID_MACRO`

Require:

* MILP solver;
* brute-force parity;
* budget invariant;
* deterministic tie resolution.

---

## Phase 9 — Implement `FABRID_MINIMAX`

Require:

* Stage-1 max-min solve;
* Stage-2 Macro solve;
* budget minimization;
* deterministic final tie break.

---

## Phase 10 — Independent final calibration

For every selected target:

* load only `BENIGN_FINAL_CAL`;
* compute (\tau_k);
* hash threshold artifact.

---

## Phase 11 — Test-blindness audit

Run T01–T18.

Stop immediately on any failure.

---

## Phase 12 — Main N-BaIoT experiment

Run:

* 10 seeds;
* 5 budgets;
* four deployable policies;
* pooled diagnostic;
* oracle diagnostic.

No detector retraining.

---

## Phase 13 — Primary statistics

Generate:

* paired effects;
* exact 1,024 sign-flip tests;
* 50,000 paired bootstrap replicates;
* Holm correction.

---

## Phase 14 — Attack-subtype-disjoint study

Run all three fixed fold rotations.

Reuse frozen detectors.

---

## Phase 15 — Family-disjoint study

For the seven clients supporting both families:

* BASHLITE → Mirai;
* Mirai → BASHLITE.

---

## Phase 16 — Allocation sensitivity

Run:

[
500
]

resampled utility/frontier allocations per seed and budget.

---

## Phase 17 — Conservative utility analysis

Resolve FABRID using the 95% one-sided subtype-recall lower-bound utility curves.

---

## Phase 18 — Weight-provenance sensitivity

Primary:

```text
EQUAL_CLIENT
```

Secondary only where meaningful:

```text
DATASET_COUNT_PROXY
OPERATIONAL_RATE
```

Never combine them without labels.

---

## Phase 19 — CIC IoT-DIAD readiness audit

Determine:

* eligible devices;
* feature leakage exclusions;
* attack counts;
* benign counts;
* usable client IDs;
* feature manifest.

Freeze eligible population before comparing policies.

---

## Phase 20 — External replication

Repeat the five-budget protocol with no outcome-driven retuning.

---

## Phase 21 — Event dataset readiness

Audit Gotham and/or CICIoMT2024.

Choose the event dataset based on provenance suitability, not on which gives FABRID better results.

Record the selection rationale before policy outcomes are generated.

---

## Phase 22 — Event-level evaluation

If gate passes:

* eventize alerts;
* enforce duty cap;
* run 0.1/0.2/0.5 events/client/hour;
* execute the 81-configuration post-processing sensitivity at 0.2 events/client/hour.

---

## Phase 23 — Final figures and tables

Generate all manuscript figures programmatically from immutable result artifacts.

No manually entered numbers.

---

## Phase 24 — Reproduction audit

On a clean environment:

1. install dependencies;
2. acquire datasets;
3. verify dataset hashes;
4. build manifests;
5. reproduce at least one seed end-to-end;
6. reproduce all post-training allocation results from frozen score artifacts.

---

## Phase 25 — Final novelty search

Immediately before submission, repeat searches for literature published after:

> **12 August 2026**

using at least:

```text
FABRID intrusion detection
federated alert budget
federated false-alert allocation
federated false-alarm budget
federated operating-point allocation
cross-client intrusion detection allocation
federated IDS workload
heterogeneous client alert allocation
resource-aware federated IDS
```

Update the contribution language if a direct collision appears.

---

# 110. Definition of Done

FABRID is scientifically implementation-complete only when:

* [ ] project identity is frozen as FABRID / FABRID-IDS;
* [ ] repository is named `fabrid-ids`;
* [ ] protocol is committed before confirmatory results;
* [ ] exactly nine N-BaIoT natural clients are used;
* [ ] benign splits follow the exact deterministic boundary rule;
* [ ] attack splits are disjoint;
* [ ] ten detector seeds are complete;
* [ ] all policies reuse identical detector and score artifacts;
* [ ] 207-point target-rate grid is frozen;
* [ ] five record-level budgets are evaluated;
* [ ] allocation and final calibration are independent;
* [ ] all test-leakage tests pass;
* [ ] MILP equals brute-force optimum on synthetic cases;
* [ ] deterministic tie-breaking passes 100 repeated runs;
* [ ] `FABRID_MACRO` is complete;
* [ ] `FABRID_MINIMAX` is complete;
* [ ] attack-subtype-disjoint experiment is complete;
* [ ] allocation-sensitivity experiment is complete;
* [ ] conservative-utility sensitivity is complete;
* [ ] equal-client versus alternative-weight semantics are explicitly separated;
* [ ] external physical-device replication is attempted under the frozen protocol;
* [ ] event-level claims are made only after provenance validation;
* [ ] exact seed-level statistical tests are complete;
* [ ] negative results are retained;
* [ ] all output artifacts are hashed;
* [ ] final literature collision search is complete;
* [ ] manuscript claims remain inside the evidence boundary.

---

# 111. Final Scientific Position

FABRID should be presented as neither an anomaly-detection architecture nor a threshold estimator.

Its core scientific premise is:

[
\boxed{
\text{A common false-alert allowance is a federation-level resource}
}
]

and:

[
\boxed{
\text{different clients can produce different detection benefit from the same additional alert allowance.}
}
]

Therefore the relevant optimization variable is not the model parameter vector.

It is:

[
\boxed{
\boldsymbol{\alpha}
===================

(\alpha_1,\ldots,\alpha_K).
}
]

The central causal experiment is:

[
\boxed{
\text{same detector}
+
\text{same scores}
+
\text{same total budget}
+
\text{different allocation}
}
]

followed by measurement of:

[
\boxed{
\text{detection utility and client protection}.
}
]

That is the scientifically defensible identity of FABRID.

---

# 112. Final Approved Identity

**Method**

> **FABRID**

**Expansion**

> **Federated Alert-Budget Reallocation for Intrusion Detection**

**Public research identifier**

> **FABRID-IDS**

**Final manuscript title**

> **FABRID-IDS: Federated Alert-Budget Reallocation for Heterogeneous IoT Intrusion Detection**

**GitHub repository**

```text
fabrid-ids
```

**Recommended package**

```text
fabrid
```

**Primary algorithms**

```text
FABRID_MACRO
FABRID_MINIMAX
```

**Primary baseline**

```text
EQ_FPR
```

**Final research decision**

> **GO — implement FABRID under this frozen specification, preserve the narrow novelty boundary, and do not relax the allocation/calibration separation, equal-client N-BaIoT budget semantics, test-blindness contract, or matched-budget comparison after confirmatory experiments begin.**
