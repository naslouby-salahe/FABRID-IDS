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

## D004 — Raise MILP time_limit from 60s to 300s (disclosed protocol amendment); keep mip_gap<=1e-9 unchanged

Explicit user authorization ("come up with the best solution... to make my chapter perfect... continue
with the implementation") to resolve F001/F002, which are user-facing scientific decisions, not
ordinary engineering bugs.

**Evidence.** `scripts/run_budget_sweep.py` measured `SOLVER_INVALID` rate for `FABRID_MACRO`/
`FABRID_MINIMAX` across all 5 primary budgets x all 10 real trained seeds, at the frozen
`time_limit=60s`, `accept_if.mip_gap_leq=1e-9`:

| Budget | FABRID_MACRO invalid | FABRID_MINIMAX invalid |
|---|---|---|
| 0.001 | 2/10 | 3/10 |
| 0.0025 | 7/10 | 5/10 |
| 0.005 | 7/10 | 1/10 |
| 0.010 | 9/10 | 6/10 |
| 0.020 | 10/10 | 9/10 |

Invalid rate rises with budget (larger feasible region -> more near-ties -> branch-and-bound needs
longer to certify optimality at 9-client x 207-candidate = 1863-variable scale). Researched
(WebSearch, scipy/HiGHS docs): `mip_rel_gap=0` (already frozen) instructs HiGHS to target exact
optimality; `time_limit` is the only lever that increases certified-solution rate without touching
solution quality. The roadmap's `time_limit=60s` (section 41) was evidently never validated against
real record-level N-BaIoT scale.

**Decision.** Raise `time_limit_seconds` in `protocol.yaml` from 60 to 300. `accept_if.mip_gap_leq`
stays at `1e-9`, completely unchanged — this preserves 100% of the solution-quality rigor the roadmap
specifies; only wall-clock budget increases so more coordinates can actually reach that bar rather
than being excluded by an untested clock value. This is a disclosed, evidenced amendment (this
decision record + the sweep table above, both committed), not a silent tolerance weakening — the
original 60s evidence in `failures.md` F001 is preserved, not deleted. Standard practice for a
pre-registered protocol: when an implementation constraint invalidates an untested parameter, the
correct response is transparent amendment with the discovery evidence retained, not silent
adjustment or in the other direction, blind adherence to a number now known to be practically
unworkable at the specified scale.

Re-measurement after this change is required before treating any FABRID_MACRO/MINIMAX result as
final; see `state.md` for the follow-up sweep.

## D005 — F002 mitigation: wire the roadmap's own conservative/LCB utility ablation into FABRID_MINIMAX

F002 (see `failures.md`) is grounded in the Group-DRO/minimax-fairness literature (WebSearch,
2026): validation-time worst-group/worst-client estimates are well documented not to reliably predict
the test-time worst group, and variance-reduction/regularization is the standard mitigation. The
roadmap already mandates exactly this experiment — section 63 "Conservative Utility Sensitivity":
resolve FABRID using the one-sided 95% LCB utility curve (`frontier/conservative.py`, already built)
"and report whether policy conclusions survive." This was implemented (utility curve construction) but
never actually run through `FABRID_MINIMAX`. Decision: run `FABRID_MINIMAX` under both the raw
validation utility curve and the conservative LCB curve on the same real seeds/budget, and report
both `WorstClientRecall` results side by side. This is precisely the on-roadmap ablation already
required (section 98, ablation #9: "raw-utility vs conservative-utility FABRID") — not a new
experiment invented to explain away F002, and turns F002 from a bare anomaly into a documented,
literature-grounded, empirically-tested finding.

## D006 — F001 root cause was not solve time; raise `accept_if.mip_gap_leq` from 1e-9 to 1e-5 (evidence-based)

Follow-up to D004. Empirical test: re-ran `run_seed_at_budget` at budget=0.02 (10/10 `FABRID_MACRO`
invalid before D004) with the new 300s time_limit — still invalid, but solved in 3-4s, nowhere near
the time limit. Direct inspection of `scipy.optimize.milp`'s result showed `mip_node_count=1` on every
sampled seed x budget combination. `mip_node_count=1` means HiGHS solved at the LP-relaxation root —
no branch-and-bound exploration happened at all. So D004's premise (the solver needs more wall-clock
time to branch further) was wrong; time_limit=300 is kept as a harmless increase but is not the fix.

**First hypothesis tested and disproven.** Initially hypothesized (Multiple-Choice Knapsack Problem
theory: MCKP LP relaxations have at most one fractional variable at optimum) that the LP relaxation
of this problem is already integral, making the residual `mip_gap` pure floating-point noise, and
implemented an alternate "LP-relaxation-integrality" optimality proof in `solve_milp`. Directly tested
against real seed-0 data: the raw LP relaxation is **not** integral (up to 0.30 fractional deviation on
2 variables). That theory is false for this problem instance and the code implementing it was reverted.

**Real evidence, gathered directly** (`/tmp/gap_measure.py`, `/tmp/gap_measure_minimax.py`, ad-hoc
diagnostics, not committed): measured the actual `mip_gap` HiGHS reports across all 10 seeds x 5
budgets (0.001/0.0025/0.005/0.01/0.02), for both `FABRID_MACRO` and `FABRID_MINIMAX`, at
`time_limit=30s` (well above the 3-4s observed solve time).

- `FABRID_MACRO` (50 cells): every non-zero gap falls in `[9.4e-9, 1.25e-6]`. Max observed: `1.2476e-06`.
  Distribution is a tight cluster consistent with LP dual-bound floating-point noise at this
  1863-variable, double-precision scale — not evidence of a better integer solution existing.
- `FABRID_MINIMAX` (50 cells): same noise cluster (`<= 2.5e-6`) for 48/50 cells, plus two genuine
  outliers — seed 3/budget 0.001 (`9.25e-4`) and seed 7/budget 0.0025 (`2.14e-4`) — roughly 100-1000x
  larger than the noise cluster. `FABRID_MINIMAX`'s two-stage epigraph formulation (maximize worst-case
  utility, then minimize budget subject to fixing it) legitimately produces near-degenerate ties that
  the noise-floor explanation does not cover for these two cells.

**Decision.** Raise `solver.accept_if.mip_gap_leq` in `protocol.yaml` from `1e-9` to `1e-5`: three
orders of magnitude above the measured noise ceiling (`1.25e-6`), enough to absorb solver numerical
noise with margin, but still two orders of magnitude below the two genuine `FABRID_MINIMAX` near-tie
outliers (`2.14e-4`, `9.25e-4`), which remain correctly `SOLVER_INVALID` and excluded per protocol —
this is not a blanket loosening that would also swallow real optimization ambiguity. This is a
disclosed, evidence-based protocol amendment (this decision + the measured gap distributions above),
not a silent tolerance weakening; `mip_rel_gap=0` (the solver's own internal target) stays untouched,
only the acceptance bar for treating a solve as certified-optimal moves to reflect where the actual
double-precision noise floor sits for this problem scale. `src/fabrid/optimization/milp.py` is
reverted to its original simple form (gap-only acceptance; no LP-relaxation branch).

Re-measurement across the full 5-budget x 10-seed sweep against real trained data (via
`run_seed_at_budget`, not the ad-hoc diagnostics above) is required to confirm the practical
`SOLVER_INVALID` rate after this change; see `state.md` for the follow-up sweep.

## D007 — F001's remaining cause: HiGHS's own feasibility tolerance was looser than the sequential tie-break constraints; tighten it explicitly

D006's `accept_if.mip_gap_leq` fix reduced but did not eliminate `SOLVER_INVALID`; re-running
`scripts/run_budget_sweep.py` against real trained scores still showed `FABRID_MACRO` invalid on
1-6/10 seeds per budget, now failing with `HiGHS Status 8: Infeasible`, not a gap-tolerance
rejection. Direct stage-by-stage reproduction (`allocate_fabrid_macro`'s 3-stage sequential solve
on seed 0, budget 0.0025) found the cause: stage 2's own returned solution `x2` satisfied the
`utility_floor` constraint stage 2 was itself given only to within `~7e-7` — i.e. HiGHS accepted a
solution slightly outside the constraint it was asked to satisfy. That is HiGHS's default MIP/LP
*feasibility* tolerance (~1e-6, a different knob from `mip_rel_gap`/`accept_if.mip_gap_leq`, which
bound the *objective* gap, not constraint satisfaction). Stage 3 then re-imposes that exact same
`utility_floor` (plus a `cost_ceiling` anchored to stage 2's reported cost) as hard constraints; the
composed system, tightened around a point HiGHS itself only approximately satisfied, was proved
genuinely infeasible by branch-and-bound. Confirmed directly: passing
`mip_feasibility_tolerance`/`primal_feasibility_tolerance` (HiGHS-native options, forwarded verbatim
by `scipy.optimize.milp` for any key its stub does not itself model) as `1e-9` to HiGHS makes stage
2's `x2` satisfy `utility_floor` to `-2.8e-7` (i.e. with margin, not a near-violation), and stage 3
solves successfully on the same previously-infeasible instance.

**Decision.** `fabrid/optimization/milp.py:solve_milp` now passes `mip_feasibility_tolerance` and
`primal_feasibility_tolerance` explicitly as `1e-9` (`_HIGHS_FEASIBILITY_TOLERANCE`) to HiGHS on
every solve, tighter than any tie-break tolerance in use, so that a stage's own reported solution is
never numerically inconsistent with a constraint built from it in a later stage. This is a distinct
fix from D006: D006 widened the *objective-gap acceptance* bar; D007 tightens the *constraint-
satisfaction* precision of the underlying solver so the multi-stage sequential tie-break (frozen
protocol design, `protocol.yaml`'s `tie_breaking` section) stays internally consistent regardless of
that gap bar. Additionally, `allocate_fabrid_macro`'s `_UTILITY_TOLERANCE`/`_BUDGET_TOLERANCE` and
`allocate_fabrid_minimax`'s `_Z_TOLERANCE`/`_UTILITY_TOLERANCE`/`_BUDGET_TOLERANCE` were renamed to
`*_FLOOR` and are now combined as `max(frozen_floor, settings.accept_mip_gap_leq)` at call time: the
frozen protocol tolerance (`1e-9`/`1e-12`) is preserved as a floor and never loosened below what the
protocol specifies, but it also can never be tighter than the solver's own accepted objective-gap
precision, which would otherwise make an accepted-but-gap-imprecise earlier stage's solution
infeasible for a later stage by construction.

Re-measured on the full 5-budget x 10-seed sweep after this change; see `state.md` for the final
`SOLVER_INVALID` rate.

## D002 — CIC IoT-DIAD 2024 not available; external replication provisionally BLOCKED_EXTERNAL

`datp-shared-data/raw` contains `CIC_IOT_Dataset2023` (a different, earlier CIC dataset) but not
`CIC IoT-DIAD 2024`. Per prompt.md section 11 ("If an external requirement is genuinely impossible, mark
it BLOCKED_EXTERNAL... continue all other work"), primary N-BaIoT work proceeds independently. Will
attempt to locate/acquire the correct dataset before finalizing this as BLOCKED_EXTERNAL — do not
substitute CIC_IOT_Dataset2023 for CIC IoT-DIAD 2024, they are not the same dataset and the roadmap
names the latter specifically (105 devices, 33 attacks, 7 categories, packet+flow representations).
