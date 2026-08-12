# Failures

## F001 — FABRID_MACRO hits SOLVER_INVALID at full real-data scale (60s time limit, 1863 binary vars)

Running `scripts/run_main_comparison.py 0` against the real persisted seed-0 score artifacts (9
clients, 207-candidate grid = 9*207=1863 binary variables) raised `SolverInvalidError` on
`allocate_fabrid_macro`'s stage-1 solve: `success=True, status=0, mip_gap=1.247625437566232e-06`.
This exceeds the frozen `accept_mip_gap_leq=1e-9` threshold (`protocol.yaml:solver.accept_if.mip_gap_leq`),
so the solver result was correctly rejected rather than silently accepted as an optimum.

This is the protocol working as specified, not a bug: "Accept a FABRID optimization result only if...
mip_gap <= 1e-9. Otherwise mark SOLVER_INVALID and exclude the coordinate pending investigation. Never
silently use a time-limited feasible solution as an optimum." At this problem size, HiGHS reaches a
gap around 1e-6 quickly but apparently needs longer than the 60s time limit to close the remaining gap
to 1e-9.

Root cause classification: environment/scale, not code or scientific-contract. No fix applied — per
prompt.md, do not "fix" by weakening the acceptance tolerance or the time limit ad hoc. Both are frozen
protocol values (`protocol.yaml`).

Resolution path for the real confirmatory run (not yet applied, needs a decision before Phase 12):
either (a) accept that some seed/budget coordinates at full N-BaIoT scale will legitimately be
`SOLVER_INVALID` and report them as such per the roadmap's own excluded-pending-investigation clause,
or (b) if a systematic pattern emerges, investigate whether HiGHS solver options (e.g. explicit
`mip_rel_gap` passed to HiGHS directly vs. scipy's wrapper, warm-starting between the four sequential
stages) can close the gap faster within 60s — an implementation/tuning question, not a protocol
change. Recorded here rather than silently retried with loosened tolerances.

Practical implication for time budgeting: the full main experiment (10 seeds x 5 budgets x
{FABRID_MACRO: 3 sequential solves, FABRID_MINIMAX: 4 sequential solves}) could take substantially
longer than the smoke/single-comparison runs suggested, if a meaningful fraction of coordinates need
the full 60s time limit per solve. Needs to be measured, not assumed, before scheduling Phase 12.

**Update after running `scripts/run_contrasts.py` across all 10 real trained seeds at budget=0.01**:
confirmed systemic, not a one-off. `FABRID_MACRO` was `SOLVER_INVALID` in 9/10 seeds (only seed 6
succeeded); `FABRID_MINIMAX` was `SOLVER_INVALID` in 6/10 seeds. Contrast A (`FABRID_MACRO - EQ_FPR`
on MacroRecall) therefore has only n=1 usable seed — far short of the paired 10-seed design the
roadmap's statistics assume. This must be resolved (solver tuning, or accepting a materially smaller
n with the loss explicitly reported) before any confirmatory claim can be made from Contrast A. Not
attempted yet.

## F002 — FABRID_MINIMAX's budget-minimization tie-break can crater a non-bottleneck client's real
   test recall, even though it satisfies its own (validation-time) minimax objective

Real finding from `results/scores/seed_6` (the one seed where `FABRID_MACRO`/`FABRID_MINIMAX` both
solved to `mip_gap<=1e-9`): under `FABRID_MINIMAX` at budget=0.01,
`SimpleHome_XCS7_1003_WHT_Security_Camera` — which gets `alpha=0.01` and **test recall 1.0** under
`EQ_FPR` — is allocated `alpha≈0.00026` (effectively nothing) under `FABRID_MINIMAX`, driving its
final threshold to `+inf` and its **test recall to 0.0**. This makes `FABRID_MINIMAX`'s
`WorstClientRecall` on this seed *worse* than `EQ_FPR`'s (contrast mean_diff=-0.4503 across the 4
seeds where both solved), the opposite of what `FABRID_MINIMAX` is meant to deliver.

This is not a code bug — the implementation matches the roadmap's exact 4-stage tie-break (section
39/42: max min-utility z; fix z; max mean utility; fix; **minimize total budget**; lexicographically
minimize alpha). The mechanism: `SimpleHome-1003`'s *validation*-time utility curve is not what binds
the minimum `z*` — some other client is the true validation-time bottleneck. Once `z*` and the
achieved mean utility are fixed, stage 3 minimizes total weighted budget among all equally-optimal
solutions, and since `SimpleHome-1003`'s alpha does not affect either binding constraint, the solver
correctly (per its literal objective) pushes it toward the cheapest allocation — here, essentially
zero. `FABRID_MINIMAX` guarantees the *validation-utility* minimum reaches `z*`; it makes no promise
about any specific client's *test* recall, and validation-vs-test divergence for one client is enough
to produce this outcome.

This is exactly the class of finding roadmap sections 62 (allocation-sensitivity), 63 (conservative/
LCB utility), and Attack 12 ("the optimizer exploits validation noise") anticipate and are designed to
diagnose — not a defect to patch. No change made to `fabrid_minimax.py`; the roadmap's exact
formulation was followed. Flagging for explicit awareness: at the current record-level, single-budget,
n=4 sample size, `FABRID_MINIMAX` does not yet show the worst-client protection benefit the roadmap
hypothesizes it should provide, and per section 99 (Negative Result Policy) this must be reported
as-is if it persists in the full confirmatory run — do not adjust budgets, folds, or the objective
after seeing this.
