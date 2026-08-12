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
