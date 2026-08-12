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

Follow-up from user feedback mid-session: renamed opaque `i1/i2/i3/n` boundary fields to descriptive
names (`train_end`/`frontier_end`/`final_cal_end`/`total_rows`), replaced hardcoded split-fraction
module constants with a typed `Protocol`/`BenignSplitFractions`/`AttackSplitFraction` loader reading
the single canonical `protocol.yaml`, added `RowCount`/`RowIndex` `NewType`s to avoid raw-int primitive
leakage, and removed all "roadmap section N" references from code/test docstrings and comments
(kept only in `docs/` where such references belong).
