# Progress Log

## 2026-08-12 — Session 1 (Phase 0-1)

- Read `prompt.md` and the full FABRID-IDS roadmap (v2.0) end to end.
- Verified shared raw data at `/home/naslouby/Projects/datp-shared-data/raw`: N-BaIoT present
  (9 device directories), CIC_IOT_Dataset2023 present (not the same as CIC IoT-DIAD 2024 required
  for external replication), Edge-IIoTset present. CIC IoT-DIAD 2024, Gotham 2025, CICIoMT2024 not present.
- Discovered `/home/naslouby/Projects/datp-core` and `/home/naslouby/Projects/datp` as prior mature
  research stacks with federated detector training, scoring, checkpointing, and threshold/calibration
  modules — the "existing DATP experimental stack" referenced in roadmap section 18. Plan to depend on
  datp-core for frozen detector/score generation rather than reimplementing it inside fabrid-ids.
- Created `data/raw` symlink to shared raw data.
- Scaffolded `fabrid/` package per roadmap section 88 architecture, `pyproject.toml`.
- Implemented and froze Phase 0 identity constants (`fabrid/__init__.py`).
- Implemented Phase 1 protocol artifacts:
  - `fabrid/config/alpha_grid.py` — exact section-29 log-spaced + explicit-point grid construction,
    dedup at 1e-12 tolerance, frozen `AlphaGrid` dataclass enforcing count==207 and sortedness.
  - `fabrid/config/alpha_grid.json` — generated and verified (207 unique values).
  - `fabrid/config/protocol.yaml` — full frozen protocol (score contract, seeds, budgets, split
    fractions, finite-sample calibration formula, solver settings, tie-breaking sequences, statistics
    settings, practical gates, event gate parameters).
  - `fabrid/config/attack_folds.yaml` — section 58 fixed fold mapping + rotations, section 59
    botnet-family-disjoint directions.
  - `fabrid/config/datasets.yaml` — exact section 22/25 N-BaIoT 9-client table with exact split counts;
    CIC IoT-DIAD 2024 eligibility contract recorded with status `candidate_external_replication`.
- Created restart tracking folder (`docs/tmp/fabrid-implementation/`) and audit matrix skeleton.

Next: Phase 2 dataset provenance (partitioner + eligibility + exclusivity tests), after confirming
exact reuse boundary with datp-core's N-BaIoT reader.
