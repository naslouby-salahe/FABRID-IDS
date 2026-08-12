# State

canonical roadmap path: `docs/FABRID-IDS Roadmap.md` (v2.0, protocol date 2026-08-12)
current git commit (as of last update): 8eefad8 (pre-checkpoint; this chunk not yet committed)
current roadmap phase: Phase 1 (Freeze protocol) complete; entering Phase 2 (dataset provenance)
current requirement/group: DATASET-*, CLIENT-*, SPLIT-* (N-BaIoT partitioner + provenance)

last completed major implementation chunk:
- Phase 0 identity freeze (`src/fabrid/__init__.py`)
- Phase 1 protocol freeze: `src/fabrid/config/protocol.yaml`, `src/fabrid/config/alpha_grid.json`
  (207 values, generated+verified via `src/fabrid/config/alpha_grid.py`), `src/fabrid/config/attack_folds.yaml`,
  `src/fabrid/config/datasets.yaml`
- Repository scaffold per roadmap section 88 architecture (`src/fabrid/{config,data,scoring,calibration,
  frontier,allocation,optimization,evaluation,statistics,audit,schemas}`)
- `data/raw` symlink -> `/home/naslouby/Projects/datp-shared-data/raw` (contains N-BaIoT, CIC_IOT_Dataset2023,
  Edge-IIoTset; CIC IoT-DIAD 2024 NOT present — external replication dataset acquisition is a known blocker,
  see failures.md)
- Audit matrix skeleton created at `docs/FABRID_IDS_Audit_Implementation_Matrix.md`

last verified major implementation chunk: none yet (no batched verification cycle run; only alpha_grid
generation script executed and count checked = 207)

next implementation chunk:
- Phase 2: N-BaIoT client inventory + canonical device/attack IDs + source-row provenance +
  deterministic source-order benign/attack partitioner (`src/fabrid/data/partitioner.py`,
  `src/fabrid/data/provenance.py`) with exclusivity tests (T01).
- Reuse assessment: `/home/naslouby/Projects/datp-core` has a mature `datp_core.data.nbaiot` reader/
  materializer and `datp_core.detector` training/scoring/checkpoint stack, plus
  `datp_core.thresholds.calibration`. Per roadmap section 18 ("Where FABRID is implemented on the
  existing DATP experimental stack, inherit its frozen preprocessing/FedAvg/..."), FABRID-IDS should
  depend on datp-core for detector training + score generation rather than reimplementing federated
  training. FABRID-IDS's own `src/fabrid/` package owns only the allocation/calibration/optimization/
  statistics decision layer that is the actual roadmap contribution. This must be confirmed by reading
  datp-core's nbaiot reader/schema and detector/scoring contracts before Phase 2 code is written.

known blockers:
- CIC IoT-DIAD 2024 raw data not present under `datp-shared-data/raw`. External replication (Phase 19-20)
  cannot proceed until acquired. Not blocking primary N-BaIoT work (Phases 2-18, 21-25 minus event/external
  branches). Will mark EXTERNAL-* / GATE-G15 rows BLOCKED_EXTERNAL if acquisition remains impossible.
- Gotham 2025 / CICIoMT2024 (event-level dataset candidates, Phase 21) also not present under raw data path.
  Event-level claims (EVENT-*, GATE-G16) likely BLOCKED_EXTERNAL pending data acquisition; record decision
  once Phase 21 is reached.

known stale/incomplete areas: whole `src/fabrid/` decision-layer implementation is new; nothing yet verified
against actual N-BaIoT data.

important pending test/audit runs:
- alpha_grid uniqueness/count check: DONE (passed, 207).
- T01-T18 mandatory scientific software tests: NOT YET WRITTEN.
- ruff/pyright/pytest batched cycle: NOT YET RUN (no substantial code beyond config/alpha_grid yet).
