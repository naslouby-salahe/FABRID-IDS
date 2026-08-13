# FABRID-IDS

FABRID-IDS implements federated alert-budget reallocation for heterogeneous IoT intrusion detection. The repository is organized around one typed execution spine and immutable research artifacts rather than standalone scripts or duplicated experiment entry points.

## Architecture

```text
src/fabrid/
├── domain/          # enums, identifiers, semantic values, coordinates, populations
├── protocol/        # frozen scientific protocol and experiment specifications
├── datasets/        # dataset-owned readers, splits, manifests, external eligibility
├── detector/        # preprocessing, federated detector training, scoring, persistence
├── calibration/     # independent final-calibration threshold construction
├── allocation/      # frontier, solver, policies, baselines, diagnostics, sensitivities
├── evaluation/      # typed result records, metrics, ranking evidence, event/workload metrics
├── analysis/        # inference, multiplicity, gates, stability, conservative evidence
├── artifacts/       # artifact layout, stores, hashes, manifests, serialization
├── pipeline/        # the only scientific execution/orchestration layer
├── reporting/       # publication tables and figures
├── validation/      # leakage, provenance, determinism, claims, data integrity
└── cli.py           # thin process boundary
```

Dependency direction is `domain → protocol/capabilities → pipeline → CLI/reporting`. Lower-level packages do not import pipeline code. There are no compatibility shims for superseded package layouts.

## Execution

The application has one user-facing entry point:

```bash
fabrid campaign <campaign-id> \
  --raw-data-root /path/to/raw \
  --outputs-root /path/to/outputs
```

Dataset acquisition/copy integrity can be checked independently:

```bash
fabrid validate-data /path/to/raw/dataset
```

The full campaign trains each detector seed once and reuses the resulting immutable score partitions for the matched-budget study and roadmap extensions. Allocation and final calibration never retrain the detector.

## Scientific contracts

- Primary dataset: N-BaIoT with physical devices as clients.
- Detector seeds: `0..9` from the typed protocol.
- Primary weighting: equal-client weighting.
- Alert-budget decisions use explicit typed budgets, policies, weights, thresholds, and solver evidence.
- Final calibration is isolated from frontier construction and from attack-test evaluation.
- Score artifacts are split by dataset × detector seed × client × split.
- Solver-invalid FABRID coordinates are excluded explicitly; tolerances are not relaxed and results are not imputed.
- Primary inference uses paired detector-seed evidence, exact sign-flip testing, bootstrap intervals, Holm correction, practical gates, and budget-compliance evidence.
- Weight sensitivity uses the preregistered `γ ∈ {0, 0.5, 1, 1.5}` dataset-volume proxy analysis and is not described as operational traffic weighting.

The authoritative scientific specification is `docs/FABRID-IDS Roadmap.md`; implementation coverage is tracked in `docs/FABRID_IDS_Audit_Implementation_Matrix.md`.

## Artifact layout

Each campaign is written beneath one artifact root:

```text
<outputs>/<campaign-id>/
├── protocol.json
├── datasets/<dataset>/
│   ├── features.json
│   └── splits.json
├── detectors/<dataset>/seed-###/
├── scores/<dataset>/seed-###/<client>/<split>.parquet
├── experiments/<experiment>/<variant>/<dataset>/seed-###/<budget>/<weight-mode>/
│   ├── allocations/
│   ├── results.parquet
│   ├── evaluation.json
│   ├── analysis/
│   └── publication/
├── analysis/
└── audit/
```

All path composition is owned by `ArtifactLayout`; scientific modules do not invent output paths.

## Development rules

The repository intentionally favors semantic types over primitive cross-module contracts: finite states use enums, structured contracts use frozen dataclasses/value objects, dictionary-shaped application APIs are avoided, and `Any`/`NewType` are not accepted substitutes for domain modeling. Tests run in parallel by default. Ruff and strict Pyright cover both `src` and `tests`.

Raw datasets, authentication/session material, generated campaign outputs, and temporary agent state are not source files and must not be committed.
