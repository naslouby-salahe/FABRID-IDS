# FABRID-IDS

Federated Alert-Budget Reallocation for Intrusion Detection Systems on
heterogeneous IoT device fleets.

FABRID-IDS trains a shared autoencoder over distributed device data, allocates
a federated false-positive budget across devices according to measured
detection utility, and evaluates the allocation with a fully reproducible,
publication-ready pipeline: matched-budget experiments, attack-subtype and
botnet-family generalization, sensitivity analyses, external replication on a
second dataset, and a provenance-gated event-level branch.

## Installation

```bash
uv sync
```

Requires Python 3.12+.

## Repository layout

Repository roots, dataset layout, and scientific settings live in the
validated production configuration:

```text
configs/fabrid.yaml
```

```text
data/raw/           raw datasets (nbaiot, cic_iot_diad, gotham, ciciomt)
data/preprocessed/  reusable preprocessing (provenance-gated reuse)
outputs/            runtime artifacts (logs, checkpoints, scores, runs, report.json, tables, figures)
results/            one publication bundle (figures, tables, experiments/)
```

## Main commands

```bash
fabrid validate                        validate the production configuration
fabrid preprocess [--overwrite]        reuse or build reusable preprocessing
fabrid experiment EXPERIMENT_ID [--overwrite]
fabrid finalize                        run replication, event-level, analysis, and reporting
fabrid report                          rebuild report.json, tables, and figures
fabrid results                         rebuild the publication bundle
fabrid status                          expected-vs-present artifacts
```

`fabrid experiment EXPERIMENT_ID` runs one configured experiment's complete
seed set through the shared runner, then writes inference (when
matched-budget evaluations exist), the report, and the publication bundle
under `results/`. Seeds always come from `configs/fabrid.yaml`; there is no
partial-seed execution.

`fabrid report` rebuilds `report.json`, tables, and figures from
existing run artifacts without retraining. `fabrid results` rebuilds the
paper bundle at `results/` (figures, JSON and CSV tables, per-experiment
artifacts under `results/experiments/`, reproducibility metadata,
and manifest).

## Results bundle

A successful run writes a checksummed bundle under `results/`
containing the exact configuration, reproducibility metadata, metrics,
statistics, tables, figures, structured report evidence, and one folder
per experiment.

## Testing

```bash
make check        # lint, typecheck, unit tests (parallel, no smoke/integration)
make smoke        # end-to-end smoke workflow with explicit reduced settings
make architecture # architecture contract tests
```

Tests run in parallel with pytest-xdist; integration tests that require real
raw datasets are marked `integration` and excluded by default.

## Reproducibility

Every run records the protocol configuration, raw-data digests,
checkpoint and score digests, and the exact command line in
`reproducibility.json` inside the results bundle. Preprocessed data,
checkpoints, and frozen scores are reused only when their provenance matches;
any rebuild is logged explicitly.
