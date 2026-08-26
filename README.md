# FABRID-IDS

**Federated Alert-Budget Reallocation for Heterogeneous IoT Intrusion Detection**

FABRID-IDS studies the **decision layer after federated model training**. Given a frozen anomaly scorer, it allocates client-specific nominal false-positive-rate (FPR) targets under a shared federation budget, independently calibrates the corresponding client thresholds, and evaluates the operating points realized on held-out data.

The project separates **ranking quality, nominal budget feasibility, threshold calibration, and realized detection behavior** rather than treating them as equivalent properties of an intrusion detector.

## Methods

FABRID-IDS evaluates:

* **Equal-FPR** — equal nominal FPR allocation across clients.
* **Macro** — constrained allocation maximizing mean client utility.
* **Tail (CVaR)** — constrained allocation optimizing lower-tail client utility.
* **Conservative calibration** — finite-sample sensitivity analysis using a more conservative calibration rank.

The detector is frozen before policy comparison. Frontier estimation, allocation, final benign-only calibration, and held-out testing use separate data roles.

## Evaluation

Primary evaluation:

* **N-BaIoT**
* 9 device-defined clients
* 10 paired detector seeds
* 5 federation FPR budgets
* frozen federated anomaly scorer

Secondary protocol stress test:

* **CIC IoT-DIAD**
* 19 clients
* reconstruction-error scoring

Additional experiments cover attack-subtype and botnet-family generalization, allocation stability, utility heterogeneity, weighting sensitivity, conservative utility, and external replication.

The central empirical finding is that satisfying a **nominal federation-level FPR constraint does not guarantee the FPR realized after independent finite-sample calibration**, even when score-ranking metrics are near ceiling.

## Installation

```bash
git clone https://github.com/naslouby-salahe/FABRID-IDS.git
cd FABRID-IDS
uv sync --dev
```

## Quick Start

```bash
make validate
make smoke
make experiment-matched-budget
make results
```

Run:

```bash
make help
```

to display the complete experiment and quality-command catalog.

Key experiment targets include:

```bash
make experiment-matched-budget
make experiment-attack-subtype-disjoint
make experiment-botnet-family-disjoint
make experiment-weight-sensitivity
make experiment-allocation-stability
make experiment-conservative-utility
make experiment-utility-heterogeneity
make experiment-external-replication
```

Use `OVERWRITE=1` when intentionally rebuilding existing experiment evidence:

```bash
make experiment-matched-budget OVERWRITE=1
```

## Quality Checks

```bash
make format
make lint
make typecheck
make test
make architecture
make check
```

## Reproducibility

Scientific configuration, seeds, budgets, dataset locations, and experiment enablement are centralized in:

```text
configs/fabrid.yaml
```

Generated experimental artifacts and manuscript-facing results are kept separate so that reported evidence can be rebuilt and verified from the configured pipeline.

## Citation

If you use FABRID-IDS in academic work, please cite the archived software release. Machine-readable citation metadata are provided in [`CITATION.cff`](CITATION.cff).

## License

FABRID-IDS is released under the [MIT License](LICENSE).
