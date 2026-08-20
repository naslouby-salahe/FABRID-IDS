# FABRID-IDS operator catalog.
#
# Quality targets run local tools. Execution targets invoke `fabrid`.
# Scientific values, roots, seeds, budgets, and enablement live in
# configs/fabrid.yaml. Pass OVERWRITE=1 to rebuild prerequisites.

PYTHON ?= .venv/bin/python
FABRID ?= .venv/bin/fabrid

ifdef OVERWRITE
OVERWRITE_FLAG := --overwrite
endif

# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------

install:
	uv sync --dev

format:
	.venv/bin/ruff format .

lint:
	.venv/bin/ruff check .

typecheck:
	.venv/bin/pyright

test:
	.venv/bin/pytest

test-unit:
	.venv/bin/pytest -m "not integration"

test-integration:
	.venv/bin/pytest -m integration

architecture:
	.venv/bin/pytest tests/architecture

smoke:
	.venv/bin/pytest -m smoke -o addopts=''

check: lint typecheck test

# ---------------------------------------------------------------------------
# Always-available commands
# ---------------------------------------------------------------------------

validate:
	$(FABRID) validate

status:
	$(FABRID) status

report:
	$(FABRID) report

results:
	$(FABRID) results

# Optional N-BaIoT feature/split build. experiment already runs
# this as a prerequisite (reuse when digests match).
preprocess:
	$(FABRID) preprocess $(OVERWRITE_FLAG)

# ---------------------------------------------------------------------------
# Experiments (each runs its complete configured seed set)
# Each still preprocesses N-BaIoT and trains/reuses the ten detector seeds.
# ---------------------------------------------------------------------------

# Primary matched-budget comparison on N-BaIoT (EQ_FPR, GREEDY,
# FABRID_MACRO, FABRID_MINIMAX, POOLED_SHARED, TEST_ORACLE).
experiment-matched-budget:
	$(FABRID) experiment matched_budget $(OVERWRITE_FLAG)

# Attack-subtype-disjoint generalization: 3 fixed fold rotations.
experiment-attack-subtype-disjoint:
	$(FABRID) experiment attack_subtype_disjoint $(OVERWRITE_FLAG)

# Botnet-family transfer on the 7 dual-botnet clients (BASHLITE <-> Mirai).
experiment-botnet-family-disjoint:
	$(FABRID) experiment botnet_family_disjoint $(OVERWRITE_FLAG)

# Dataset-count weight gamma sweep {0, 0.5, 1, 1.5}.
experiment-weight-sensitivity:
	$(FABRID) experiment weight_sensitivity $(OVERWRITE_FLAG)

# Resample frontier+validation and re-solve FABRID_MACRO (stability).
experiment-allocation-stability:
	$(FABRID) experiment allocation_stability $(OVERWRITE_FLAG)

# Re-solve FABRID on 95% LCB utility curves.
experiment-conservative-utility:
	$(FABRID) experiment conservative_utility $(OVERWRITE_FLAG)

# Utility-curve dispersion H_U (mechanism diagnostic).
experiment-utility-heterogeneity:
	$(FABRID) experiment utility_heterogeneity $(OVERWRITE_FLAG)

# CIC IoT-DIAD external replication with primary N-BaIoT checkpoints.
experiment-external-replication:
	$(FABRID) experiment external_replication $(OVERWRITE_FLAG)

# Gotham/CICIoMT event-level branch (gated; no claim if provenance/scores fail).
experiment-event-level:
	$(FABRID) experiment event_level $(OVERWRITE_FLAG)

# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

help:
	@printf '%s\n' \
		'Quality' \
		'  make install | format | lint | typecheck | test | test-unit | test-integration | architecture | smoke | check' \
		'' \
		'Always available' \
		'  make validate              # production YAML' \
		'  make status                # expected vs present artifacts' \
		'  make preprocess            # optional N-BaIoT feature/split build' \
		'  make report                # rebuild report from existing artifacts' \
		'  make results               # rebuild results/ paper bundle' \
		'' \
		'Experiments (add OVERWRITE=1 to rebuild prerequisites)' \
		'  make experiment-matched-budget' \
		'  make experiment-attack-subtype-disjoint' \
		'  make experiment-botnet-family-disjoint' \
		'  make experiment-weight-sensitivity' \
		'  make experiment-allocation-stability' \
		'  make experiment-conservative-utility' \
		'  make experiment-utility-heterogeneity' \
		'  make experiment-external-replication' \
		'  make experiment-event-level'

.DEFAULT_GOAL := help

.PHONY: install format lint typecheck test test-unit test-integration architecture smoke check \
	validate status preprocess report results help \
	experiment-matched-budget experiment-attack-subtype-disjoint \
	experiment-botnet-family-disjoint experiment-weight-sensitivity \
	experiment-allocation-stability experiment-conservative-utility \
	experiment-utility-heterogeneity experiment-external-replication \
	experiment-event-level
