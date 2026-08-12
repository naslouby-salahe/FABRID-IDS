"""Measure SOLVER_INVALID rate across all 5 primary budgets x 10 seeds (F001 investigation).

Does not compute full statistics — just characterizes how often FABRID_MACRO/
FABRID_MINIMAX solve within tolerance at each budget, to inform the F001
decision (accept reduced n, or investigate solver tuning).
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

from fabrid.config.detector import load_detector_seeds
from fabrid.config.protocol import load_protocol
from fabrid.evaluation.record_level import ClientId
from fabrid.experiments.main_experiment import run_seed_at_budget
from fabrid.schemas.allocation import AllocationPolicy

_RESULTS_DIR = Path(__file__).parents[1] / "results" / "scores"
_ALPHA_GRID_PATH = Path(__file__).parents[1] / "src" / "fabrid" / "config" / "alpha_grid.json"
_BUDGETS = (0.001, 0.0025, 0.005, 0.010, 0.020)


def _load_seed_artifacts(seed: int) -> dict[ClientId, object]:
    seed_dir = _RESULTS_DIR / f"seed_{seed}"
    manifest = json.loads((seed_dir / "manifest.json").read_text())
    artifacts = {}
    for client_name in manifest:
        with (seed_dir / f"{client_name}.pkl").open("rb") as handle:
            artifacts[ClientId(client_name)] = pickle.load(handle)  # noqa: S301
    return artifacts


def main() -> None:
    alpha_grid = tuple(json.loads(_ALPHA_GRID_PATH.read_text())["alpha_grid"])
    protocol = load_protocol()
    seeds = load_detector_seeds()

    # cache artifacts per seed across budgets to avoid re-reading 10x900MB from disk 5 times
    artifacts_by_seed = {seed: _load_seed_artifacts(seed) for seed in seeds}
    print(f"loaded artifacts for {len(artifacts_by_seed)} seeds")

    invalid_counts = {
        budget: {AllocationPolicy.FABRID_MACRO: 0, AllocationPolicy.FABRID_MINIMAX: 0}
        for budget in _BUDGETS
    }

    for budget in _BUDGETS:
        for seed in seeds:
            result = run_seed_at_budget(
                artifacts_by_seed[seed],
                alpha_grid,
                protocol.utility_eligibility,
                budget,
                protocol.alpha_max,
                protocol.solver_settings,
                seed,
            )
            for policy in (AllocationPolicy.FABRID_MACRO, AllocationPolicy.FABRID_MINIMAX):
                if policy in result.excluded_policies:
                    invalid_counts[budget][policy] += 1
        macro_invalid = invalid_counts[budget][AllocationPolicy.FABRID_MACRO]
        minimax_invalid = invalid_counts[budget][AllocationPolicy.FABRID_MINIMAX]
        print(
            f"budget={budget}: FABRID_MACRO invalid {macro_invalid}/10, "
            f"FABRID_MINIMAX invalid {minimax_invalid}/10"
        )


if __name__ == "__main__":
    main()
