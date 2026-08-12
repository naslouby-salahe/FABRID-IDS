"""D005 real-data comparison: raw-utility vs conservative/LCB-utility FABRID_MINIMAX.

Roadmap ablation #9 (conservative utility sensitivity): does resolving
FABRID_MINIMAX from the one-sided 95% LCB utility curve instead of raw
validation utility change the WorstClientRecall outcome, for the F002
finding (FABRID_MINIMAX sometimes yields worse WorstClientRecall than
EQ_FPR due to validation/test worst-client divergence)?
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

from fabrid.config.detector import load_detector_seeds
from fabrid.config.protocol import load_protocol
from fabrid.evaluation.record_level import ClientId
from fabrid.experiments.main_experiment import (
    run_conservative_minimax_at_budget,
    run_seed_at_budget,
)
from fabrid.schemas.allocation import AllocationPolicy

_RESULTS_DIR = Path(__file__).parents[1] / "results" / "scores"
_ALPHA_GRID_PATH = Path(__file__).parents[1] / "src" / "fabrid" / "config" / "alpha_grid.json"
_BUDGET = 0.01


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

    print(f"{'seed':>4} {'EQ_FPR worst':>14} {'raw MINIMAX worst':>18} {'LCB MINIMAX worst':>18}")
    for seed in seeds:
        artifacts = _load_seed_artifacts(seed)
        raw_result = run_seed_at_budget(
            artifacts,
            alpha_grid,
            protocol.utility_eligibility,
            _BUDGET,
            protocol.alpha_max,
            protocol.solver_settings,
            seed,
        )
        eq_fpr_worst = raw_result.worst_client_recall_by_policy.get(AllocationPolicy.EQ_FPR)
        raw_minimax_worst = raw_result.worst_client_recall_by_policy.get(
            AllocationPolicy.FABRID_MINIMAX
        )
        exclusion_reason = raw_result.excluded_policies.get(AllocationPolicy.FABRID_MINIMAX, "?")
        raw_minimax_status = (
            f"{raw_minimax_worst:.4f}"
            if raw_minimax_worst is not None
            else f"EXCLUDED({exclusion_reason[:40]})"
        )

        conservative_result = run_conservative_minimax_at_budget(
            artifacts, alpha_grid, protocol.utility_eligibility, _BUDGET, protocol.solver_settings
        )
        lcb_status = (
            f"{conservative_result[1]:.4f}" if conservative_result is not None else "EXCLUDED"
        )

        print(f"{seed:>4} {eq_fpr_worst:>14.4f} {raw_minimax_status:>18} {lcb_status:>18}")


if __name__ == "__main__":
    main()
