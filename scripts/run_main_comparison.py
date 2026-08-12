"""Real-data sanity comparison: EQ_FPR vs FABRID_MACRO/MINIMAX on persisted seed score artifacts.

Not the roadmap's confirmatory experiment (no statistics, single budget, no
attack-subtype-disjoint protocol) — a scientific sanity check that the full
pipeline (persisted real scores -> frontier -> allocation -> final
calibration -> metrics) produces plausible, internally consistent results.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

from fabrid.allocation.equal_fpr import allocate_equal_fpr
from fabrid.allocation.fabrid_macro import allocate_fabrid_macro
from fabrid.allocation.fabrid_minimax import allocate_fabrid_minimax
from fabrid.calibration.final_calibration import calibrate_final_thresholds
from fabrid.config.protocol import load_protocol
from fabrid.evaluation.record_level import ClientId
from fabrid.frontier.builder import build_federation_frontier
from fabrid.optimization.milp import SolverInvalidError
from fabrid.schemas.allocation import Allocation
from fabrid.scoring.frontier_inputs import (
    all_test_auroc,
    attack_test_scores_by_subtype,
    benign_final_cal_scores,
    build_client_frontier_inputs,
)

_RESULTS_DIR = Path(__file__).parents[1] / "results" / "scores"
_ALPHA_GRID_PATH = Path(__file__).parents[1] / "src" / "fabrid" / "config" / "alpha_grid.json"
_BUDGET = 0.01


def _macro_recall(allocation: Allocation, artifacts: dict) -> float:
    thresholds = calibrate_final_thresholds(
        allocation, {c: benign_final_cal_scores(a) for c, a in artifacts.items()}
    )
    recalls = []
    for client_id, result in thresholds.items():
        subtype_scores = attack_test_scores_by_subtype(artifacts[client_id])
        if not subtype_scores:
            continue
        subtype_tprs = [
            float((scores > result.threshold.value).mean()) for scores in subtype_scores.values()
        ]
        recalls.append(sum(subtype_tprs) / len(subtype_tprs))
    return sum(recalls) / len(recalls)


def main(seed: int) -> None:
    alpha_grid = tuple(json.loads(_ALPHA_GRID_PATH.read_text())["alpha_grid"])
    protocol = load_protocol()

    seed_dir = _RESULTS_DIR / f"seed_{seed}"
    manifest = json.loads((seed_dir / "manifest.json").read_text())
    artifacts = {}
    for client_name in manifest:
        with (seed_dir / f"{client_name}.pkl").open("rb") as handle:
            artifacts[ClientId(client_name)] = pickle.load(handle)  # noqa: S301

    print(f"loaded {len(artifacts)} client artifacts for seed {seed}")

    aurocs = {name: all_test_auroc(artifact) for name, artifact in artifacts.items()}
    print(f"AUROC range: {min(aurocs.values()):.3f} - {max(aurocs.values()):.3f}")

    client_inputs = {
        client_id: build_client_frontier_inputs(artifact, alpha_grid)
        for client_id, artifact in artifacts.items()
    }
    guardrails = protocol.utility_eligibility
    federation = build_federation_frontier(client_inputs, alpha_grid, guardrails)
    print(f"fallback_rate={federation.fallback_rate:.3f}")

    eligible_ids = federation.eligible_client_ids()
    utility_curves = federation.utility_curves()
    weight = dict.fromkeys(artifacts, 1.0 / len(artifacts))

    eq_fpr = allocate_equal_fpr(list(artifacts.keys()), _BUDGET, protocol.alpha_max)

    eligible_weight = {c: weight[c] for c in eligible_ids}
    policy_allocators = {
        "FABRID_MACRO": lambda: allocate_fabrid_macro(
            utility_curves, eligible_weight, _BUDGET, protocol.solver_settings
        ),
        "FABRID_MINIMAX": lambda: allocate_fabrid_minimax(
            utility_curves, eligible_weight, _BUDGET, protocol.solver_settings
        ),
    }

    allocations: dict[str, Allocation] = {"EQ_FPR": eq_fpr}
    for policy_name, allocate in policy_allocators.items():
        if not eligible_ids:
            continue
        try:
            allocations[policy_name] = allocate()
        except SolverInvalidError as error:
            print(f"{policy_name}: SOLVER_INVALID, excluding this coordinate: {error}")

    for policy_name, allocation in allocations.items():
        macro_recall = _macro_recall(allocation, {c: artifacts[c] for c in allocation.decisions})
        print(f"{policy_name}: MacroRecall={macro_recall:.4f}")

    # AUROC is computed once per client directly from the shared ScoreArtifact and never
    # recomputed per policy, so T08 (AUROC invariance across policies) holds by construction
    # here rather than needing a runtime check against this script's per-client AUROC dict.
    print("per-client AUROC computed once from shared scores; policies never recompute scores")


if __name__ == "__main__":
    seed_value = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    main(seed_value)
