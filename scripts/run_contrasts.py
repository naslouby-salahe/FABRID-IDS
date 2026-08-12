"""Run the primary contrasts (FABRID_MACRO/MINIMAX vs EQ_FPR) across all persisted seeds at one budget.

Not the full roadmap confirmatory protocol (single budget, no Holm correction
across the 5-budget family, no attack-subtype-disjoint check) — a real
statistical sanity pass across all 10 trained seeds.
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
from fabrid.statistics.contrasts import macro_recall_contrast, worst_client_recall_contrast

_RESULTS_DIR = Path(__file__).parents[1] / "results" / "scores"
_ALPHA_GRID_PATH = Path(__file__).parents[1] / "src" / "fabrid" / "config" / "alpha_grid.json"
_BUDGET = 0.01
_BOOTSTRAP_RESAMPLES = 50_000


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

    results = []
    for seed in seeds:
        artifacts = _load_seed_artifacts(seed)
        result = run_seed_at_budget(
            artifacts,
            alpha_grid,
            protocol.utility_eligibility,
            _BUDGET,
            protocol.alpha_max,
            protocol.solver_settings,
            seed,
        )
        excluded = list(result.excluded_policies.keys())
        print(f"seed {seed}: {sorted(p.value for p in result.macro_recall_by_policy)} "
              f"excluded={[p.value for p in excluded]}")
        results.append(result)

    results_tuple = tuple(results)

    contrast_a = macro_recall_contrast(
        results_tuple,
        AllocationPolicy.FABRID_MACRO,
        AllocationPolicy.EQ_FPR,
        _BOOTSTRAP_RESAMPLES,
        bootstrap_seed=0,
    )
    print(f"\nContrast A (FABRID_MACRO - EQ_FPR, MacroRecall), n={len(contrast_a.included_seeds)}, "
          f"excluded_seeds={contrast_a.excluded_seeds}")
    print(f"  mean_diff={contrast_a.bootstrap.mean_difference:.4f}, "
          f"95% CI=[{contrast_a.bootstrap.confidence_interval_low:.4f}, "
          f"{contrast_a.bootstrap.confidence_interval_high:.4f}]")
    print(f"  sign-flip p={contrast_a.sign_flip.p_value:.4f}")

    contrast_b = worst_client_recall_contrast(
        results_tuple,
        AllocationPolicy.FABRID_MINIMAX,
        AllocationPolicy.EQ_FPR,
        _BOOTSTRAP_RESAMPLES,
        bootstrap_seed=0,
    )
    print(f"\nContrast B (FABRID_MINIMAX - EQ_FPR, WorstClientRecall), "
          f"n={len(contrast_b.included_seeds)}, excluded_seeds={contrast_b.excluded_seeds}")
    print(f"  mean_diff={contrast_b.bootstrap.mean_difference:.4f}, "
          f"95% CI=[{contrast_b.bootstrap.confidence_interval_low:.4f}, "
          f"{contrast_b.bootstrap.confidence_interval_high:.4f}]")
    print(f"  sign-flip p={contrast_b.sign_flip.p_value:.4f}")


if __name__ == "__main__":
    main()
