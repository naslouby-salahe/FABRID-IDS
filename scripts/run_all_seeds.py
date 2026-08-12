"""Run full-scale training + score generation for every seed not yet persisted.

Skips seeds whose `results/scores/seed_<n>/manifest.json` already exists, so
this script is safe to resume after a partial run.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from run_seed_training import RESULTS_DIR, run_seed  # noqa: E402

from fabrid.config.detector import load_detector_seeds  # noqa: E402

if __name__ == "__main__":
    seeds = load_detector_seeds()
    for seed in seeds:
        manifest_path = RESULTS_DIR / f"seed_{seed}" / "manifest.json"
        if manifest_path.exists():
            print(f"seed {seed} already complete, skipping")
            continue
        t0 = time.time()
        print(f"=== starting seed {seed} ===")
        run_seed(seed)
        print(f"=== seed {seed} done in {time.time() - t0:.1f}s ===")
