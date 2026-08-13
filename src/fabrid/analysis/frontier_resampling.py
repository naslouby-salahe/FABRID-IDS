from __future__ import annotations

import numpy as np

from fabrid.allocation.frontier_inputs import (
    AttackSubtypeScores,
    FrontierScorePopulation,
)
from fabrid.domain.scores import ScoreVector
from fabrid.domain.values import AnalysisSeed


def _bootstrap_scores(scores: ScoreVector, rng: np.random.Generator) -> ScoreVector:
    indexes = rng.integers(
        low=0,
        high=scores.row_count.value,
        size=scores.row_count.value,
    )
    return ScoreVector(np.asarray(scores.values[indexes], dtype=np.float64))


def bootstrap_frontier_populations(
    populations: tuple[FrontierScorePopulation, ...],
    seed: AnalysisSeed,
) -> tuple[FrontierScorePopulation, ...]:
    if not populations:
        raise ValueError("frontier bootstrap requires at least one client population")
    client_ids = tuple(population.client_id for population in populations)
    if len(set(client_ids)) != len(client_ids):
        raise ValueError("frontier bootstrap contains duplicate clients")

    rng = np.random.default_rng(seed.value)
    return tuple(
        FrontierScorePopulation(
            client_id=population.client_id,
            benign_frontier=_bootstrap_scores(population.benign_frontier, rng),
            attack_validation=tuple(
                AttackSubtypeScores(
                    subtype=subtype.subtype,
                    scores=_bootstrap_scores(subtype.scores, rng),
                )
                for subtype in population.attack_validation
            ),
        )
        for population in populations
    )
