from __future__ import annotations

import numpy as np
import pytest

from fabrid.domain.enums import AllocationPolicy
from fabrid.domain.scores import ScoreVector
from fabrid.domain.values import Probability
from fabrid.evaluation.ranking import (
    FrozenScoreRanking,
    PolicyRankingEvidence,
    assert_auroc_invariant,
    compute_auprc,
    compute_auroc,
)


def _ranking(benign: tuple[float, ...], attack: tuple[float, ...]) -> FrozenScoreRanking:
    return FrozenScoreRanking(
        benign=ScoreVector(np.asarray(benign, dtype=np.float64)),
        attack=ScoreVector(np.asarray(attack, dtype=np.float64)),
    )


def test_perfect_frozen_score_separation_has_unit_auroc_and_auprc() -> None:
    ranking = _ranking((0.1, 0.2), (0.8, 0.9))

    assert compute_auroc(ranking) == Probability(1.0)
    assert compute_auprc(ranking) == Probability(1.0)


def test_auroc_can_represent_chance_ranking() -> None:
    ranking = _ranking((2.0, 3.0), (1.0, 4.0))

    assert compute_auroc(ranking) == Probability(0.5)


def test_ranking_metrics_require_both_populations() -> None:
    with pytest.raises(ValueError):
        _ranking((), (1.0,))
    with pytest.raises(ValueError):
        _ranking((1.0,), ())


def test_auroc_invariant_accepts_equal_frozen_score_evidence() -> None:
    evidence = (
        PolicyRankingEvidence(AllocationPolicy.EQ_FPR, Probability(0.8), Probability(0.7)),
        PolicyRankingEvidence(AllocationPolicy.GREEDY, Probability(0.8), Probability(0.7)),
        PolicyRankingEvidence(AllocationPolicy.FABRID_MACRO, Probability(0.8), Probability(0.7)),
    )

    assert_auroc_invariant(evidence)


def test_auroc_invariant_rejects_policy_dependent_ranking() -> None:
    with pytest.raises(ValueError):
        assert_auroc_invariant(
            (
                PolicyRankingEvidence(AllocationPolicy.EQ_FPR, Probability(0.8), Probability(0.7)),
                PolicyRankingEvidence(AllocationPolicy.GREEDY, Probability(0.9), Probability(0.7)),
            )
        )
