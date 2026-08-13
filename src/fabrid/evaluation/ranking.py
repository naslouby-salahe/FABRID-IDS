from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata

from fabrid.domain.enums import AllocationPolicy
from fabrid.domain.scores import ScoreVector
from fabrid.domain.values import Probability


AUROC_IDENTITY_TOLERANCE = Probability(1.0e-12)


@dataclass(frozen=True, slots=True)
class FrozenScoreRanking:
    benign: ScoreVector
    attack: ScoreVector

    def __post_init__(self) -> None:
        if self.benign.row_count.value == 0:
            raise ValueError("ranking metrics require at least one benign score")
        if self.attack.row_count.value == 0:
            raise ValueError("ranking metrics require at least one attack score")


@dataclass(frozen=True, slots=True)
class PolicyRankingEvidence:
    policy: AllocationPolicy
    auroc: Probability
    auprc: Probability


def compute_auroc(ranking: FrozenScoreRanking) -> Probability:
    scores = np.concatenate((ranking.benign.values, ranking.attack.values))
    attack_mask = np.concatenate(
        (
            np.zeros(ranking.benign.row_count.value, dtype=np.bool_),
            np.ones(ranking.attack.row_count.value, dtype=np.bool_),
        )
    )
    ranks = rankdata(scores)
    positive_count = ranking.attack.row_count.value
    negative_count = ranking.benign.row_count.value
    positive_rank_sum = float(np.sum(ranks[attack_mask]))
    area = (
        positive_rank_sum - positive_count * (positive_count + 1) / 2
    ) / (positive_count * negative_count)
    return Probability(float(area))


def compute_auprc(ranking: FrozenScoreRanking) -> Probability:
    scores = np.concatenate((ranking.benign.values, ranking.attack.values))
    attack_mask = np.concatenate(
        (
            np.zeros(ranking.benign.row_count.value, dtype=np.bool_),
            np.ones(ranking.attack.row_count.value, dtype=np.bool_),
        )
    )
    order = np.argsort(-scores, kind="stable")
    sorted_attacks = attack_mask[order]
    cumulative_true_positive = np.cumsum(sorted_attacks)
    cumulative_false_positive = np.cumsum(~sorted_attacks)
    recall = cumulative_true_positive / ranking.attack.row_count.value
    precision = cumulative_true_positive / (
        cumulative_true_positive + cumulative_false_positive
    )
    recall_with_origin = np.concatenate((np.asarray((0.0,)), recall))
    precision_with_origin = np.concatenate((np.asarray((1.0,)), precision))
    return Probability(float(np.trapezoid(precision_with_origin, recall_with_origin)))


def assert_auroc_invariant(
    evidence: tuple[PolicyRankingEvidence, ...],
    tolerance: Probability = AUROC_IDENTITY_TOLERANCE,
) -> None:
    if not evidence:
        raise ValueError("AUROC invariance requires at least one policy")
    policies = tuple(item.policy for item in evidence)
    if len(set(policies)) != len(policies):
        raise ValueError("AUROC invariance evidence contains duplicate policies")
    areas = tuple(item.auroc.value for item in evidence)
    if max(areas) - min(areas) >= tolerance.value:
        raise ValueError("AUROC invariant violated across frozen-score policies")
