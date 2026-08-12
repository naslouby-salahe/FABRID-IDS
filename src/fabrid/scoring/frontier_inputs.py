"""Derive `ClientFrontierInputs` (provisional thresholds + subtype confusion counts) from a
persisted `ScoreArtifact`, bridging the frozen scores to the frontier/allocation layer.
"""

from __future__ import annotations

import numpy as np

from fabrid.calibration.order_statistic import calibrate_threshold
from fabrid.data.partitioner import AttackSplit, BenignSplit
from fabrid.evaluation.record_level import AttackSubtype
from fabrid.frontier.builder import ClientFrontierInputs
from fabrid.frontier.utility import SubtypeConfusionCounts
from fabrid.schemas.score_artifact import ScoreArtifact
from fabrid.scoring.score_contract import compute_auprc, compute_auroc


def _scores_by_split(artifact: ScoreArtifact, split_id: BenignSplit | AttackSplit) -> np.ndarray:
    return np.array([record.score for record in artifact.records if record.split_id == split_id])


def _attack_scores_by_subtype(
    artifact: ScoreArtifact, split_id: AttackSplit
) -> dict[AttackSubtype, np.ndarray]:
    grouped: dict[AttackSubtype, list[float]] = {}
    for record in artifact.records:
        if record.split_id != split_id:
            continue
        if record.attack_type is None:
            raise ValueError(f"attack-split record {record.sample_id} has no attack_type")
        grouped.setdefault(record.attack_type, []).append(record.score)
    return {subtype: np.array(scores) for subtype, scores in grouped.items()}


def build_client_frontier_inputs(
    artifact: ScoreArtifact, alpha_grid: tuple[float, ...]
) -> ClientFrontierInputs:
    frontier_scores = _scores_by_split(artifact, BenignSplit.FRONTIER)
    subtype_validation_scores = _attack_scores_by_subtype(artifact, AttackSplit.VALIDATION)
    subtype_validation_row_counts = {
        subtype: scores.shape[0] for subtype, scores in subtype_validation_scores.items()
    }

    subtype_confusion_by_candidate: list[dict[AttackSubtype, SubtypeConfusionCounts]] = []
    for alpha in alpha_grid:
        threshold = calibrate_threshold(frontier_scores, alpha)
        counts: dict[AttackSubtype, SubtypeConfusionCounts] = {}
        for subtype, scores in subtype_validation_scores.items():
            true_positive = int(np.sum(threshold.alerts(scores)))
            counts[subtype] = SubtypeConfusionCounts(
                true_positive=true_positive, false_negative=scores.shape[0] - true_positive
            )
        subtype_confusion_by_candidate.append(counts)

    return ClientFrontierInputs(
        benign_frontier_scores=frontier_scores,
        subtype_validation_row_counts=subtype_validation_row_counts,
        subtype_confusion_by_candidate=subtype_confusion_by_candidate,
    )


def benign_final_cal_scores(artifact: ScoreArtifact) -> np.ndarray:
    return _scores_by_split(artifact, BenignSplit.FINAL_CAL)


def benign_test_scores(artifact: ScoreArtifact) -> np.ndarray:
    return _scores_by_split(artifact, BenignSplit.TEST)


def attack_test_scores_by_subtype(artifact: ScoreArtifact) -> dict[AttackSubtype, np.ndarray]:
    return _attack_scores_by_subtype(artifact, AttackSplit.TEST)


def _all_test_scores_and_labels(artifact: ScoreArtifact) -> tuple[np.ndarray, np.ndarray]:
    benign = benign_test_scores(artifact)
    attack_by_subtype = attack_test_scores_by_subtype(artifact)
    attack = np.concatenate(list(attack_by_subtype.values())) if attack_by_subtype else np.array([])
    scores = np.concatenate([benign, attack])
    is_attack = np.concatenate(
        [np.zeros(benign.shape[0], dtype=bool), np.ones(attack.shape[0], dtype=bool)]
    )
    return scores, is_attack


def all_test_auroc(artifact: ScoreArtifact) -> float:
    """AUROC over BENIGN_TEST vs all ATTACK_TEST rows, for the AUROC-invariance audit."""
    scores, is_attack = _all_test_scores_and_labels(artifact)
    return compute_auroc(scores, is_attack)


def all_test_auprc(artifact: ScoreArtifact) -> float:
    """AUPRC over BENIGN_TEST vs all ATTACK_TEST rows; describes the frozen detector's score
    ranking, same as AUROC — never recomputed per policy.
    """
    scores, is_attack = _all_test_scores_and_labels(artifact)
    return compute_auprc(scores, is_attack)
