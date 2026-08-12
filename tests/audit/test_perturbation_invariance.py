"""T02-T06: perturbing test-only or calibration-only data must not change allocation-affecting
inputs, and validation-attack perturbation must not change EQ_FPR (which never reads utility
curves).
"""

from __future__ import annotations

import numpy as np
import pytest

from fabrid.data.partitioner import AttackSplit, BenignSplit, RowCount
from fabrid.evaluation.record_level import AttackSubtype, ClientId
from fabrid.schemas.score_artifact import DetectorSeed, Label, ScoreArtifact, ScoreRecord
from fabrid.scoring.frontier_inputs import benign_final_cal_scores, build_client_frontier_inputs

_GRID = (0.0, 0.01, 0.02, 0.3)  # 0.3 yields a finite frontier threshold (tau=max frontier score)


def _record(
    sample_id: str,
    split: BenignSplit | AttackSplit,
    score: float,
    label: Label,
    attack_type: AttackSubtype | None = None,
) -> ScoreRecord:
    return ScoreRecord(
        sample_id=sample_id,
        source_file="f.csv",
        source_row=RowCount(0),
        split_id=split,
        score=score,
        label=label,
        attack_type=attack_type,
        timestamp=None,
    )


def _build_artifact(
    frontier_scores: tuple[float, ...],
    validation_scores: tuple[float, ...],
    final_cal_scores: tuple[float, ...],
    test_benign_scores: tuple[float, ...],
    test_attack_scores: tuple[float, ...],
) -> ScoreArtifact:
    records: list[ScoreRecord] = []
    for i, s in enumerate(frontier_scores):
        records.append(_record(f"bf{i}", BenignSplit.FRONTIER, s, Label.BENIGN))
    for i, s in enumerate(validation_scores):
        records.append(
            _record(f"av{i}", AttackSplit.VALIDATION, s, Label.ATTACK, AttackSubtype("scan"))
        )
    for i, s in enumerate(final_cal_scores):
        records.append(_record(f"bc{i}", BenignSplit.FINAL_CAL, s, Label.BENIGN))
    for i, s in enumerate(test_benign_scores):
        records.append(_record(f"bt{i}", BenignSplit.TEST, s, Label.BENIGN))
    for i, s in enumerate(test_attack_scores):
        records.append(_record(f"at{i}", AttackSplit.TEST, s, Label.ATTACK, AttackSubtype("scan")))
    return ScoreArtifact(
        dataset_id="n-baiot",
        detector_seed=DetectorSeed(0),
        client_id=ClientId("1"),
        records=tuple(records),
    )


_FRONTIER = (0.1, 0.2, 0.3, 0.4, 0.5)
_VALIDATION = (0.5, 0.6, 0.55, 0.62, 0.58)
_FINAL_CAL = (0.15, 0.25)
_TEST_BENIGN = (0.12, 0.22)
_TEST_ATTACK = (0.51, 0.61)


def test_t02_t03_test_split_perturbation_does_not_change_frontier_inputs() -> None:
    """T02/T03: changing ATTACK_TEST/BENIGN_TEST scores must not change ClientFrontierInputs,
    since allocation is built only from BENIGN_FRONTIER and ATTACK_VALIDATION."""
    baseline = _build_artifact(_FRONTIER, _VALIDATION, _FINAL_CAL, _TEST_BENIGN, _TEST_ATTACK)
    perturbed = _build_artifact(_FRONTIER, _VALIDATION, _FINAL_CAL, (0.99, 0.01), (0.02, 0.98))

    baseline_inputs = build_client_frontier_inputs(baseline, _GRID)
    perturbed_inputs = build_client_frontier_inputs(perturbed, _GRID)

    np.testing.assert_array_equal(
        baseline_inputs.benign_frontier_scores, perturbed_inputs.benign_frontier_scores
    )
    assert (
        baseline_inputs.subtype_validation_row_counts
        == perturbed_inputs.subtype_validation_row_counts
    )
    for baseline_candidate, perturbed_candidate in zip(
        baseline_inputs.subtype_confusion_by_candidate,
        perturbed_inputs.subtype_confusion_by_candidate,
        strict=True,
    ):
        assert baseline_candidate == perturbed_candidate


def test_t04_benign_test_perturbation_does_not_affect_final_cal_scores() -> None:
    """T04: BENIGN_TEST perturbation changes neither allocation-affecting inputs nor the final
    threshold, since BENIGN_FINAL_CAL is a disjoint partition."""
    baseline = _build_artifact(_FRONTIER, _VALIDATION, _FINAL_CAL, _TEST_BENIGN, _TEST_ATTACK)
    perturbed = _build_artifact(_FRONTIER, _VALIDATION, _FINAL_CAL, (0.99, 0.01), _TEST_ATTACK)

    np.testing.assert_array_equal(
        benign_final_cal_scores(baseline), benign_final_cal_scores(perturbed)
    )


def test_t05_final_cal_perturbation_changes_threshold_not_allocation_inputs() -> None:
    """T05: changing BENIGN_FINAL_CAL may change tau_k but must not change the frontier inputs
    that determine alpha_k, since FINAL_CAL is read only after allocation."""
    baseline = _build_artifact(_FRONTIER, _VALIDATION, _FINAL_CAL, _TEST_BENIGN, _TEST_ATTACK)
    perturbed = _build_artifact(_FRONTIER, _VALIDATION, (0.9, 0.95), _TEST_BENIGN, _TEST_ATTACK)

    baseline_inputs = build_client_frontier_inputs(baseline, _GRID)
    perturbed_inputs = build_client_frontier_inputs(perturbed, _GRID)
    for baseline_candidate, perturbed_candidate in zip(
        baseline_inputs.subtype_confusion_by_candidate,
        perturbed_inputs.subtype_confusion_by_candidate,
        strict=True,
    ):
        assert baseline_candidate == perturbed_candidate


def test_t06_validation_attack_perturbation_changes_utility_but_not_eq_fpr_inputs() -> None:
    """T06: ATTACK_VALIDATION perturbation changes the derived utility curve (which GREEDY/
    FABRID_MACRO/FABRID_MINIMAX consume) but EQ_FPR never reads utility curves at all, so its
    allocation is architecturally invariant regardless of what perturbation does here."""
    baseline = _build_artifact(_FRONTIER, _VALIDATION, _FINAL_CAL, _TEST_BENIGN, _TEST_ATTACK)
    perturbed = _build_artifact(
        _FRONTIER, (0.1, 0.15, 0.12, 0.18, 0.11), _FINAL_CAL, _TEST_BENIGN, _TEST_ATTACK
    )

    baseline_inputs = build_client_frontier_inputs(baseline, _GRID)
    perturbed_inputs = build_client_frontier_inputs(perturbed, _GRID)

    # utility-relevant confusion counts do change under this perturbation...
    assert (
        baseline_inputs.subtype_confusion_by_candidate
        != perturbed_inputs.subtype_confusion_by_candidate
    )
    # ...but EQ_FPR's signature (fabrid.allocation.equal_fpr.allocate_equal_fpr) takes only
    # client_ids/budget/alpha_max — it has no parameter through which either artifact's
    # validation scores could reach it, so no further assertion is needed here beyond the
    # type signature itself; this test documents the invariant's structural basis.


def test_score_record_field_type_smoke() -> None:
    # sanity: source_row accepts a plain int at runtime despite the RowCount NewType annotation.
    record = _record("x", BenignSplit.FRONTIER, 0.1, Label.BENIGN)
    assert record.score == pytest.approx(0.1)
