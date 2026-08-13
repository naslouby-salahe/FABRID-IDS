from __future__ import annotations

import pytest

from fabrid.datasets.splitting import (
    compute_attack_split_boundary,
    compute_benign_split_boundaries,
)
from fabrid.domain.enums import AttackSplit, BenignSplit
from fabrid.domain.identifiers import ClientId
from fabrid.domain.values import RowCount, SourceRowIndex
from fabrid.protocol.specification import PROTOCOL


@pytest.mark.parametrize(
    ("client_id", "total", "train", "frontier", "final_cal", "test"),
    (
        (ClientId("Danmini_Doorbell"), RowCount(49_548), RowCount(24_774), RowCount(9_909), RowCount(4_955), RowCount(9_910)),
        (ClientId("Ennio_Doorbell"), RowCount(39_100), RowCount(19_550), RowCount(7_820), RowCount(3_910), RowCount(7_820)),
        (ClientId("Ecobee_Thermostat"), RowCount(13_113), RowCount(6_556), RowCount(2_623), RowCount(1_311), RowCount(2_623)),
        (ClientId("Philips_B120N10_Baby_Monitor"), RowCount(175_240), RowCount(87_620), RowCount(35_047), RowCount(17_525), RowCount(35_048)),
        (ClientId("Provision_PT_737E_Security_Camera"), RowCount(62_154), RowCount(31_077), RowCount(12_430), RowCount(6_216), RowCount(12_431)),
        (ClientId("Provision_PT_838_Security_Camera"), RowCount(98_514), RowCount(49_257), RowCount(19_702), RowCount(9_852), RowCount(19_703)),
        (ClientId("SimpleHome_XCS7_1002_WHT_Security_Camera"), RowCount(46_585), RowCount(23_292), RowCount(9_317), RowCount(4_659), RowCount(9_317)),
        (ClientId("SimpleHome_XCS7_1003_WHT_Security_Camera"), RowCount(19_528), RowCount(9_764), RowCount(3_905), RowCount(1_953), RowCount(3_906)),
        (ClientId("Samsung_SNH_1011_N_Webcam"), RowCount(52_150), RowCount(26_075), RowCount(10_430), RowCount(5_215), RowCount(10_430)),
    ),
)
def test_benign_split_matches_locked_nbaiot_counts(
    client_id: ClientId,
    total: RowCount,
    train: RowCount,
    frontier: RowCount,
    final_cal: RowCount,
    test: RowCount,
) -> None:
    counts = compute_benign_split_boundaries(total, PROTOCOL.benign_splits).counts()

    assert counts.train == train, client_id.value
    assert counts.frontier == frontier, client_id.value
    assert counts.final_cal == final_cal, client_id.value
    assert counts.test == test, client_id.value
    assert counts.total == total


def test_split_assignment_is_exclusive_at_boundaries() -> None:
    benign = compute_benign_split_boundaries(RowCount(10), PROTOCOL.benign_splits)
    assert benign.split_of(SourceRowIndex(0)) is BenignSplit.TRAIN
    assert benign.split_of(SourceRowIndex(5)) is BenignSplit.FRONTIER
    assert benign.split_of(SourceRowIndex(7)) is BenignSplit.FINAL_CAL
    assert benign.split_of(SourceRowIndex(8)) is BenignSplit.TEST

    attack = compute_attack_split_boundary(RowCount(100), PROTOCOL.attack_split)
    assert attack.validation_end == RowCount(20)
    assert attack.counts().validation == RowCount(20)
    assert attack.counts().test == RowCount(80)
    assert attack.split_of(SourceRowIndex(19)) is AttackSplit.VALIDATION
    assert attack.split_of(SourceRowIndex(20)) is AttackSplit.TEST


def test_split_assignment_rejects_out_of_population_index() -> None:
    benign = compute_benign_split_boundaries(RowCount(10), PROTOCOL.benign_splits)
    attack = compute_attack_split_boundary(RowCount(10), PROTOCOL.attack_split)

    with pytest.raises(ValueError):
        benign.split_of(SourceRowIndex(10))
    with pytest.raises(ValueError):
        attack.split_of(SourceRowIndex(10))
