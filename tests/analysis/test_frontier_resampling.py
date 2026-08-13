from __future__ import annotations

import numpy as np

from fabrid.allocation.frontier_inputs import AttackSubtypeScores, FrontierScorePopulation
from fabrid.analysis.frontier_resampling import bootstrap_frontier_populations
from fabrid.domain.identifiers import AttackSubtypeId, ClientId
from fabrid.domain.scores import ScoreVector
from fabrid.domain.values import AnalysisSeed, RowCount


def _population(client_id: ClientId) -> FrontierScorePopulation:
    return FrontierScorePopulation(
        client_id=client_id,
        benign_frontier=ScoreVector(
            np.asarray((0.1, 0.2, 0.3, 0.4, 0.5), dtype=np.float64)
        ),
        attack_validation=(
            AttackSubtypeScores(
                AttackSubtypeId("scan"),
                ScoreVector(np.asarray((0.6, 0.7, 0.8), dtype=np.float64)),
            ),
            AttackSubtypeScores(
                AttackSubtypeId("udp"),
                ScoreVector(np.asarray((0.55, 0.65, 0.75, 0.85), dtype=np.float64)),
            ),
        ),
    )


def test_frontier_bootstrap_is_deterministic_for_same_analysis_seed() -> None:
    source = (_population(ClientId("first")), _population(ClientId("second")))

    first = bootstrap_frontier_populations(source, AnalysisSeed(7))
    second = bootstrap_frontier_populations(source, AnalysisSeed(7))

    assert tuple(
        tuple(population.benign_frontier.values) for population in first
    ) == tuple(
        tuple(population.benign_frontier.values) for population in second
    )
    assert tuple(
        tuple(tuple(subtype.scores.values) for subtype in population.attack_validation)
        for population in first
    ) == tuple(
        tuple(tuple(subtype.scores.values) for subtype in population.attack_validation)
        for population in second
    )


def test_frontier_bootstrap_preserves_client_and_subtype_sample_counts() -> None:
    source = (_population(ClientId("device")),)

    resampled = bootstrap_frontier_populations(source, AnalysisSeed(11))[0]

    assert resampled.client_id == ClientId("device")
    assert resampled.benign_frontier.row_count == RowCount(5)
    assert tuple(
        (subtype.subtype, subtype.scores.row_count)
        for subtype in resampled.attack_validation
    ) == (
        (AttackSubtypeId("scan"), RowCount(3)),
        (AttackSubtypeId("udp"), RowCount(4)),
    )


def test_frontier_bootstrap_can_repeat_observations_without_artifact_identity_fabrication() -> None:
    source = (_population(ClientId("device")),)

    resampled = bootstrap_frontier_populations(source, AnalysisSeed(0))[0]

    assert resampled.benign_frontier.row_count == source[0].benign_frontier.row_count
    assert set(resampled.benign_frontier.values).issubset(
        set(source[0].benign_frontier.values)
    )
