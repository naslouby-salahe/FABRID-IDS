from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
from pydantic import TypeAdapter

from fabrid.allocation.optimization import OptimizedAllocation
from fabrid.allocation.policies import allocate_fabrid_macro
from fabrid.allocation.problem import (
    AllocationProblem,
    ClientRowCount,
    FrontierScoreArtifacts,
    build_allocation_problem,
    eligible_subtypes,
    equal_client_weights,
)
from fabrid.artifacts.parquet import read_parquet_models
from fabrid.artifacts.paths import ExperimentCoordinate
from fabrid.config import (
    AnalysisArtifactId,
    AttackSplit,
    BenignSplit,
    DatasetId,
    ExperimentId,
    ExperimentVariantId,
    FabridConfig,
    Label,
    SolverConfig,
    WeightMode,
)
from fabrid.detector.scoring import ScoreCoordinate, ScorePartitionArtifact, ScoreRecord
from fabrid.errors import SolverInvalidError
from fabrid.evaluation.metrics import ClientStabilitySummary, StabilityReplicate
from fabrid.experiments.matched_budget import (
    ClientEvaluationArtifacts,
    EvaluationProvenance,
    LoadedClientScores,
    LoadedSeedScores,
    build_frontier_inputs,
)
from fabrid.experiments.sensitivity import (
    conservative_utility_curves,
    measure_seed_utility_heterogeneity,
    one_sided_binomial_lower_bound,
    resample_attack_validation,
    resample_partition,
    run_allocation_stability_seed_budget,
    run_conservative_utility_seed_budget,
    run_weight_sensitivity_seed,
)
from tests.support import isolated_paths, production_protocol, smoke_protocol

from ..allocation.synthetic_federation import synthetic_records


def _loaded_client(
    client_id: str,
    *,
    benign_frontier: np.ndarray,
    attack_validation: tuple[tuple[str, np.ndarray], ...],
) -> LoadedClientScores:
    attack_records = tuple(
        record
        for subtype, scores in attack_validation
        for record in synthetic_records(
            client_id,
            scores,
            AttackSplit.VALIDATION,
            label=Label.ATTACK,
            subtype=subtype,
        )
    )
    return LoadedClientScores(
        client_id=client_id,
        frontier=FrontierScoreArtifacts(
            benign_frontier=_partition(
                synthetic_records(
                    client_id, benign_frontier, BenignSplit.FRONTIER, label=Label.BENIGN
                ),
                client_id,
                BenignSplit.FRONTIER,
            ),
            attack_validation=_partition(attack_records, client_id, AttackSplit.VALIDATION),
        ),
        evaluation=ClientEvaluationArtifacts(
            client_id=client_id,
            final_calibration=_partition(
                synthetic_records(
                    client_id, np.linspace(0.0, 1.0, 150), BenignSplit.FINAL_CAL, label=Label.BENIGN
                ),
                client_id,
                BenignSplit.FINAL_CAL,
            ),
            benign_test=_partition(
                synthetic_records(
                    client_id, np.linspace(0.0, 1.0, 200), BenignSplit.TEST, label=Label.BENIGN
                ),
                client_id,
                BenignSplit.TEST,
            ),
            attack_test=_partition(
                synthetic_records(
                    client_id,
                    np.linspace(0.5, 1.0, 120),
                    AttackSplit.TEST,
                    label=Label.ATTACK,
                    subtype="mirai",
                ),
                client_id,
                AttackSplit.TEST,
            ),
        ),
    )


def _loaded_scores() -> LoadedSeedScores:
    benign = np.linspace(0.0, 1.0, 400)
    attack = np.linspace(0.5, 1.0, 100)
    subtypes = (("mirai", attack), ("bashlite", attack))
    return LoadedSeedScores(
        clients=(
            _loaded_client("a", benign_frontier=benign, attack_validation=subtypes),
            _loaded_client("b", benign_frontier=benign, attack_validation=subtypes),
        )
    )


def _allocation_problem(loaded: LoadedSeedScores, config: FabridConfig) -> AllocationProblem:
    return build_allocation_problem(
        build_frontier_inputs(loaded, config.alpha_grid),
        equal_client_weights(loaded.population),
        config.budgets[0].value,
        config.utility_eligibility,
        config.maximum_target_rate,
    )


def _partition(
    records: tuple[ScoreRecord, ...],
    client_id: str,
    split: BenignSplit | AttackSplit,
) -> ScorePartitionArtifact:
    return ScorePartitionArtifact(
        coordinate=ScoreCoordinate(
            dataset_id=DatasetId.NBAIOT, detector_seed=0, client_id=client_id
        ),
        split=split,
        records=records,
    )


def _provenance() -> EvaluationProvenance:
    return TypeAdapter(EvaluationProvenance).validate_python(
        {
            "model_sha256": "a" * 64,
            "score_sha256": "b" * 64,
            "split_sha256": "c" * 64,
            "feature_sha256": "d" * 64,
            "protocol_sha256": "e" * 64,
            "git_commit": "f" * 40,
        }
    )


def test_one_sided_lcb_is_monotone_in_detections() -> None:
    confidence = production_protocol().sensitivity.conservative_utility_confidence
    lower = one_sided_binomial_lower_bound(10, 90, confidence)
    higher = one_sided_binomial_lower_bound(50, 50, confidence)
    assert 0.0 < lower < higher < 1.0


def test_one_sided_lcb_zero_when_no_detections() -> None:
    confidence = production_protocol().sensitivity.conservative_utility_confidence
    assert one_sided_binomial_lower_bound(0, 100, confidence) == 0.0


def test_one_sided_lcb_never_exceeds_empirical_recall() -> None:
    confidence = production_protocol().sensitivity.conservative_utility_confidence
    true_positive = 80
    false_negative = 20
    recall = one_sided_binomial_lower_bound(true_positive, false_negative, confidence)
    assert recall < true_positive / (true_positive + false_negative)


def test_conservative_utility_curves_are_typed_and_bounded() -> None:
    config = production_protocol()
    problem = _allocation_problem(_loaded_scores(), config)
    curves = conservative_utility_curves(problem, config.utility_eligibility, config.sensitivity)
    assert len(curves) == 2
    for curve in curves:
        assert curve.points[0].target_rate == config.alpha_grid[0]
        assert all(0.0 <= point.utility <= 1.0 for point in curve.points)
        assert len(curve.points) == len(config.alpha_grid)


def test_conservative_utility_lcb_below_empirical_recall() -> None:
    config = production_protocol()
    problem = _allocation_problem(_loaded_scores(), config)
    curves = conservative_utility_curves(problem, config.utility_eligibility, config.sensitivity)
    for curve, client in zip(curves, problem.inputs.clients, strict=True):
        selection = eligible_subtypes(client, config.utility_eligibility)
        for point, candidate in zip(curve.points, client.candidates, strict=True):
            eligible_recalls = tuple(
                subtype.counts.true_positive_rate()
                for subtype in candidate.subtypes
                if selection.contains(subtype.subtype)
            )
            empirical_utility = sum(eligible_recalls) / len(eligible_recalls)
            if empirical_utility == 0.0:
                assert point.utility == 0.0
            else:
                assert point.utility < empirical_utility


def test_conservative_utility_uses_sensitivity_confidence() -> None:
    config = production_protocol()
    problem = _allocation_problem(_loaded_scores(), config)
    tighter = conservative_utility_curves(
        problem,
        config.utility_eligibility,
        config.sensitivity.model_copy(update={"conservative_utility_confidence": 0.99}),
    )
    looser = conservative_utility_curves(
        problem,
        config.utility_eligibility,
        config.sensitivity.model_copy(update={"conservative_utility_confidence": 0.80}),
    )
    compared = 0
    for tight_curve, loose_curve in zip(tighter, looser, strict=True):
        for tight_point, loose_point in zip(tight_curve.points, loose_curve.points, strict=True):
            if loose_point.utility == 0.0:
                assert tight_point.utility == 0.0
                continue
            compared += 1
            assert tight_point.utility < loose_point.utility
    assert compared > 0


def test_utility_heterogeneity_emits_one_row_per_candidate() -> None:
    config = production_protocol()
    loaded = _loaded_scores()
    rows = measure_seed_utility_heterogeneity(0, loaded, config)
    assert len(rows) == len(config.alpha_grid)
    for row, candidate_rate in zip(rows, config.alpha_grid, strict=True):
        assert row.seed == 0
        assert row.candidate_alpha == candidate_rate
        assert 0.0 <= row.dispersion <= 1.0
        assert row.aggregate == rows[0].aggregate
    assert rows[0].aggregate == sum(row.dispersion for row in rows) / len(rows)


def test_utility_heterogeneity_empty_for_single_client() -> None:
    config = production_protocol()
    loaded = LoadedSeedScores(
        clients=tuple(client for client in _loaded_scores().clients if client.client_id == "a")
    )
    assert measure_seed_utility_heterogeneity(0, loaded, config) == ()


def test_utility_heterogeneity_empty_for_single_eligible_client() -> None:
    config = production_protocol()
    benign = np.linspace(0.0, 1.0, 400)
    loaded = LoadedSeedScores(
        clients=(
            _loaded_client(
                "a",
                benign_frontier=benign,
                attack_validation=(
                    ("mirai", np.linspace(0.5, 1.0, 100)),
                    ("bashlite", np.linspace(0.5, 1.0, 100)),
                ),
            ),
            _loaded_client(
                "b",
                benign_frontier=benign,
                attack_validation=(("mirai", np.linspace(0.5, 1.0, 20)),),
            ),
        )
    )
    assert measure_seed_utility_heterogeneity(0, loaded, config) == ()


def test_utility_heterogeneity_uses_allocation_utility_not_conservative_lcb() -> None:
    config = production_protocol()
    benign = np.linspace(0.0, 1.0, 400)
    high = np.full(100, 10.0)
    low = np.full(100, -10.0)
    rare_high = np.full(10, 10.0)
    rare_low = np.full(10, -10.0)
    loaded = LoadedSeedScores(
        clients=(
            _loaded_client(
                "a",
                benign_frontier=benign,
                attack_validation=(
                    ("mirai", high),
                    ("bashlite", high),
                    ("gafgyt_junk", rare_low),
                ),
            ),
            _loaded_client(
                "b",
                benign_frontier=benign,
                attack_validation=(
                    ("mirai", low),
                    ("bashlite", low),
                    ("gafgyt_junk", rare_high),
                ),
            ),
        )
    )
    rows = measure_seed_utility_heterogeneity(0, loaded, config)
    assert len(rows) == len(config.alpha_grid)
    resolved = tuple(row for row in rows if row.dispersion > 0.0)
    assert resolved
    for row in resolved:
        assert row.dispersion == 0.5
    confidence = config.sensitivity.conservative_utility_confidence
    conservative_pair_sd = one_sided_binomial_lower_bound(100, 0, confidence) / 2.0
    assert conservative_pair_sd < 0.5
    assert all(row.dispersion > conservative_pair_sd for row in resolved)


def test_stability_slices_cover_replicate_range() -> None:
    slice_size = 32
    for replicate_count in (1, 32, 33, 64, 500):
        slices = tuple(range(0, replicate_count, slice_size))
        assert slices[0] == 0
        covered: set[int] = set()
        for start in slices:
            covered.update(range(start, min(start + slice_size, replicate_count)))
        assert covered == set(range(replicate_count)), replicate_count


def test_stability_attack_resample_preserves_subtype_composition() -> None:
    artifact = _partition(
        synthetic_records(
            "a", np.array([0.1, 0.2]), AttackSplit.VALIDATION, label=Label.ATTACK, subtype="x"
        )
        + synthetic_records(
            "a",
            np.array([0.3, 0.4, 0.5]),
            AttackSplit.VALIDATION,
            label=Label.ATTACK,
            subtype="y",
        ),
        "a",
        AttackSplit.VALIDATION,
    )
    resampled = resample_attack_validation(artifact, random.Random(7))
    assert len(resampled.records) == len(artifact.records)
    assert sum(record.attack_subtype == "x" for record in resampled.records) == 2
    assert sum(record.attack_subtype == "y" for record in resampled.records) == 3


def test_generic_resample_rejects_attack_validation_artifacts() -> None:
    rng = random.Random(7)
    artifact = _partition(
        synthetic_records(
            "a", np.array([0.1]), AttackSplit.VALIDATION, label=Label.ATTACK, subtype="x"
        ),
        "a",
        AttackSplit.VALIDATION,
    )
    with pytest.raises(ValueError, match="subtype-stratified"):
        resample_partition(artifact, rng)


def test_stability_replicate_tolerates_infeasible_macro_solve(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = smoke_protocol()
    loaded = _loaded_scores()
    paths = isolated_paths(tmp_path)
    real_allocator = allocate_fabrid_macro
    attempts = 0

    def first_infeasible_then_real(
        problem: AllocationProblem,
        solver: SolverConfig,
    ) -> OptimizedAllocation:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise SolverInvalidError("SOLVER_INVALID: status=INFEASIBLE")
        return real_allocator(problem, solver)

    monkeypatch.setattr(
        "fabrid.experiments.sensitivity.allocate_fabrid_macro",
        first_infeasible_then_real,
    )
    summaries = run_allocation_stability_seed_budget(
        0,
        config.budgets[0],
        loaded,
        config,
        paths,
        DatasetId.NBAIOT,
    )
    assert tuple(summary.client_id for summary in summaries) == tuple(
        client.client_id for client in loaded.clients
    )


def test_stability_replicates_include_deployed_outcomes(tmp_path: Path) -> None:
    config = smoke_protocol()
    loaded = _loaded_scores()
    paths = isolated_paths(tmp_path)
    run_allocation_stability_seed_budget(
        0, config.budgets[0], loaded, config, paths, DatasetId.NBAIOT
    )
    coordinate = ExperimentCoordinate(
        experiment_id=ExperimentId.ALLOCATION_STABILITY,
        variant_id=ExperimentVariantId.PRIMARY,
        dataset_id=DatasetId.NBAIOT,
        detector_seed=0,
        budget_id=config.budgets[0].budget_id,
        budget=config.budgets[0].value,
        weight_mode=WeightMode.EQUAL_CLIENT,
    )
    rows = read_parquet_models(
        paths.analysis_path(coordinate, AnalysisArtifactId.STABILITY_REPLICATES),
        StabilityReplicate,
    )
    assert rows
    assert all(0.0 <= row.final_calibration_fpr <= 1.0 for row in rows)
    assert all(0.0 <= row.benign_test_fpr <= 1.0 for row in rows)
    assert all(0.0 <= row.macro_attack_recall <= 1.0 for row in rows)
    assert all(row.false_alert_count >= 0 for row in rows)
    assert all(row.federation_fpr >= 0.0 for row in rows)


def test_stability_reuses_only_completed_cell(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = smoke_protocol()
    loaded = _loaded_scores()
    paths = isolated_paths(tmp_path)
    expected = run_allocation_stability_seed_budget(
        0, config.budgets[0], loaded, config, paths, DatasetId.NBAIOT
    )

    def should_not_recompute(*args: object, **kwargs: object) -> tuple[StabilityReplicate, ...]:
        raise AssertionError("completed stability evidence was recomputed")

    monkeypatch.setattr(
        "fabrid.experiments.sensitivity._compute_stability_replicates", should_not_recompute
    )
    observed = run_allocation_stability_seed_budget(
        0, config.budgets[0], loaded, config, paths, DatasetId.NBAIOT
    )
    assert all(isinstance(summary, ClientStabilitySummary) for summary in observed)
    assert observed == expected


def test_conservative_utility_seed_budget_emits_macro_and_cvar(tmp_path: Path) -> None:
    config = smoke_protocol()
    loaded = _loaded_scores()
    paths = isolated_paths(tmp_path)
    runs = run_conservative_utility_seed_budget(
        0,
        config.budgets[0],
        loaded,
        config,
        _provenance(),
        paths,
        DatasetId.NBAIOT,
    )
    assert tuple(run.evaluation.experiment.variant_id for run in runs) == (
        ExperimentVariantId.CONSERVATIVE_MACRO,
        ExperimentVariantId.CONSERVATIVE_CVAR,
    )
    for run in runs:
        assert run.evaluation.experiment.experiment_id is ExperimentId.CONSERVATIVE_UTILITY
        assert run.evaluation.experiment.weight_mode is WeightMode.EQUAL_CLIENT


def test_weight_sensitivity_uses_configured_gamma_variants(tmp_path: Path) -> None:
    config = smoke_protocol()
    loaded = _loaded_scores()
    paths = isolated_paths(tmp_path)
    variants = config.sensitivity.weight_gamma_variants
    runs = run_weight_sensitivity_seed(
        0,
        loaded,
        config,
        _provenance(),
        paths,
        DatasetId.NBAIOT,
        (
            ClientRowCount(client_id="a", row_count=100),
            ClientRowCount(client_id="b", row_count=400),
        ),
    )
    expected_variant_ids = tuple(variant.variant_id for variant in variants for _ in config.budgets)
    assert tuple(run.evaluation.experiment.variant_id for run in runs) == expected_variant_ids
    for run in runs:
        assert run.evaluation.experiment.experiment_id is ExperimentId.WEIGHT_SENSITIVITY
        assert run.evaluation.experiment.weight_mode is WeightMode.DATASET_COUNT_PROXY


def _mixed_eligibility_scores() -> LoadedSeedScores:
    benign = np.linspace(0.0, 1.0, 400)
    return LoadedSeedScores(
        clients=(
            _loaded_client(
                "a",
                benign_frontier=benign,
                attack_validation=(
                    ("mirai", np.linspace(0.5, 1.0, 100)),
                    ("bashlite", np.linspace(0.5, 1.0, 100)),
                ),
            ),
            _loaded_client(
                "b",
                benign_frontier=benign,
                attack_validation=(("mirai", np.linspace(0.5, 1.0, 20)),),
            ),
        )
    )


def test_conservative_utility_curves_cover_only_eligible_clients() -> None:
    config = production_protocol()
    problem = _allocation_problem(_mixed_eligibility_scores(), config)
    population = problem.frontier.eligible_population()
    assert population is not None
    assert population.clients == ("a",)
    curves = conservative_utility_curves(problem, config.utility_eligibility, config.sensitivity)
    assert tuple(curve.client_id for curve in curves) == population.clients


def test_conservative_utility_averages_only_eligible_subtypes() -> None:
    config = production_protocol()
    benign = np.linspace(0.0, 1.0, 400)
    loaded = LoadedSeedScores(
        clients=(
            _loaded_client(
                "a",
                benign_frontier=benign,
                attack_validation=(
                    ("mirai", np.full(100, 10.0)),
                    ("bashlite", np.full(100, 10.0)),
                    ("gafgyt_junk", np.full(10, -10.0)),
                ),
            ),
            _loaded_client(
                "b",
                benign_frontier=benign,
                attack_validation=(
                    ("mirai", np.full(100, 10.0)),
                    ("bashlite", np.full(100, 10.0)),
                ),
            ),
        )
    )
    problem = _allocation_problem(loaded, config)
    curves = conservative_utility_curves(problem, config.utility_eligibility, config.sensitivity)
    confidence = config.sensitivity.conservative_utility_confidence
    eligible_only = one_sided_binomial_lower_bound(100, 0, confidence)
    all_subtypes = (2.0 * eligible_only + 0.0) / 3.0
    resolved = tuple(point.utility for point in curves[0].points if point.utility > 0.0)
    assert resolved
    assert all(utility == eligible_only for utility in resolved)
    assert all(utility != all_subtypes for utility in resolved)


def test_conservative_utility_seed_budget_survives_one_fallback_client(
    tmp_path: Path,
) -> None:
    config = smoke_protocol()
    paths = isolated_paths(tmp_path)
    runs = run_conservative_utility_seed_budget(
        0,
        config.budgets[0],
        _mixed_eligibility_scores(),
        config,
        _provenance(),
        paths,
        DatasetId.NBAIOT,
    )
    assert tuple(run.evaluation.experiment.variant_id for run in runs) == (
        ExperimentVariantId.CONSERVATIVE_MACRO,
        ExperimentVariantId.CONSERVATIVE_CVAR,
    )


def test_conservative_utility_seed_budget_skips_when_none_eligible(tmp_path: Path) -> None:
    config = smoke_protocol()
    paths = isolated_paths(tmp_path)
    benign = np.linspace(0.0, 1.0, 400)
    loaded = LoadedSeedScores(
        clients=(
            _loaded_client(
                "a",
                benign_frontier=benign,
                attack_validation=(("mirai", np.linspace(0.5, 1.0, 20)),),
            ),
            _loaded_client(
                "b",
                benign_frontier=benign,
                attack_validation=(("bashlite", np.linspace(0.5, 1.0, 20)),),
            ),
        )
    )
    runs = run_conservative_utility_seed_budget(
        0,
        config.budgets[0],
        loaded,
        config,
        _provenance(),
        paths,
        DatasetId.NBAIOT,
    )
    assert runs == ()
    coordinate = ExperimentCoordinate(
        experiment_id=ExperimentId.CONSERVATIVE_UTILITY,
        variant_id=ExperimentVariantId.CONSERVATIVE_MACRO,
        dataset_id=DatasetId.NBAIOT,
        detector_seed=0,
        budget_id=config.budgets[0].budget_id,
        budget=config.budgets[0].value,
        weight_mode=WeightMode.EQUAL_CLIENT,
    )
    assert not paths.analysis_path(
        coordinate, AnalysisArtifactId.CONSERVATIVE_UTILITY_CURVES
    ).exists()
