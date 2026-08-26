from __future__ import annotations

import hashlib
import random
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass

import numpy as np
from polars.exceptions import PolarsError
from pydantic import ValidationError
from scipy.stats import beta

from fabrid.allocation.optimization import OptimizedAllocation
from fabrid.allocation.policies import allocate_fabrid_cvar, allocate_fabrid_macro
from fabrid.allocation.problem import (
    AllocationProblem,
    AttackSubtypeSelection,
    CandidateConfusions,
    ClientRowCount,
    ClientUtilityCurve,
    ClientUtilityCurves,
    ClientUtilityPoint,
    FederationFrontierInputs,
    FrontierScoreArtifacts,
    build_allocation_problem,
    build_client_frontier_inputs,
    client_utility_variance,
    dataset_count_weights,
    eligible_subtypes,
    equal_client_weights,
    merge_full_allocation,
    weight_gamma_transform,
)
from fabrid.artifacts.parquet import read_parquet_models, write_parquet_models
from fabrid.artifacts.paths import (
    AllocationCoordinate,
    ArtifactPaths,
    ExperimentCoordinate,
)
from fabrid.config import (
    PERCENTAGE_POINTS_PER_UNIT,
    AllocationPolicy,
    AnalysisArtifactId,
    AttackSplit,
    BudgetId,
    BudgetLevel,
    ClientId,
    DatasetId,
    DetectionUtility,
    DetectorSeed,
    ExperimentId,
    ExperimentVariantId,
    FabridConfig,
    FalseAlertCount,
    FalsePositiveBudget,
    FalsePositiveRate,
    MacroRecall,
    Probability,
    ReplicateIndex,
    RowCount,
    SensitivityConfig,
    SolverConfig,
    TargetFalsePositiveRate,
    Threshold,
    UtilityEligibilityConfig,
    WeightMode,
    WorkerCount,
)
from fabrid.detector.calibration import (
    FinalCalibrationDecision,
    FinalCalibrationInputs,
    calibrate_final_thresholds,
)
from fabrid.detector.scoring import ScorePartitionArtifact
from fabrid.errors import SolverInvalidError
from fabrid.evaluation.metrics import (
    ClientResultRecord,
    ClientStabilitySummary,
    CompletedPolicyEvaluation,
    SeedBudgetEvaluation,
    StabilityReplicate,
    UtilityCurveRow,
    UtilityHeterogeneity,
    budget_usage_ratio,
    evaluate_allocation,
)
from fabrid.experiments.matched_budget import (
    EvaluationProvenance,
    LoadedSeedScores,
    SeedBudgetRun,
    build_frontier_inputs,
    persist_seed_budget,
    run_seed_budget,
)


def _replicate_rng(
    seed: DetectorSeed, budget: FalsePositiveBudget, replicate_index: ReplicateIndex
) -> random.Random:
    material = hashlib.sha256(f"{seed}:{budget}:{replicate_index}".encode()).digest()
    return random.Random(int.from_bytes(material[:8], byteorder="big", signed=False))


@dataclass(frozen=True, slots=True)
class _StabilityWorkerState:
    seed: DetectorSeed
    budget_level: BudgetLevel
    loaded: LoadedSeedScores
    config: FabridConfig


_worker_state: _StabilityWorkerState | None = None


def _initialize_stability_worker(state: _StabilityWorkerState) -> None:
    global _worker_state
    _worker_state = state


@dataclass(frozen=True, slots=True)
class _StabilityOutcome:
    client_id: ClientId
    final_calibration_fpr: FalsePositiveRate
    benign_test_fpr: FalsePositiveRate
    macro_attack_recall: MacroRecall
    false_alert_count: FalseAlertCount
    deployed_threshold: Threshold


def _run_stability_replicate(
    replicate_index: ReplicateIndex,
) -> tuple[StabilityReplicate, ...]:
    state = _worker_state
    assert state is not None, "stability worker state must be set before replicates run"
    rng = _replicate_rng(state.seed, state.budget_level.value, replicate_index)
    inputs = FederationFrontierInputs(
        clients=tuple(
            build_client_frontier_inputs(
                FrontierScoreArtifacts(
                    benign_frontier=resample_partition(client.frontier.benign_frontier, rng),
                    attack_validation=resample_attack_validation(
                        client.frontier.attack_validation, rng
                    ),
                ),
                state.config.alpha_grid,
                frontier_row_count=client.frontier.benign_frontier.row_count,
            )
            for client in state.loaded.clients
        )
    )
    weights = equal_client_weights(state.loaded.population)
    problem = build_allocation_problem(
        inputs,
        weights,
        state.budget_level.value,
        state.config.utility_eligibility,
        state.config.maximum_target_rate,
    )
    curves = problem.frontier.eligible_curves()
    if curves is None:
        return ()
    try:
        optimized = allocate_fabrid_macro(problem, state.config.solver)
    except SolverInvalidError:
        return ()
    allocation = merge_full_allocation(AllocationPolicy.FABRID_MACRO, problem, optimized.allocation)
    final_calibration = calibrate_final_thresholds(
        decisions=tuple(
            FinalCalibrationDecision(client_id=decision.client_id, target_rate=decision.target_rate)
            for decision in allocation.decisions
        ),
        inputs=FinalCalibrationInputs(
            clients=tuple(client.evaluation.final_calibration for client in state.loaded.clients)
        ),
    )
    outcomes: list[_StabilityOutcome] = []
    for client in state.loaded.clients:
        calibration = final_calibration.for_client(client.client_id)
        threshold = calibration.threshold
        benign_scores = client.evaluation.benign_test.score_values()
        false_alert_count = int(np.count_nonzero(benign_scores > threshold))
        benign_test_fpr = false_alert_count / benign_scores.size
        subtype_scores = client.evaluation.attack_test.subtype_scores()
        macro_recall = sum(
            float(np.count_nonzero(item.scores > threshold) / item.scores.size)
            for item in subtype_scores
        ) / len(subtype_scores)
        outcomes.append(
            _StabilityOutcome(
                client_id=client.client_id,
                final_calibration_fpr=calibration.reachability.attained_rate,
                benign_test_fpr=benign_test_fpr,
                macro_attack_recall=macro_recall,
                false_alert_count=false_alert_count,
                deployed_threshold=threshold,
            )
        )
    federation_fpr = sum(
        weights.for_client(outcome.client_id) * outcome.benign_test_fpr for outcome in outcomes
    )
    usage = budget_usage_ratio(federation_fpr, state.budget_level.value)
    assert usage is not None
    outcome_by_client = {outcome.client_id: outcome for outcome in outcomes}
    return tuple(
        StabilityReplicate(
            seed=state.seed,
            budget_id=state.budget_level.budget_id,
            replicate_index=replicate_index,
            client_id=decision.client_id,
            alpha_selected=decision.target_rate,
            deployed_threshold=outcome_by_client[decision.client_id].deployed_threshold,
            final_calibration_fpr=outcome_by_client[decision.client_id].final_calibration_fpr,
            benign_test_fpr=outcome_by_client[decision.client_id].benign_test_fpr,
            macro_attack_recall=outcome_by_client[decision.client_id].macro_attack_recall,
            false_alert_count=outcome_by_client[decision.client_id].false_alert_count,
            federation_fpr=federation_fpr,
            budget_usage=usage,
        )
        for decision in allocation.decisions
    )


def resample_partition(
    artifact: ScorePartitionArtifact,
    rng: random.Random,
) -> ScorePartitionArtifact:
    if artifact.split is AttackSplit.VALIDATION:
        raise ValueError("attack validation requires subtype-stratified resampling")
    records = artifact.records
    return ScorePartitionArtifact(
        coordinate=artifact.coordinate,
        split=artifact.split,
        records=tuple(records[rng.randrange(len(records))] for _ in range(len(records))),
    )


def resample_attack_validation(
    artifact: ScorePartitionArtifact,
    rng: random.Random,
) -> ScorePartitionArtifact:
    if artifact.split is not AttackSplit.VALIDATION:
        raise ValueError("stratified resampling requires ATTACK_VALIDATION scores")
    for record in artifact.records:
        if record.attack_subtype is None:
            raise ValueError("attack-validation score is missing an attack subtype")
    subtypes = tuple(item.subtype for item in artifact.subtype_scores())
    records = tuple(
        record
        for subtype in subtypes
        for source in (
            tuple(record for record in artifact.records if record.attack_subtype == subtype),
        )
        for record in (source[rng.randrange(len(source))] for _ in range(len(source)))
    )
    return ScorePartitionArtifact(
        coordinate=artifact.coordinate,
        split=artifact.split,
        records=records,
    )


def _summarize_client_stability(
    seed: DetectorSeed,
    budget_id: BudgetId,
    client_id: ClientId,
    selected_rates: tuple[TargetFalsePositiveRate, ...],
    sensitivity: SensitivityConfig,
) -> ClientStabilitySummary:
    if not selected_rates:
        raise ValueError("stability summary requires at least one selected target rate")
    unique_rates = tuple(sorted(set(selected_rates)))
    frequencies = tuple(selected_rates.count(rate) for rate in unique_rates)
    modal_index = max(range(len(unique_rates)), key=frequencies.__getitem__)
    modal_count = frequencies[modal_index]
    total = len(selected_rates)
    lower_percentile, upper_percentile = sensitivity.stability_percentiles
    return ClientStabilitySummary(
        seed=seed,
        budget_id=budget_id,
        client_id=client_id,
        modal_alpha=unique_rates[modal_index],
        modal_frequency=modal_count / total,
        median_alpha=float(np.median(selected_rates)),
        percentile_5=float(
            np.percentile(selected_rates, lower_percentile * PERCENTAGE_POINTS_PER_UNIT)
        ),
        percentile_95=float(
            np.percentile(selected_rates, upper_percentile * PERCENTAGE_POINTS_PER_UNIT)
        ),
        instability=1.0 - modal_count / total,
    )


def _run_stability_slice(
    slice_start: ReplicateIndex,
    slice_size: RowCount,
    replicate_count: RowCount,
) -> tuple[ReplicateIndex, tuple[StabilityReplicate, ...]]:
    end = min(slice_start + slice_size, replicate_count)
    return slice_start, tuple(
        row
        for replicate_index in range(slice_start, end)
        for row in _run_stability_replicate(replicate_index)
    )


def _compute_stability_replicates(
    seed: DetectorSeed,
    budget_level: BudgetLevel,
    loaded: LoadedSeedScores,
    config: FabridConfig,
    replicate_count: RowCount,
    worker_limit: WorkerCount | None = None,
) -> tuple[StabilityReplicate, ...]:
    global _worker_state
    previous = _worker_state
    _worker_state = _StabilityWorkerState(seed, budget_level, loaded, config)
    try:
        workers = worker_limit or config.sensitivity.stability_workers
        if workers <= 1 or replicate_count <= 1:
            rows = tuple(
                row
                for replicate_index in range(replicate_count)
                for row in _run_stability_replicate(replicate_index)
            )
            return rows
        return _run_stability_slices(
            workers,
            replicate_count,
            _StabilityWorkerState(seed, budget_level, loaded, config),
        )
    finally:
        _worker_state = previous


def _run_stability_slices(
    workers: WorkerCount,
    replicate_count: RowCount,
    state: _StabilityWorkerState,
) -> tuple[StabilityReplicate, ...]:
    slice_size = 32
    slice_count = (replicate_count + slice_size - 1) // slice_size
    if slice_count <= 1:
        return _run_stability_slice(0, slice_size, replicate_count)[1]
    completed: list[tuple[StabilityReplicate, ...] | None] = [None] * slice_count
    pending = set(range(slice_count))
    for _ in range(2):
        if not pending:
            break
        pool = ProcessPoolExecutor(
            max_workers=workers,
            initializer=_initialize_stability_worker,
            initargs=(state,),
        )
        try:
            futures = {
                pool.submit(
                    _run_stability_slice, position * slice_size, slice_size, replicate_count
                ): position
                for position in pending
            }
            for future in as_completed(futures):
                position = futures[future]
                try:
                    _, rows = future.result()
                except BrokenProcessPool:
                    break
                else:
                    completed[position] = rows
                    pending.discard(position)
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
    if pending:
        for position in sorted(pending):
            completed[position] = _run_stability_slice(
                position * slice_size, slice_size, replicate_count
            )[1]
    return tuple(row for position in range(slice_count) for row in (completed[position] or ()))


def run_allocation_stability_seed_budget(
    seed: DetectorSeed,
    budget_level: BudgetLevel,
    loaded: LoadedSeedScores,
    config: FabridConfig,
    paths: ArtifactPaths,
    dataset_id: DatasetId,
    worker_limit: WorkerCount | None = None,
) -> tuple[ClientStabilitySummary, ...]:
    sensitivity = config.sensitivity
    replicate_count = sensitivity.allocation_sensitivity_replicates
    coordinate = ExperimentCoordinate(
        experiment_id=ExperimentId.ALLOCATION_STABILITY,
        variant_id=ExperimentVariantId.PRIMARY,
        dataset_id=dataset_id,
        detector_seed=seed,
        budget_id=budget_level.budget_id,
        budget=budget_level.value,
        weight_mode=WeightMode.EQUAL_CLIENT,
    )
    completed = _load_completed_stability_summaries(coordinate, loaded, replicate_count, paths)
    if completed is not None:
        return completed
    replicate_rows = _compute_stability_replicates(
        seed, budget_level, loaded, config, replicate_count, worker_limit
    )
    summaries = tuple(
        _summarize_client_stability(
            seed,
            budget_level.budget_id,
            client.client_id,
            tuple(
                row.alpha_selected for row in replicate_rows if row.client_id == client.client_id
            ),
            sensitivity,
        )
        for client in loaded.clients
        if any(row.client_id == client.client_id for row in replicate_rows)
    )
    write_parquet_models(
        paths.analysis_path(coordinate, AnalysisArtifactId.STABILITY_REPLICATES),
        tuple(replicate_rows),
    )
    write_parquet_models(
        paths.analysis_path(coordinate, AnalysisArtifactId.STABILITY_SUMMARY),
        summaries,
    )
    return summaries


def _load_completed_stability_summaries(
    coordinate: ExperimentCoordinate,
    loaded: LoadedSeedScores,
    replicate_count: RowCount,
    paths: ArtifactPaths,
) -> tuple[ClientStabilitySummary, ...] | None:
    replicate_path = paths.analysis_path(coordinate, AnalysisArtifactId.STABILITY_REPLICATES)
    summary_path = paths.analysis_path(coordinate, AnalysisArtifactId.STABILITY_SUMMARY)
    if not (replicate_path.is_file() and summary_path.is_file()):
        return None
    try:
        replicates = read_parquet_models(replicate_path, StabilityReplicate)
        summaries = read_parquet_models(summary_path, ClientStabilitySummary)
    except (OSError, PolarsError, ValidationError, ValueError):
        return None
    client_ids = tuple(client.client_id for client in loaded.clients)
    expected_replicates = replicate_count * len(client_ids)
    if len(replicates) != expected_replicates or len(summaries) != len(client_ids):
        return None
    if (
        any(
            row.seed != coordinate.detector_seed
            or row.budget_id != coordinate.budget_id
            or row.client_id not in client_ids
            for row in replicates
        )
        or any(
            row.seed != coordinate.detector_seed
            or row.budget_id != coordinate.budget_id
            or row.client_id not in client_ids
            for row in summaries
        )
        or {row.client_id for row in summaries} != set(client_ids)
        or any(
            sum(row.client_id == client_id for row in replicates) != replicate_count
            for client_id in client_ids
        )
    ):
        return None
    return summaries


def one_sided_binomial_lower_bound(
    true_positive: RowCount,
    false_negative: RowCount,
    confidence: Probability,
) -> Probability:
    if true_positive == 0:
        return 0.0
    return float(beta.ppf(1.0 - confidence, true_positive, false_negative + 1))


def _conservative_candidate_utility(
    candidate: CandidateConfusions,
    selection: AttackSubtypeSelection,
    confidence: Probability,
) -> DetectionUtility:
    recalls = tuple(
        one_sided_binomial_lower_bound(
            subtype.counts.true_positive,
            subtype.counts.false_negative,
            confidence,
        )
        for subtype in candidate.subtypes
        if selection.contains(subtype.subtype)
    )
    if not recalls:
        raise ValueError("conservative utility requires at least one eligible subtype")
    return sum(recalls) / len(recalls)


def conservative_utility_curves(
    problem: AllocationProblem,
    eligibility: UtilityEligibilityConfig,
    sensitivity: SensitivityConfig,
) -> tuple[ClientUtilityCurve, ...]:
    population = problem.frontier.eligible_population()
    if population is None:
        return ()
    eligible_ids = set(population.clients)
    confidence = sensitivity.conservative_utility_confidence
    curves: list[ClientUtilityCurve] = []
    for client in problem.inputs.clients:
        if client.client_id not in eligible_ids:
            continue
        selection = eligible_subtypes(client, eligibility)
        curves.append(
            ClientUtilityCurve(
                client_id=client.client_id,
                points=tuple(
                    ClientUtilityPoint(
                        target_rate=candidate.target_rate,
                        utility=_conservative_candidate_utility(candidate, selection, confidence),
                        utility_variance=client_utility_variance(candidate, selection),
                    )
                    for candidate in client.candidates
                ),
            )
        )
    return tuple(curves)


def _seed_allocation_problem(
    loaded: LoadedSeedScores,
    config: FabridConfig,
    budget: FalsePositiveBudget,
) -> AllocationProblem:
    return build_allocation_problem(
        build_frontier_inputs(loaded, config.alpha_grid),
        equal_client_weights(loaded.population),
        budget,
        config.utility_eligibility,
        config.maximum_target_rate,
    )


Allocator = Callable[[AllocationProblem, SolverConfig], OptimizedAllocation]


def _run_conservative_policy(
    coordinate: AllocationCoordinate,
    problem: AllocationProblem,
    solver: SolverConfig,
    loaded: LoadedSeedScores,
    provenance: EvaluationProvenance,
    allocate: Allocator,
) -> tuple[CompletedPolicyEvaluation, tuple[ClientResultRecord, ...]]:
    optimized = allocate(problem, solver)
    full = merge_full_allocation(coordinate.policy, problem, optimized.allocation)
    fallback_rate = problem.frontier.fallback_rate
    weights = problem.weights
    return evaluate_allocation(
        coordinate,
        full,
        tuple(client.evaluation.final_calibration for client in loaded.clients),
        tuple(client.evaluation.benign_test for client in loaded.clients),
        tuple(client.evaluation.attack_test for client in loaded.clients),
        weights,
        optimized.solver,
        fallback_rate,
        provenance,
    )


def run_conservative_utility_seed_budget(
    seed: DetectorSeed,
    budget_level: BudgetLevel,
    loaded: LoadedSeedScores,
    config: FabridConfig,
    provenance: EvaluationProvenance,
    paths: ArtifactPaths,
    dataset_id: DatasetId,
) -> tuple[SeedBudgetRun, ...]:
    problem = _seed_allocation_problem(loaded, config, budget_level.value)
    conservative = conservative_utility_curves(
        problem, config.utility_eligibility, config.sensitivity
    )
    if not conservative:
        return ()
    curves = ClientUtilityCurves(clients=conservative)
    runs: list[SeedBudgetRun] = []
    for policy, variant, allocate in (
        (
            AllocationPolicy.FABRID_MACRO,
            ExperimentVariantId.CONSERVATIVE_MACRO,
            allocate_fabrid_macro,
        ),
        (
            AllocationPolicy.FABRID_CVAR,
            ExperimentVariantId.CONSERVATIVE_CVAR,
            allocate_fabrid_cvar,
        ),
    ):
        coordinate = ExperimentCoordinate(
            experiment_id=ExperimentId.CONSERVATIVE_UTILITY,
            variant_id=variant,
            dataset_id=dataset_id,
            detector_seed=seed,
            budget_id=budget_level.budget_id,
            budget=budget_level.value,
            weight_mode=WeightMode.EQUAL_CLIENT,
        )
        evaluation, records = _run_conservative_policy(
            AllocationCoordinate(coordinate, policy),
            problem.with_eligible_curves(curves),
            config.solver,
            loaded,
            provenance,
            allocate,
        )
        run = SeedBudgetRun(
            evaluation=SeedBudgetEvaluation(experiment=coordinate, policies=(evaluation,)),
            records=records,
            policy_runs=(),
        )
        runs.append(run)
    curve_coordinate = ExperimentCoordinate(
        experiment_id=ExperimentId.CONSERVATIVE_UTILITY,
        variant_id=ExperimentVariantId.CONSERVATIVE_MACRO,
        dataset_id=dataset_id,
        detector_seed=seed,
        budget_id=budget_level.budget_id,
        budget=budget_level.value,
        weight_mode=WeightMode.EQUAL_CLIENT,
    )
    write_parquet_models(
        paths.analysis_path(curve_coordinate, AnalysisArtifactId.CONSERVATIVE_UTILITY_CURVES),
        tuple(
            UtilityCurveRow(
                client_id=curve.client_id,
                target_rate=point.target_rate,
                utility=point.utility,
            )
            for curve in curves.clients
            for point in curve.points
        ),
    )
    return tuple(runs)


def run_weight_sensitivity_seed(
    seed: DetectorSeed,
    loaded: LoadedSeedScores,
    config: FabridConfig,
    provenance: EvaluationProvenance,
    paths: ArtifactPaths,
    dataset_id: DatasetId,
    benign_row_counts: tuple[ClientRowCount, ...],
) -> tuple[SeedBudgetRun, ...]:
    reference = dataset_count_weights(loaded.population, benign_row_counts)
    runs: list[SeedBudgetRun] = []
    for gamma_variant in config.sensitivity.weight_gamma_variants:
        weights = weight_gamma_transform(reference, gamma_variant.gamma)
        for budget_level in config.budgets:
            run = run_seed_budget(
                ExperimentCoordinate(
                    experiment_id=ExperimentId.WEIGHT_SENSITIVITY,
                    variant_id=gamma_variant.variant_id,
                    dataset_id=dataset_id,
                    detector_seed=seed,
                    budget_id=budget_level.budget_id,
                    budget=budget_level.value,
                    weight_mode=WeightMode.DATASET_COUNT_PROXY,
                ),
                loaded,
                config,
                provenance,
                weights,
            )
            runs.append(run)
            persist_seed_budget(paths, run)
    return tuple(runs)


def measure_seed_utility_heterogeneity(
    seed: DetectorSeed,
    loaded: LoadedSeedScores,
    config: FabridConfig,
) -> tuple[UtilityHeterogeneity, ...]:
    problem = _seed_allocation_problem(loaded, config, config.budgets[0].value)
    curves = problem.frontier.eligible_curves()
    if curves is None or len(curves.clients) < 2:
        return ()
    utilities = np.asarray(
        [[point.utility for point in curve.points] for curve in curves.clients],
        dtype=np.float64,
    )
    dispersions = tuple(float(np.std(utilities[:, index])) for index in range(utilities.shape[1]))
    aggregate = float(np.mean(dispersions))
    candidate_rates = tuple(point.target_rate for point in curves.clients[0].points)
    return tuple(
        UtilityHeterogeneity(
            seed=seed,
            candidate_alpha=candidate_rate,
            dispersion=dispersion,
            aggregate=aggregate,
        )
        for candidate_rate, dispersion in zip(candidate_rates, dispersions, strict=True)
    )
