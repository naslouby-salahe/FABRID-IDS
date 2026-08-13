from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fabrid.allocation.contracts import Allocation, FederationWeights
from fabrid.allocation.solver import SolverEvidence
from fabrid.artifacts.score import ScorePartitionArtifact
from fabrid.calibration.final_calibration import (
    FinalCalibrationInputs,
    calibrate_final_thresholds,
)
from fabrid.calibration.order_statistic import alerts_above_threshold
from fabrid.domain.coordinates import AllocationCoordinate
from fabrid.domain.enums import AttackSplit, BenignSplit
from fabrid.domain.identifiers import AttackSubtypeId, ClientId
from fabrid.domain.population import ClientPopulation
from fabrid.domain.provenance import ExperimentProvenance
from fabrid.domain.scores import ScoreVector
from fabrid.domain.values import (
    FalseAlertCount,
    FalsePositiveRate,
    Probability,
    RowCount,
    TruePositiveRate,
)
from fabrid.evaluation.metrics import (
    ClientFalsePositiveRate,
    ClientMacroRecall,
    SubtypeRecall,
    budget_usage_ratio,
    client_macro_recall,
    federation_fpr,
    federation_macro_recall,
    worst_client_recall,
)
from fabrid.evaluation.results import (
    CalibrationOutcome,
    ClientResultRecord,
    ClientWeights,
    CompletedPolicyEvaluation,
    ConfusionCounts,
    DetectionMetrics,
)


@dataclass(frozen=True, slots=True)
class ClientEvaluationArtifacts:
    final_calibration: ScorePartitionArtifact
    benign_test: ScorePartitionArtifact
    attack_test: ScorePartitionArtifact

    def __post_init__(self) -> None:
        if self.final_calibration.split is not BenignSplit.FINAL_CAL:
            raise ValueError("evaluation requires BENIGN_FINAL_CAL for final calibration")
        if self.benign_test.split is not BenignSplit.TEST:
            raise ValueError("evaluation requires BENIGN_TEST")
        if self.attack_test.split is not AttackSplit.TEST:
            raise ValueError("evaluation requires ATTACK_TEST")
        coordinates = (
            self.final_calibration.coordinate,
            self.benign_test.coordinate,
            self.attack_test.coordinate,
        )
        if len(set(coordinates)) != 1:
            raise ValueError("client evaluation artifacts must share one score coordinate")

    @property
    def client_id(self) -> ClientId:
        return self.benign_test.coordinate.client_id


@dataclass(frozen=True, slots=True)
class FederationEvaluationArtifacts:
    clients: tuple[ClientEvaluationArtifacts, ...]

    def __post_init__(self) -> None:
        if not self.clients:
            raise ValueError("federation evaluation requires at least one client")
        client_ids = tuple(client.client_id for client in self.clients)
        if len(set(client_ids)) != len(client_ids):
            raise ValueError("federation evaluation contains duplicate clients")

    def for_client(self, client_id: ClientId) -> ClientEvaluationArtifacts:
        for client in self.clients:
            if client.client_id == client_id:
                return client
        raise KeyError(client_id.value)


@dataclass(frozen=True, slots=True)
class ClientExperimentProvenance:
    client_id: ClientId
    provenance: ExperimentProvenance


@dataclass(frozen=True, slots=True)
class EvaluationProvenance:
    clients: tuple[ClientExperimentProvenance, ...]

    def __post_init__(self) -> None:
        if not self.clients:
            raise ValueError("evaluation provenance requires at least one client")
        client_ids = tuple(client.client_id for client in self.clients)
        if len(set(client_ids)) != len(client_ids):
            raise ValueError("evaluation provenance contains duplicate clients")

    def for_client(self, client_id: ClientId) -> ExperimentProvenance:
        for client in self.clients:
            if client.client_id == client_id:
                return client.provenance
        raise KeyError(client_id.value)

    def subset(self, population: ClientPopulation) -> EvaluationProvenance:
        return EvaluationProvenance(
            tuple(
                ClientExperimentProvenance(
                    client_id=client_id,
                    provenance=self.for_client(client_id),
                )
                for client_id in population.clients
            )
        )


@dataclass(frozen=True, slots=True)
class AttackSubtypeScores:
    subtype: AttackSubtypeId
    scores: ScoreVector


@dataclass(frozen=True, slots=True)
class SubtypeEvaluation:
    subtype: AttackSubtypeId
    attack_test_count: RowCount
    true_positive: RowCount
    false_negative: RowCount
    true_positive_rate: TruePositiveRate


@dataclass(frozen=True, slots=True)
class PolicyEvaluationResult:
    summary: CompletedPolicyEvaluation
    records: tuple[ClientResultRecord, ...]


def _scores(artifact: ScorePartitionArtifact) -> ScoreVector:
    return ScoreVector(
        np.fromiter(
            (record.score.value for record in artifact.records),
            dtype=np.float64,
            count=len(artifact.records),
        )
    )


def _attack_subtypes(artifact: ScorePartitionArtifact) -> tuple[AttackSubtypeScores, ...]:
    subtypes = tuple(
        sorted(
            {
                record.attack_subtype
                for record in artifact.records
                if record.attack_subtype is not None
            },
            key=lambda subtype: subtype.value,
        )
    )
    return tuple(
        AttackSubtypeScores(
            subtype=subtype,
            scores=ScoreVector(
                np.fromiter(
                    (
                        record.score.value
                        for record in artifact.records
                        if record.attack_subtype == subtype
                    ),
                    dtype=np.float64,
                )
            ),
        )
        for subtype in subtypes
    )


def evaluate_allocation(
    coordinate: AllocationCoordinate,
    allocation: Allocation,
    artifacts: FederationEvaluationArtifacts,
    weights: FederationWeights,
    solver: SolverEvidence,
    provenance: EvaluationProvenance,
    fallback_rate: Probability,
) -> PolicyEvaluationResult:
    if coordinate.policy is not allocation.policy:
        raise ValueError("allocation coordinate and allocation policy must match")

    allocation_clients = {decision.client_id for decision in allocation.decisions}
    artifact_clients = {client.client_id for client in artifacts.clients}
    weight_clients = {client.client_id for client in weights.clients}
    provenance_clients = {client.client_id for client in provenance.clients}
    if not (
        allocation_clients
        == artifact_clients
        == weight_clients
        == provenance_clients
    ):
        raise ValueError(
            "allocation, evaluation artifacts, weights, and provenance must cover one client set"
        )

    final_calibration = calibrate_final_thresholds(
        allocation,
        FinalCalibrationInputs(
            tuple(client.final_calibration for client in artifacts.clients)
        ),
    )

    client_macro_recalls: list[ClientMacroRecall] = []
    client_false_positive_rates: list[ClientFalsePositiveRate] = []
    records: list[ClientResultRecord] = []

    for decision in allocation.decisions:
        client_artifacts = artifacts.for_client(decision.client_id)
        calibration = final_calibration.for_client(decision.client_id)
        benign_scores = _scores(client_artifacts.benign_test)
        benign_alerts = alerts_above_threshold(benign_scores, calibration.threshold)
        false_positive_count = RowCount(int(np.count_nonzero(benign_alerts.values)))
        true_negative_count = RowCount(
            benign_scores.row_count.value - false_positive_count.value
        )
        client_fpr = FalsePositiveRate(
            0.0
            if benign_scores.row_count.value == 0
            else false_positive_count.value / benign_scores.row_count.value
        )
        client_false_positive_rates.append(
            ClientFalsePositiveRate(decision.client_id, client_fpr)
        )

        subtype_scores = _attack_subtypes(client_artifacts.attack_test)
        if not subtype_scores:
            raise ValueError(
                f"client {decision.client_id.value} has no ATTACK_TEST subtype rows"
            )

        subtype_recalls: list[SubtypeRecall] = []
        subtype_evaluations: list[SubtypeEvaluation] = []
        for subtype in subtype_scores:
            alerts = alerts_above_threshold(subtype.scores, calibration.threshold)
            true_positive = RowCount(int(np.count_nonzero(alerts.values)))
            false_negative = RowCount(
                subtype.scores.row_count.value - true_positive.value
            )
            true_positive_rate = TruePositiveRate(
                true_positive.value / subtype.scores.row_count.value
            )
            subtype_recalls.append(
                SubtypeRecall(subtype=subtype.subtype, rate=true_positive_rate)
            )
            subtype_evaluations.append(
                SubtypeEvaluation(
                    subtype=subtype.subtype,
                    attack_test_count=subtype.scores.row_count,
                    true_positive=true_positive,
                    false_negative=false_negative,
                    true_positive_rate=true_positive_rate,
                )
            )

        macro_recall = client_macro_recall(tuple(subtype_recalls))
        client_macro_recalls.append(
            ClientMacroRecall(decision.client_id, macro_recall)
        )
        client_weight = weights.for_client(decision.client_id)

        for subtype in subtype_evaluations:
            records.append(
                ClientResultRecord(
                    allocation=coordinate,
                    client_id=decision.client_id,
                    calibration=CalibrationOutcome(
                        target_rate=calibration.target_rate,
                        threshold=calibration.threshold,
                        calibration_count=calibration.calibration_count,
                    ),
                    weights=ClientWeights(
                        nominal=client_weight,
                        realized=client_weight,
                    ),
                    benign_test_count=benign_scores.row_count,
                    attack_test_count=subtype.attack_test_count,
                    attack_subtype=subtype.subtype,
                    confusion=ConfusionCounts(
                        true_positive=subtype.true_positive,
                        false_negative=subtype.false_negative,
                        false_positive=false_positive_count,
                        true_negative=true_negative_count,
                    ),
                    metrics=DetectionMetrics(
                        false_positive_rate=client_fpr,
                        true_positive_rate=subtype.true_positive_rate,
                        macro_attack_recall=macro_recall,
                        false_alert_count=FalseAlertCount(false_positive_count.value),
                    ),
                    solver=solver,
                    provenance=provenance.for_client(decision.client_id),
                )
            )

    federation_rate = federation_fpr(tuple(client_false_positive_rates), weights)
    return PolicyEvaluationResult(
        summary=CompletedPolicyEvaluation(
            policy=allocation.policy,
            macro_recall=federation_macro_recall(tuple(client_macro_recalls)),
            worst_client_recall=worst_client_recall(tuple(client_macro_recalls)),
            federation_fpr=federation_rate,
            budget_usage=budget_usage_ratio(
                federation_rate,
                coordinate.experiment.budget,
            ),
            fallback_rate=fallback_rate,
        ),
        records=tuple(records),
    )
