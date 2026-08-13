from __future__ import annotations

from dataclasses import dataclass

from fabrid.allocation.contracts import (
    ClientUtilityCurve,
    ClientUtilityCurves,
    ClientUtilityPoint,
)
from fabrid.calibration.order_statistic import calibrate_threshold
from fabrid.domain.enums import EligibilityStatus
from fabrid.domain.identifiers import AttackSubtypeId, ClientId
from fabrid.domain.population import ClientPopulation
from fabrid.domain.scores import ScoreVector
from fabrid.domain.values import (
    DetectionUtility,
    Probability,
    RowCount,
    TargetFalsePositiveRate,
    Threshold,
    TruePositiveRate,
)
from fabrid.protocol.models import UtilityEligibility


@dataclass(frozen=True, slots=True)
class SubtypeConfusionCounts:
    true_positive: RowCount
    false_negative: RowCount

    def __post_init__(self) -> None:
        if self.true_positive.value + self.false_negative.value == 0:
            raise ValueError("subtype confusion counts must cover at least one attack row")

    @property
    def attack_rows(self) -> RowCount:
        return RowCount(self.true_positive.value + self.false_negative.value)

    def true_positive_rate(self) -> TruePositiveRate:
        return TruePositiveRate(self.true_positive.value / self.attack_rows.value)


@dataclass(frozen=True, slots=True)
class SubtypeConfusion:
    subtype: AttackSubtypeId
    counts: SubtypeConfusionCounts


@dataclass(frozen=True, slots=True)
class CandidateConfusions:
    target_rate: TargetFalsePositiveRate
    subtypes: tuple[SubtypeConfusion, ...]

    def __post_init__(self) -> None:
        subtype_ids = tuple(subtype.subtype for subtype in self.subtypes)
        if len(set(subtype_ids)) != len(subtype_ids):
            raise ValueError("candidate confusion data contains duplicate subtypes")

    def for_subtype(self, subtype: AttackSubtypeId) -> SubtypeConfusionCounts:
        for item in self.subtypes:
            if item.subtype == subtype:
                return item.counts
        raise KeyError(subtype.value)


@dataclass(frozen=True, slots=True)
class AttackSubtypeSelection:
    subtypes: tuple[AttackSubtypeId, ...]

    def __post_init__(self) -> None:
        if len(set(self.subtypes)) != len(self.subtypes):
            raise ValueError("attack subtype selection contains duplicates")

    def contains(self, subtype: AttackSubtypeId) -> bool:
        return subtype in self.subtypes


@dataclass(frozen=True, slots=True)
class ClientFrontierInputs:
    client_id: ClientId
    benign_frontier_scores: ScoreVector
    candidates: tuple[CandidateConfusions, ...]

    def __post_init__(self) -> None:
        if not self.candidates:
            raise ValueError("client frontier inputs require candidate data")
        target_rates = tuple(candidate.target_rate.value for candidate in self.candidates)
        if target_rates != tuple(sorted(target_rates)):
            raise ValueError("frontier candidates must be sorted by target rate")
        if len(set(target_rates)) != len(target_rates):
            raise ValueError("frontier target rates must be unique")
        reference = self.candidates[0]
        reference_subtypes = tuple(subtype.subtype for subtype in reference.subtypes)
        reference_rows = tuple(subtype.counts.attack_rows for subtype in reference.subtypes)
        for candidate in self.candidates[1:]:
            candidate_subtypes = tuple(subtype.subtype for subtype in candidate.subtypes)
            candidate_rows = tuple(subtype.counts.attack_rows for subtype in candidate.subtypes)
            if candidate_subtypes != reference_subtypes or candidate_rows != reference_rows:
                raise ValueError(
                    "all frontier candidates must describe the same validation population"
                )

    def subtype_rows(self) -> tuple[tuple[AttackSubtypeId, RowCount], ...]:
        return tuple(
            (subtype.subtype, subtype.counts.attack_rows)
            for subtype in self.candidates[0].subtypes
        )


@dataclass(frozen=True, slots=True)
class FederationFrontierInputs:
    clients: tuple[ClientFrontierInputs, ...]

    def __post_init__(self) -> None:
        if not self.clients:
            raise ValueError("federation frontier inputs require at least one client")
        client_ids = tuple(client.client_id for client in self.clients)
        if len(set(client_ids)) != len(client_ids):
            raise ValueError("federation frontier inputs contain duplicate clients")


@dataclass(frozen=True, slots=True)
class CandidateThreshold:
    target_rate: TargetFalsePositiveRate
    threshold: Threshold


@dataclass(frozen=True, slots=True)
class EligibleClientFrontier:
    client_id: ClientId
    status: EligibilityStatus
    utility_curve: ClientUtilityCurve
    provisional_thresholds: tuple[CandidateThreshold, ...]

    def __post_init__(self) -> None:
        if self.status is not EligibilityStatus.ELIGIBLE:
            raise ValueError("eligible client frontier must have ELIGIBLE status")


@dataclass(frozen=True, slots=True)
class FallbackClientFrontier:
    client_id: ClientId
    status: EligibilityStatus

    def __post_init__(self) -> None:
        if self.status is not EligibilityStatus.FALLBACK:
            raise ValueError("fallback client frontier must have FALLBACK status")


ClientFrontier = EligibleClientFrontier | FallbackClientFrontier


@dataclass(frozen=True, slots=True)
class FederationFrontier:
    clients: tuple[ClientFrontier, ...]
    fallback_rate: Probability

    def eligible_curves(self) -> ClientUtilityCurves | None:
        curves = tuple(
            client.utility_curve
            for client in self.clients
            if isinstance(client, EligibleClientFrontier)
        )
        return None if not curves else ClientUtilityCurves(curves)

    def eligible_population(self) -> ClientPopulation | None:
        clients = tuple(
            client.client_id
            for client in self.clients
            if isinstance(client, EligibleClientFrontier)
        )
        return None if not clients else ClientPopulation(clients)


def eligible_subtypes(
    inputs: ClientFrontierInputs,
    guardrails: UtilityEligibility,
) -> AttackSubtypeSelection:
    return AttackSubtypeSelection(
        tuple(
            subtype
            for subtype, row_count in inputs.subtype_rows()
            if row_count.value >= guardrails.minimum_rows_per_subtype.value
        )
    )


def client_eligibility(
    inputs: ClientFrontierInputs,
    guardrails: UtilityEligibility,
) -> EligibilityStatus:
    total_rows = sum(row_count.value for _, row_count in inputs.subtype_rows())
    if total_rows < guardrails.minimum_attack_validation_rows.value:
        return EligibilityStatus.FALLBACK
    selection = eligible_subtypes(inputs, guardrails)
    if len(selection.subtypes) < guardrails.minimum_eligible_subtypes.value:
        return EligibilityStatus.FALLBACK
    return EligibilityStatus.ELIGIBLE


def _client_utility(
    candidate: CandidateConfusions,
    selection: AttackSubtypeSelection,
) -> DetectionUtility:
    rates = tuple(
        item.counts.true_positive_rate().value
        for item in candidate.subtypes
        if selection.contains(item.subtype)
    )
    if not rates:
        raise ValueError("client utility requires at least one eligible subtype")
    return DetectionUtility(sum(rates) / len(rates))


def build_client_frontier(
    inputs: ClientFrontierInputs,
    guardrails: UtilityEligibility,
) -> ClientFrontier:
    status = client_eligibility(inputs, guardrails)
    if status is EligibilityStatus.FALLBACK:
        return FallbackClientFrontier(inputs.client_id, status)

    selection = eligible_subtypes(inputs, guardrails)
    curve = ClientUtilityCurve(
        client_id=inputs.client_id,
        points=tuple(
            ClientUtilityPoint(
                target_rate=candidate.target_rate,
                utility=_client_utility(candidate, selection),
            )
            for candidate in inputs.candidates
        ),
    )
    thresholds = tuple(
        CandidateThreshold(
            target_rate=candidate.target_rate,
            threshold=calibrate_threshold(
                inputs.benign_frontier_scores,
                candidate.target_rate,
            ),
        )
        for candidate in inputs.candidates
    )
    return EligibleClientFrontier(
        client_id=inputs.client_id,
        status=status,
        utility_curve=curve,
        provisional_thresholds=thresholds,
    )


def build_federation_frontier(
    inputs: FederationFrontierInputs,
    guardrails: UtilityEligibility,
) -> FederationFrontier:
    clients = tuple(build_client_frontier(client, guardrails) for client in inputs.clients)
    fallback_count = sum(
        1 for client in clients if isinstance(client, FallbackClientFrontier)
    )
    return FederationFrontier(
        clients=clients,
        fallback_rate=Probability(fallback_count / len(clients)),
    )


def restrict_to_subtypes(
    inputs: ClientFrontierInputs,
    selection: AttackSubtypeSelection,
) -> ClientFrontierInputs:
    return ClientFrontierInputs(
        client_id=inputs.client_id,
        benign_frontier_scores=inputs.benign_frontier_scores,
        candidates=tuple(
            CandidateConfusions(
                target_rate=candidate.target_rate,
                subtypes=tuple(
                    subtype
                    for subtype in candidate.subtypes
                    if selection.contains(subtype.subtype)
                ),
            )
            for candidate in inputs.candidates
        ),
    )
