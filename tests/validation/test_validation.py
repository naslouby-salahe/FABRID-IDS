from __future__ import annotations

import pytest

from fabrid.allocation.contracts import (
    Allocation,
    AllocationDecision,
    AllocationWeights,
    ClientBudgetWeight,
)
from fabrid.domain.enums import AllocationPolicy, BenignSplit
from fabrid.domain.identifiers import ArtifactDigest, ClientId, SampleId
from fabrid.domain.population import ClientPopulation
from fabrid.domain.values import (
    ClientWeight,
    FalsePositiveBudget,
    Probability,
    RowCount,
    TargetFalsePositiveRate,
)
from fabrid.evaluation.ranking import PolicyRankingEvidence
from fabrid.validation.budget import (
    BudgetValidationError,
    validate_budget_feasibility,
    validate_population_coverage,
)
from fabrid.validation.claims import PublicationText, validate_no_forbidden_claims
from fabrid.validation.determinism import DeterminismValidationError, validate_determinism
from fabrid.validation.leakage import (
    PartitionExclusivityValidationError,
    PartitionSamples,
    validate_partition_exclusivity,
)
from fabrid.validation.scores import (
    PolicyScoreDigest,
    ScoreIdentityValidationError,
    validate_auroc_identity,
    validate_score_digest_identity,
)


def test_budget_and_population_validation_use_typed_allocation_contracts() -> None:
    first = ClientId("first")
    second = ClientId("second")
    allocation = Allocation(
        policy=AllocationPolicy.EQ_FPR,
        decisions=(
            AllocationDecision(first, TargetFalsePositiveRate(0.01)),
            AllocationDecision(second, TargetFalsePositiveRate(0.01)),
        ),
    )
    weights = AllocationWeights(
        (
            ClientBudgetWeight(first, ClientWeight(0.5)),
            ClientBudgetWeight(second, ClientWeight(0.5)),
        )
    )

    validate_budget_feasibility(allocation, weights, FalsePositiveBudget(0.01))
    validate_population_coverage(allocation, ClientPopulation((first, second)))

    with pytest.raises(BudgetValidationError):
        validate_budget_feasibility(allocation, weights, FalsePositiveBudget(0.005))
    with pytest.raises(BudgetValidationError):
        validate_population_coverage(allocation, ClientPopulation((first,)))


def test_determinism_validation_rejects_non_repeatable_computation() -> None:
    counter = RowCount(0)

    def stable() -> TargetFalsePositiveRate:
        return TargetFalsePositiveRate(0.01)

    validate_determinism(stable, RowCount(3))

    values = iter((TargetFalsePositiveRate(0.01), TargetFalsePositiveRate(0.02)))
    with pytest.raises(DeterminismValidationError):
        validate_determinism(lambda: next(values), RowCount(2))

    assert counter == RowCount(0)


def test_score_digest_and_auroc_identity_require_same_frozen_scores() -> None:
    digest = ArtifactDigest("a" * 64)
    validate_score_digest_identity(
        (
            PolicyScoreDigest(AllocationPolicy.EQ_FPR, digest),
            PolicyScoreDigest(AllocationPolicy.GREEDY, digest),
        )
    )
    validate_auroc_identity(
        (
            PolicyRankingEvidence(AllocationPolicy.EQ_FPR, Probability(0.8), Probability(0.7)),
            PolicyRankingEvidence(AllocationPolicy.GREEDY, Probability(0.8), Probability(0.7)),
        )
    )

    with pytest.raises(ScoreIdentityValidationError):
        validate_score_digest_identity(
            (
                PolicyScoreDigest(AllocationPolicy.EQ_FPR, ArtifactDigest("a" * 64)),
                PolicyScoreDigest(AllocationPolicy.GREEDY, ArtifactDigest("b" * 64)),
            )
        )


def test_partition_exclusivity_rejects_cross_split_sample_reuse() -> None:
    first = SampleId("first")
    second = SampleId("second")
    validate_partition_exclusivity(
        (
            PartitionSamples(BenignSplit.TRAIN, (first,)),
            PartitionSamples(BenignSplit.TEST, (second,)),
        )
    )

    with pytest.raises(PartitionExclusivityValidationError):
        validate_partition_exclusivity(
            (
                PartitionSamples(BenignSplit.TRAIN, (first,)),
                PartitionSamples(BenignSplit.TEST, (first,)),
            )
        )


def test_forbidden_claim_guard_uses_finite_enum_vocabulary() -> None:
    validate_no_forbidden_claims(PublicationText("FABRID allocates a false-positive budget."))

    with pytest.raises(ValueError):
        validate_no_forbidden_claims(
            PublicationText("The method is a novel MILP optimizer for the IDS.")
        )
