from __future__ import annotations

from dataclasses import dataclass

from fabrid.domain.identifiers import ArtifactDigest, GitCommit


@dataclass(frozen=True, slots=True)
class DetectorProvenance:
    model: ArtifactDigest
    preprocessing: ArtifactDigest
    feature_manifest: ArtifactDigest


@dataclass(frozen=True, slots=True)
class ScoreProvenance:
    benign_frontier: ArtifactDigest
    benign_final_cal: ArtifactDigest
    benign_test: ArtifactDigest
    attack_validation: ArtifactDigest
    attack_test: ArtifactDigest
    split_manifest: ArtifactDigest
    protocol: ArtifactDigest


@dataclass(frozen=True, slots=True)
class ExperimentProvenance:
    detector: DetectorProvenance
    scores: ScoreProvenance
    git_commit: GitCommit
