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
    score: ArtifactDigest
    split_manifest: ArtifactDigest
    protocol: ArtifactDigest


@dataclass(frozen=True, slots=True)
class ExperimentProvenance:
    detector: DetectorProvenance
    score: ScoreProvenance
    git_commit: GitCommit
