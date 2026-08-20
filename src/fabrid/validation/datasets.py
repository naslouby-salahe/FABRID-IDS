from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, model_validator

from fabrid.artifacts.json import digest_text, validated_digest
from fabrid.config import (
    ApplicationConfig,
    ArtifactDigest,
    AttackSplitConfig,
    AttackSubtypeId,
    BenignSplitConfig,
    ClientId,
    ColumnName,
    DatasetId,
    EnvironmentText,
    Identifier,
    NbaiotDatasetConfig,
    ProtocolVersion,
    RowCount,
)
from fabrid.datasets.registry import ClientPopulation, DeviceSplitPlan


class PreprocessingRelevantConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    dataset_id: DatasetId
    benign_splits: BenignSplitConfig
    attack_split: AttackSplitConfig
    feature_column_prefix: Identifier
    nbaiot: NbaiotDatasetConfig


def preprocessing_relevant_config(application: ApplicationConfig) -> PreprocessingRelevantConfig:
    return PreprocessingRelevantConfig(
        dataset_id=application.protocol.dataset_id,
        benign_splits=application.protocol.benign_splits,
        attack_split=application.protocol.attack_split,
        feature_column_prefix=application.preprocessing.feature_column_prefix,
        nbaiot=application.datasets.nbaiot,
    )


class PreprocessingProvenance(BaseModel):
    model_config = ConfigDict(frozen=True)
    dataset_id: DatasetId
    schema_version: ProtocolVersion
    raw_digest: ArtifactDigest
    preprocessing_config_digest: ArtifactDigest
    feature_manifest_digest: ArtifactDigest
    split_manifest_digest: ArtifactDigest
    features_digest: ArtifactDigest
    built_at: EnvironmentText

    @property
    def combined_digest(self) -> ArtifactDigest:
        return digest_text(
            (
                self.dataset_id.value,
                self.schema_version,
                self.raw_digest,
                self.preprocessing_config_digest,
                self.feature_manifest_digest,
                self.split_manifest_digest,
                self.features_digest,
            )
        )


def raw_dataset_directory_digest(root: Path) -> ArtifactDigest:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        with path.open("rb") as handle:
            while chunk := handle.read(1 << 20):
                digest.update(chunk)
    return validated_digest(digest.hexdigest())


@dataclass(frozen=True, slots=True)
class DatasetChecksum:
    dataset_id: DatasetId
    digest: ArtifactDigest


class FeatureManifest(BaseModel):
    model_config = ConfigDict(frozen=True)
    dataset_id: DatasetId
    columns: tuple[ColumnName, ...]

    @property
    def digest(self) -> ArtifactDigest:
        digest = hashlib.sha256()
        for column in self.columns:
            encoded = column.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
            digest.update(encoded)
        return validated_digest(digest.hexdigest())


class SplitAttackBoundaryManifest(BaseModel):
    model_config = ConfigDict(frozen=True)
    subtype: AttackSubtypeId
    validation_end: RowCount
    total_rows: RowCount

    @model_validator(mode="after")
    def _check_boundary_in_range(self) -> SplitAttackBoundaryManifest:
        if self.validation_end > self.total_rows:
            raise ValueError(
                "attack validation boundary exceeds the attack row total "
                f"({self.validation_end} > {self.total_rows})"
            )
        return self


class SplitClientManifest(BaseModel):
    model_config = ConfigDict(frozen=True)
    client_id: ClientId
    benign_train_end: RowCount
    benign_frontier_end: RowCount
    benign_final_cal_end: RowCount
    benign_total_rows: RowCount
    attacks: tuple[SplitAttackBoundaryManifest, ...]

    @model_validator(mode="after")
    def _check_boundaries(self) -> SplitClientManifest:
        boundaries = (
            self.benign_train_end,
            self.benign_frontier_end,
            self.benign_final_cal_end,
            self.benign_total_rows,
        )
        if any(right < left for left, right in zip(boundaries, boundaries[1:], strict=False)):
            raise ValueError(
                "benign split boundaries must be non-decreasing: train, frontier, "
                f"final-calibration, total ({boundaries})"
            )
        subtypes = tuple(attack.subtype for attack in self.attacks)
        if len(set(subtypes)) != len(subtypes):
            raise ValueError(f"client {self.client_id} declares duplicate attack subtypes")
        return self


class SplitManifest(BaseModel):
    model_config = ConfigDict(frozen=True)
    dataset_id: DatasetId
    clients: tuple[SplitClientManifest, ...]

    @model_validator(mode="after")
    def _check_unique_clients(self) -> SplitManifest:
        client_ids = tuple(client.client_id for client in self.clients)
        if len(set(client_ids)) != len(client_ids):
            raise ValueError("split manifest contains duplicate clients")
        return self

    @property
    def digest(self) -> ArtifactDigest:
        return validated_digest(hashlib.sha256(self.model_dump_json().encode("utf-8")).hexdigest())


def build_split_manifest(
    dataset_id: DatasetId,
    population: ClientPopulation,
    plans: tuple[DeviceSplitPlan, ...],
) -> SplitManifest:
    if len(population.clients) != len(plans):
        raise ValueError("split plan count must match the client population size")
    return SplitManifest(
        dataset_id=dataset_id,
        clients=tuple(
            SplitClientManifest(
                client_id=client_id,
                benign_train_end=plan.benign.train_end,
                benign_frontier_end=plan.benign.frontier_end,
                benign_final_cal_end=plan.benign.final_cal_end,
                benign_total_rows=plan.benign.total_rows,
                attacks=tuple(
                    SplitAttackBoundaryManifest(
                        subtype=item.subtype,
                        validation_end=item.boundary.validation_end,
                        total_rows=item.boundary.total_rows,
                    )
                    for item in plan.attacks
                ),
            )
            for client_id, plan in zip(population.clients, plans, strict=True)
        ),
    )
