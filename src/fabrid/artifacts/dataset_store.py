from __future__ import annotations

from dataclasses import dataclass

from pydantic import TypeAdapter

from fabrid.artifacts.json_store import StoredJsonArtifact, write_typed_json
from fabrid.datasets.manifests import DatasetSplitManifest, FeatureManifest
from fabrid.domain.enums import DatasetId
from fabrid.domain.identifiers import CampaignId
from fabrid.pipeline.context import PipelinePaths

_FEATURE_MANIFEST_ADAPTER = TypeAdapter(FeatureManifest)
_SPLIT_MANIFEST_ADAPTER = TypeAdapter(DatasetSplitManifest)


@dataclass(frozen=True, slots=True)
class StoredDatasetManifests:
    features: StoredJsonArtifact
    splits: StoredJsonArtifact


def persist_dataset_manifests(
    campaign_id: CampaignId,
    dataset_id: DatasetId,
    feature_manifest: FeatureManifest,
    split_manifest: DatasetSplitManifest,
    paths: PipelinePaths,
) -> StoredDatasetManifests:
    if split_manifest.dataset_id is not dataset_id:
        raise ValueError("split manifest dataset id does not match requested dataset")
    layout = paths.artifacts
    return StoredDatasetManifests(
        features=write_typed_json(
            feature_manifest,
            _FEATURE_MANIFEST_ADAPTER,
            layout.feature_manifest_path(campaign_id, dataset_id),
        ),
        splits=write_typed_json(
            split_manifest,
            _SPLIT_MANIFEST_ADAPTER,
            layout.split_manifest_path(campaign_id, dataset_id),
        ),
    )
