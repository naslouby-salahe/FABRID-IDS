from __future__ import annotations

from dataclasses import dataclass

from pydantic import TypeAdapter

from fabrid.artifacts.json_store import StoredJsonArtifact, write_typed_json
from fabrid.artifacts.layout import ArtifactLayout
from fabrid.domain.identifiers import ArtifactName, CampaignId
from fabrid.pipeline.stability import CampaignAllocationStability

_ALLOCATION_STABILITY_NAME = ArtifactName("allocation_stability")
_ALLOCATION_STABILITY_ADAPTER = TypeAdapter(CampaignAllocationStability)


@dataclass(frozen=True, slots=True)
class StoredCampaignAllocationStability:
    analysis: CampaignAllocationStability
    artifact: StoredJsonArtifact


def persist_campaign_allocation_stability(
    campaign_id: CampaignId,
    analysis: CampaignAllocationStability,
    layout: ArtifactLayout,
) -> StoredCampaignAllocationStability:
    return StoredCampaignAllocationStability(
        analysis=analysis,
        artifact=write_typed_json(
            analysis,
            _ALLOCATION_STABILITY_ADAPTER,
            layout.campaign_analysis_path(campaign_id, _ALLOCATION_STABILITY_NAME),
        ),
    )
