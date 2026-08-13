from __future__ import annotations

from dataclasses import dataclass

from fabrid.domain.identifiers import CampaignId
from fabrid.pipeline.context import PipelinePaths
from fabrid.pipeline.stability_campaign import run_campaign_allocation_stability
from fabrid.pipeline.stability_store import StoredCampaignAllocationStability
from fabrid.protocol.models import FabridProtocol


@dataclass(frozen=True, slots=True)
class MandatoryCampaignEvidence:
    allocation_stability: StoredCampaignAllocationStability


def build_mandatory_campaign_evidence(
    campaign_id: CampaignId,
    paths: PipelinePaths,
    protocol: FabridProtocol,
) -> MandatoryCampaignEvidence:
    return MandatoryCampaignEvidence(
        allocation_stability=run_campaign_allocation_stability(
            campaign_id=campaign_id,
            paths=paths,
            protocol=protocol,
        )
    )
