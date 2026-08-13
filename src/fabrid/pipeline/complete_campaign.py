from __future__ import annotations

from dataclasses import dataclass

from fabrid.domain.identifiers import CampaignId
from fabrid.pipeline.campaign import FabridCampaign as CoreFabridCampaign
from fabrid.pipeline.campaign import run_fabrid_campaign as run_core_fabrid_campaign
from fabrid.pipeline.context import PipelinePaths
from fabrid.pipeline.mandatory_evidence import (
    MandatoryCampaignEvidence,
    build_mandatory_campaign_evidence,
)
from fabrid.protocol.models import FabridProtocol
from fabrid.protocol.specification import PROTOCOL


@dataclass(frozen=True, slots=True)
class FabridCampaign:
    execution: CoreFabridCampaign
    evidence: MandatoryCampaignEvidence


def run_fabrid_campaign(
    campaign_id: CampaignId,
    paths: PipelinePaths,
    protocol: FabridProtocol = PROTOCOL,
) -> FabridCampaign:
    execution = run_core_fabrid_campaign(campaign_id, paths, protocol)
    evidence = build_mandatory_campaign_evidence(campaign_id, paths, protocol)
    return FabridCampaign(execution=execution, evidence=evidence)
