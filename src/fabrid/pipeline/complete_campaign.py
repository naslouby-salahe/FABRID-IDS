from __future__ import annotations

from dataclasses import dataclass

from fabrid.domain.identifiers import CampaignId
from fabrid.pipeline.campaign import FabridCampaign as CoreFabridCampaign
from fabrid.pipeline.campaign import run_fabrid_campaign as run_core_fabrid_campaign
from fabrid.pipeline.context import PipelinePaths
from fabrid.pipeline.stability_store import StoredCampaignAllocationStability
from fabrid.protocol.models import FabridProtocol
from fabrid.protocol.specification import PROTOCOL
