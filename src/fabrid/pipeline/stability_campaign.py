from __future__ import annotations

from fabrid.datasets.nbaiot.specification import NBAIOT_PRIMARY_POPULATION
from fabrid.domain.identifiers import CampaignId
from fabrid.pipeline.allocation import load_seed_scores
from fabrid.pipeline.context import PipelinePaths
from fabrid.pipeline.stability import (
    CampaignAllocationStability,
    SeedBudgetAllocationStability,
    run_seed_budget_allocation_stability,
)
from fabrid.pipeline.stability_store import (
    StoredCampaignAllocationStability,
    persist_campaign_allocation_stability,
)
from fabrid.protocol.models import FabridProtocol
from fabrid.protocol.specification import PROTOCOL
