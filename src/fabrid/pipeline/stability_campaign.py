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


def run_campaign_allocation_stability(
    campaign_id: CampaignId,
    paths: PipelinePaths,
    protocol: FabridProtocol = PROTOCOL,
) -> StoredCampaignAllocationStability:
    cells: list[SeedBudgetAllocationStability] = []
    for detector_seed in protocol.detector.seeds:
        scores = load_seed_scores(
            campaign_id=campaign_id,
            detector_seed=detector_seed,
            population=NBAIOT_PRIMARY_POPULATION,
            paths=paths,
        )
