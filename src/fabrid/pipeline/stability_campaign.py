from __future__ import annotations

from fabrid.datasets.nbaiot.specification import NBAIOT_PRIMARY_POPULATION
from fabrid.domain.identifiers import CampaignId
from fabrid.pipeline.allocation import load_seed_scores
from fabrid.pipeline.context import PipelinePaths
from fabrid.pipeline.stability import CampaignAllocationStability, SeedBudgetAllocationStability, run_seed_budget_allocation_stability
from fabrid.pipeline.stability_store import StoredCampaignAllocationStability, persist_campaign_allocation_stability
from fabrid.protocol.models import FabridProtocol


def run_campaign_allocation_stability(
    campaign_id: CampaignId, paths: PipelinePaths, protocol: FabridProtocol
) -> StoredCampaignAllocationStability:
    cells: list[SeedBudgetAllocationStability] = []
    for seed in protocol.detector.seeds:
        scores = load_seed_scores(campaign_id, seed, NBAIOT_PRIMARY_POPULATION, paths)
        cells.extend(
            run_seed_budget_allocation_stability(campaign_id, seed, budget, scores, protocol)
            for budget in protocol.budgets
        )
    return persist_campaign_allocation_stability(
        campaign_id, CampaignAllocationStability(tuple(cells)), paths.artifacts
    )
