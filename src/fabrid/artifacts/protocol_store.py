from __future__ import annotations

from pydantic import TypeAdapter

from fabrid.artifacts.json_store import StoredJsonArtifact, write_typed_json
from fabrid.domain.identifiers import CampaignId
from fabrid.pipeline.context import PipelinePaths
from fabrid.protocol.models import FabridProtocol

_PROTOCOL_ADAPTER = TypeAdapter(FabridProtocol)


def persist_protocol_snapshot(
    campaign_id: CampaignId,
    protocol: FabridProtocol,
    paths: PipelinePaths,
) -> StoredJsonArtifact:
    return write_typed_json(
        protocol,
        _PROTOCOL_ADAPTER,
        paths.artifacts.protocol_path(campaign_id),
    )
