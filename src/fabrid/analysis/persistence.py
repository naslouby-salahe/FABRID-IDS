from __future__ import annotations

from dataclasses import dataclass

from pydantic import TypeAdapter

from fabrid.analysis.gates import PracticalGateAnalysis
from fabrid.analysis.primary import PrimaryInference
from fabrid.artifacts.json_store import StoredJsonArtifact, write_typed_json
from fabrid.artifacts.layout import ArtifactLayout
from fabrid.domain.identifiers import ArtifactName, CampaignId

_PRIMARY_INFERENCE_NAME = ArtifactName("primary_inference")
_PRACTICAL_GATES_NAME = ArtifactName("practical_gates")
_PRIMARY_INFERENCE_ADAPTER = TypeAdapter(PrimaryInference)
_PRACTICAL_GATES_ADAPTER = TypeAdapter(PracticalGateAnalysis)


@dataclass(frozen=True, slots=True)
class StoredPrimaryInference:
    inference: PrimaryInference
    artifact: StoredJsonArtifact


@dataclass(frozen=True, slots=True)
class StoredPracticalGates:
    analysis: PracticalGateAnalysis
    artifact: StoredJsonArtifact


def persist_primary_inference(
    campaign_id: CampaignId,
    inference: PrimaryInference,
    layout: ArtifactLayout,
) -> StoredPrimaryInference:
    return StoredPrimaryInference(
        inference=inference,
        artifact=write_typed_json(
            inference,
            _PRIMARY_INFERENCE_ADAPTER,
            layout.campaign_analysis_path(campaign_id, _PRIMARY_INFERENCE_NAME),
        ),
    )


def persist_practical_gates(
    campaign_id: CampaignId,
    analysis: PracticalGateAnalysis,
    layout: ArtifactLayout,
) -> StoredPracticalGates:
    return StoredPracticalGates(
        analysis=analysis,
        artifact=write_typed_json(
            analysis,
            _PRACTICAL_GATES_ADAPTER,
            layout.campaign_analysis_path(campaign_id, _PRACTICAL_GATES_NAME),
        ),
    )
