from __future__ import annotations

from pydantic import TypeAdapter

from fabrid.artifacts.json_store import StoredJsonArtifact, write_typed_json
from fabrid.artifacts.layout import ArtifactLayout
from fabrid.evaluation.results import SeedBudgetEvaluation

_EVALUATION_ADAPTER = TypeAdapter(SeedBudgetEvaluation)


def persist_seed_budget_evaluation(
    evaluation: SeedBudgetEvaluation,
    layout: ArtifactLayout,
) -> StoredJsonArtifact:
    return write_typed_json(
        evaluation,
        _EVALUATION_ADAPTER,
        layout.evaluation_summary_path(evaluation.experiment),
    )
