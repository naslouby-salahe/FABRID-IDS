from __future__ import annotations

from pydantic import TypeAdapter

from fabrid.artifacts.json_store import StoredJsonArtifact, write_typed_json
from fabrid.evaluation.results import SeedBudgetEvaluation
from fabrid.pipeline.context import PipelinePaths

_EVALUATION_ADAPTER = TypeAdapter(SeedBudgetEvaluation)


def persist_seed_budget_evaluation(
    evaluation: SeedBudgetEvaluation,
    paths: PipelinePaths,
) -> StoredJsonArtifact:
    return write_typed_json(
        evaluation,
        _EVALUATION_ADAPTER,
        paths.artifacts.evaluation_summary_path(evaluation.experiment),
    )
