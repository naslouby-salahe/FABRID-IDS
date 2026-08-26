from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fabrid.artifacts.paths import ArtifactPaths
from fabrid.config import ApplicationConfig, DatasetId, ExperimentId, load_application_config
from fabrid.datasets.registry import resolve_raw_dataset_root


@dataclass(frozen=True, slots=True)
class ApplicationContext:
    config: ApplicationConfig
    paths: ArtifactPaths
    repository_root: Path

    def with_single_experiment(self, experiment_id: ExperimentId) -> ApplicationContext:
        return ApplicationContext(
            config=self.config.model_copy(
                update={"experiments": self.config.experiments.only(experiment_id)}
            ),
            paths=self.paths,
            repository_root=self.repository_root,
        )

    def raw_dataset_root(self, dataset_id: DatasetId) -> Path:
        return resolve_raw_dataset_root(self.paths.raw_data_root, dataset_id, self.config.datasets)


def load_application_context() -> ApplicationContext:
    repository_root = Path.cwd()
    config = load_application_config()
    return ApplicationContext(
        config=config,
        paths=ArtifactPaths.from_settings(config.paths),
        repository_root=repository_root,
    )
