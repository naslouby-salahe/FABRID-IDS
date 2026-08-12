"""Typed loader for the frozen attack-subtype-disjoint / botnet-family-disjoint generalization
fold assignment (`attack_folds.yaml`). The fold->subtype mapping is fixed globally, not derived
by hashing, so it must be read from this frozen artifact rather than computed ad hoc.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import NewType

from fabrid.config.protocol import read_yaml_mapping
from fabrid.evaluation.record_level import AttackSubtype

ATTACK_FOLDS_PATH = Path(__file__).with_name("attack_folds.yaml")
DATASETS_PATH = Path(__file__).with_name("datasets.yaml")

FoldId = NewType("FoldId", int)


class BotnetFamily(StrEnum):
    BASHLITE = "bashlite"
    MIRAI = "mirai"


@dataclass(frozen=True, slots=True)
class AttackFold:
    fold_id: FoldId
    subtypes: tuple[AttackSubtype, ...]

    def __post_init__(self) -> None:
        if not self.subtypes:
            raise ValueError(f"fold {self.fold_id} must contain at least one attack subtype")


@dataclass(frozen=True, slots=True)
class FoldRotation:
    validation_fold: FoldId
    test_folds: tuple[FoldId, ...]

    def __post_init__(self) -> None:
        if self.validation_fold in self.test_folds:
            raise ValueError(f"validation_fold {self.validation_fold} must not also be a test fold")
        if not self.test_folds:
            raise ValueError("test_folds must contain at least one fold")


@dataclass(frozen=True, slots=True)
class BotnetFamilyDirection:
    validation_family: BotnetFamily
    test_family: BotnetFamily

    def __post_init__(self) -> None:
        if self.validation_family == self.test_family:
            raise ValueError(
                f"validation_family and test_family must differ, both are {self.validation_family}"
            )


@dataclass(frozen=True, slots=True)
class BotnetFamilyDisjointConfig:
    eligible_client_count: int
    directions: tuple[BotnetFamilyDirection, ...]

    def __post_init__(self) -> None:
        if self.eligible_client_count <= 0:
            raise ValueError(
                f"eligible_client_count must be positive, got {self.eligible_client_count}"
            )
        if not self.directions:
            raise ValueError("directions must contain at least one direction")


@dataclass(frozen=True, slots=True)
class AttackFoldsConfig:
    folds: tuple[AttackFold, ...]
    rotations: tuple[FoldRotation, ...]
    botnet_family_disjoint: BotnetFamilyDisjointConfig

    def fold_subtypes(self, fold_id: FoldId) -> tuple[AttackSubtype, ...]:
        for fold in self.folds:
            if fold.fold_id == fold_id:
                return fold.subtypes
        raise KeyError(f"no fold with id {fold_id}")

    def validation_subtypes(self, rotation: FoldRotation) -> tuple[AttackSubtype, ...]:
        return self.fold_subtypes(rotation.validation_fold)

    def test_subtypes(self, rotation: FoldRotation) -> tuple[AttackSubtype, ...]:
        return tuple(
            subtype
            for test_fold in rotation.test_folds
            for subtype in self.fold_subtypes(test_fold)
        )


def load_attack_folds(path: Path = ATTACK_FOLDS_PATH) -> AttackFoldsConfig:
    payload = read_yaml_mapping(path)

    folds = tuple(
        AttackFold(
            fold_id=FoldId(int(fold_id)),
            subtypes=tuple(AttackSubtype(s) for s in subtypes),
        )
        for fold_id, subtypes in payload["folds"].items()
    )

    rotations = tuple(
        FoldRotation(
            validation_fold=FoldId(int(rotation["validation_fold"])),
            test_folds=tuple(FoldId(int(f)) for f in rotation["test_folds"]),
        )
        for rotation in payload["rotations"].values()
    )

    family_raw = payload["botnet_family_disjoint"]
    directions = tuple(
        BotnetFamilyDirection(
            validation_family=BotnetFamily(direction["validation_family"]),
            test_family=BotnetFamily(direction["test_family"]),
        )
        for direction in family_raw["directions"].values()
    )
    botnet_family_disjoint = BotnetFamilyDisjointConfig(
        eligible_client_count=int(family_raw["eligible_client_count"]),
        directions=directions,
    )

    return AttackFoldsConfig(
        folds=folds,
        rotations=rotations,
        botnet_family_disjoint=botnet_family_disjoint,
    )


def load_botnet_family_subtypes(
    path: Path = DATASETS_PATH,
) -> dict[BotnetFamily, tuple[AttackSubtype, ...]]:
    """Which attack subtypes belong to each botnet family, per `datasets.yaml`'s
    `nbaiot.attack_subtypes` section.
    """
    payload = read_yaml_mapping(path)
    families_raw = payload["nbaiot"]["attack_subtypes"]
    return {
        BotnetFamily(family): tuple(AttackSubtype(s) for s in subtypes)
        for family, subtypes in families_raw.items()
    }
