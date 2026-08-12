from __future__ import annotations

import pytest

from fabrid.config.attack_folds import (
    AttackFold,
    BotnetFamily,
    BotnetFamilyDirection,
    FoldId,
    FoldRotation,
    load_attack_folds,
)
from fabrid.evaluation.record_level import AttackSubtype


def test_load_attack_folds_reads_frozen_yaml() -> None:
    config = load_attack_folds()

    assert len(config.folds) == 3
    all_subtypes = {subtype for fold in config.folds for subtype in fold.subtypes}
    assert len(all_subtypes) == 10
    assert len(config.rotations) == 3
    assert config.botnet_family_disjoint.eligible_client_count == 7
    assert len(config.botnet_family_disjoint.directions) == 2


def test_rotations_are_disjoint_from_their_own_validation_fold() -> None:
    config = load_attack_folds()
    for rotation in config.rotations:
        assert rotation.validation_fold not in rotation.test_folds


def test_fold_subtypes_lookup() -> None:
    config = load_attack_folds()
    rotation = config.rotations[0]
    validation_subtypes = config.validation_subtypes(rotation)
    test_subtypes = config.test_subtypes(rotation)

    assert set(validation_subtypes).isdisjoint(set(test_subtypes))
    assert set(validation_subtypes) | set(test_subtypes) == {
        s for fold in config.folds for s in fold.subtypes
    }


def test_fold_subtypes_unknown_fold_raises() -> None:
    config = load_attack_folds()
    with pytest.raises(KeyError):
        config.fold_subtypes(FoldId(999))


def test_attack_fold_requires_at_least_one_subtype() -> None:
    with pytest.raises(ValueError, match="at least one attack subtype"):
        AttackFold(fold_id=FoldId(0), subtypes=())


def test_fold_rotation_rejects_validation_fold_in_test_folds() -> None:
    with pytest.raises(ValueError, match="must not also be a test fold"):
        FoldRotation(validation_fold=FoldId(0), test_folds=(FoldId(0), FoldId(1)))


def test_botnet_family_direction_rejects_identical_families() -> None:
    with pytest.raises(ValueError, match="must differ"):
        BotnetFamilyDirection(validation_family=BotnetFamily.MIRAI, test_family=BotnetFamily.MIRAI)


def test_attack_subtype_type_alias_roundtrip() -> None:
    config = load_attack_folds()
    subtype = config.folds[0].subtypes[0]
    assert isinstance(subtype, str)
    assert AttackSubtype(subtype) == subtype
