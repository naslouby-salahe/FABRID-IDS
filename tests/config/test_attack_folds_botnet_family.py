from __future__ import annotations

from fabrid.config.attack_folds import BotnetFamily, load_botnet_family_subtypes


def test_load_botnet_family_subtypes_covers_all_ten_subtypes() -> None:
    family_subtypes = load_botnet_family_subtypes()

    assert set(family_subtypes) == {BotnetFamily.BASHLITE, BotnetFamily.MIRAI}
    assert len(family_subtypes[BotnetFamily.BASHLITE]) == 5
    assert len(family_subtypes[BotnetFamily.MIRAI]) == 5
    all_subtypes = set(family_subtypes[BotnetFamily.BASHLITE]) | set(
        family_subtypes[BotnetFamily.MIRAI]
    )
    assert len(all_subtypes) == 10


def test_family_subtypes_are_disjoint() -> None:
    family_subtypes = load_botnet_family_subtypes()
    assert set(family_subtypes[BotnetFamily.BASHLITE]).isdisjoint(
        set(family_subtypes[BotnetFamily.MIRAI])
    )
