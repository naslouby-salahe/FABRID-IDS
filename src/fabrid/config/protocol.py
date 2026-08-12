"""Typed loader for the frozen protocol artifact (`protocol.yaml`).

`protocol.yaml` is the single canonical source for every protocol constant
(split fractions, budgets, solver settings, ...). Other modules must load
their constants through this module rather than re-declaring literal values.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

PROTOCOL_PATH = Path(__file__).with_name("protocol.yaml")


@dataclass(frozen=True, slots=True)
class BenignSplitFractions:
    """Cumulative source-order boundary fractions for the benign partitioner."""

    train_end_fraction: float
    frontier_end_fraction: float
    final_cal_end_fraction: float

    def __post_init__(self) -> None:
        if not (
            0.0
            <= self.train_end_fraction
            <= self.frontier_end_fraction
            <= self.final_cal_end_fraction
            <= 1.0
        ):
            raise ValueError(
                "benign split fractions must satisfy "
                "0 <= train_end <= frontier_end <= final_cal_end <= 1, got "
                f"{self.train_end_fraction}, {self.frontier_end_fraction}, "
                f"{self.final_cal_end_fraction}"
            )


@dataclass(frozen=True, slots=True)
class AttackSplitFraction:
    """Source-order boundary fraction for the attack-row partitioner."""

    validation_end_fraction: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.validation_end_fraction <= 1.0):
            raise ValueError(
                f"validation_end_fraction must be in [0, 1], got {self.validation_end_fraction}"
            )


@dataclass(frozen=True, slots=True)
class UtilityEligibilityGuardrails:
    """Preregistered data-sufficiency guardrails for validation-informed utility estimation."""

    min_attack_validation_rows: int
    min_eligible_subtypes: int
    min_rows_per_eligible_subtype: int

    def __post_init__(self) -> None:
        if self.min_attack_validation_rows < 0:
            raise ValueError(
                f"min_attack_validation_rows must be non-negative, got "
                f"{self.min_attack_validation_rows}"
            )
        if self.min_eligible_subtypes < 0:
            raise ValueError(
                f"min_eligible_subtypes must be non-negative, got {self.min_eligible_subtypes}"
            )
        if self.min_rows_per_eligible_subtype < 0:
            raise ValueError(
                f"min_rows_per_eligible_subtype must be non-negative, got "
                f"{self.min_rows_per_eligible_subtype}"
            )


@dataclass(frozen=True, slots=True)
class SolverSettings:
    """MILP solver configuration and the strict solve-acceptance rule."""

    mip_rel_gap: float
    time_limit_seconds: float
    accept_mip_gap_leq: float

    def __post_init__(self) -> None:
        if self.mip_rel_gap < 0:
            raise ValueError(f"mip_rel_gap must be non-negative, got {self.mip_rel_gap}")
        if self.time_limit_seconds <= 0:
            raise ValueError(f"time_limit_seconds must be positive, got {self.time_limit_seconds}")
        if self.accept_mip_gap_leq < 0:
            raise ValueError(
                f"accept_mip_gap_leq must be non-negative, got {self.accept_mip_gap_leq}"
            )


@dataclass(frozen=True, slots=True)
class Protocol:
    """Typed view over the constants this codebase currently consumes from `protocol.yaml`.

    Extend as additional protocol sections gain typed consumers; do not read
    `protocol.yaml` ad hoc elsewhere.
    """

    benign_split_fractions: BenignSplitFractions
    attack_split_fraction: AttackSplitFraction
    alpha_max: float
    solver_settings: SolverSettings
    utility_eligibility: UtilityEligibilityGuardrails


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload: Any = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a mapping at the top level of {path}")
    return cast(dict[str, Any], payload)


def load_protocol(path: Path = PROTOCOL_PATH) -> Protocol:
    payload = _read_yaml(path)

    benign_fractions_raw = payload["benign_split_fractions"]
    benign_split_fractions = BenignSplitFractions(
        train_end_fraction=float(benign_fractions_raw["train"]),
        frontier_end_fraction=float(benign_fractions_raw["frontier"]),
        final_cal_end_fraction=float(benign_fractions_raw["final_cal"]),
    )

    attack_split_raw = payload["attack_split"]
    attack_split_fraction = AttackSplitFraction(
        validation_end_fraction=float(attack_split_raw["validation_fraction"]),
    )

    alpha_max = float(payload["alpha_grid"]["alpha_max"])

    solver_raw = payload["solver"]
    solver_settings = SolverSettings(
        mip_rel_gap=float(solver_raw["mip_rel_gap"]),
        time_limit_seconds=float(solver_raw["time_limit_seconds"]),
        accept_mip_gap_leq=float(solver_raw["accept_if"]["mip_gap_leq"]),
    )

    eligibility_raw = payload["utility_eligibility"]
    utility_eligibility = UtilityEligibilityGuardrails(
        min_attack_validation_rows=int(eligibility_raw["min_attack_validation_rows"]),
        min_eligible_subtypes=int(eligibility_raw["min_eligible_subtypes"]),
        min_rows_per_eligible_subtype=int(eligibility_raw["min_rows_per_eligible_subtype"]),
    )

    return Protocol(
        benign_split_fractions=benign_split_fractions,
        attack_split_fraction=attack_split_fraction,
        utility_eligibility=utility_eligibility,
        alpha_max=alpha_max,
        solver_settings=solver_settings,
    )
