"""Weight-heterogeneity sensitivity: w_k^(gamma) proportional to w_k^gamma, renormalized.

gamma=0 recovers equal-client weighting; gamma=1 recovers the original
reference weights; gamma>1 amplifies concentration, gamma<1 reduces it.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum

from fabrid.evaluation.record_level import ClientId


class WeightConcentration(float, Enum):
    """Preregistered gamma exponents for the weight-heterogeneity sweep."""

    EQUAL_CLIENT = 0.0
    REDUCED = 0.5
    ORIGINAL = 1.0
    AMPLIFIED = 1.5


def gamma_reweight(
    reference_weight: Mapping[ClientId, float], gamma: float
) -> dict[ClientId, float]:
    if not reference_weight:
        raise ValueError("gamma_reweight requires at least one client")
    if any(w <= 0 for w in reference_weight.values()):
        raise ValueError("all reference weights must be strictly positive")

    # w**0 == 1 for every positive w, so gamma=0 already yields equal shares
    # through the general formula below without a special case.
    powered = {client_id: w**gamma for client_id, w in reference_weight.items()}
    total = sum(powered.values())
    return {client_id: value / total for client_id, value in powered.items()}


def preregistered_gamma_sweep(
    reference_weight: Mapping[ClientId, float],
) -> dict[WeightConcentration, dict[ClientId, float]]:
    return {
        concentration: gamma_reweight(reference_weight, concentration.value)
        for concentration in WeightConcentration
    }
