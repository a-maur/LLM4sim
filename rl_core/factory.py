from __future__ import annotations

from typing import Any, Dict, Optional

from .distributions import (
    Distribution,
    DiscreteChoiceDistribution,
    GaussianDistribution,
    UniformDistribution,
    default_random_distribution,
)


def build_distribution(
    spec: Optional[Dict[str, Any]], default_dim: int = 1
) -> Distribution:
    """Build a distribution from config spec.

    Supported examples:
    - {"type": "gaussian", "mean": [0.0], "std": [1.0]}
    - {"type": "uniform", "low": [0.0], "high": [1.0]}
    - {"type": "choice", "choices": [0, 1, 2]}

    If spec is missing or invalid, fallback to random default.
    """
    if not spec:
        return default_random_distribution(dim=default_dim)

    try:
        kind = str(spec.get("type", "")).lower().strip()

        if kind == "gaussian":
            mean = spec.get("mean", [0.0] * default_dim)
            std = spec.get("std", [1.0] * default_dim)
            return GaussianDistribution(mean=mean, std=std)

        if kind == "uniform":
            low = spec.get("low", [-1.0] * default_dim)
            high = spec.get("high", [1.0] * default_dim)
            return UniformDistribution(low=low, high=high)

        if kind in {"choice", "discrete_choice"}:
            choices = spec.get("choices", [0.0, 1.0])
            return DiscreteChoiceDistribution(choices=choices)
    except Exception:
        pass

    return default_random_distribution(dim=default_dim)
