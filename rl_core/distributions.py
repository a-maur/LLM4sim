from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import random


class Distribution(ABC):
    """Abstract distribution used by simulations."""

    @abstractmethod
    def sample(self):
        raise NotImplementedError


@dataclass
class GaussianDistribution(Distribution):
    mean: Sequence[float]
    std: Sequence[float]

    def sample(self) -> List[float]:
        return [random.gauss(m, s) for m, s in zip(self.mean, self.std)]


@dataclass
class UniformDistribution(Distribution):
    low: Sequence[float]
    high: Sequence[float]

    def sample(self) -> List[float]:
        return [random.uniform(a, b) for a, b in zip(self.low, self.high)]


@dataclass
class DiscreteChoiceDistribution(Distribution):
    choices: Sequence[float]

    def sample(self) -> float:
        return random.choice(list(self.choices))


def default_random_distribution(dim: int = 1) -> Distribution:
    """Default placeholder distribution when config is absent.

    Uses uniform in [-1, 1] for each dimension.
    """
    return UniformDistribution(low=[-1.0] * dim, high=[1.0] * dim)
