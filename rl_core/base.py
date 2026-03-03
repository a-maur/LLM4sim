from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


@dataclass
class StepResult:
    next_state: Dict[str, Any]
    reward: float
    done: bool
    truncated: bool = False
    info: Dict[str, Any] = field(default_factory=dict)


class BaseSimulation(ABC):
    """Problem-specific transition logic.

    This class must only encode causality and state transition rules.
    Stochastic inputs should come through dependency injection.
    """

    @abstractmethod
    def initial_state(self) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def transition(
        self, state: Dict[str, Any], action: Any, step_idx: int
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        raise NotImplementedError


class BaseEnvironment(ABC):
    """Minimal environment interface, similar to Gym APIs."""

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def step(self, action: Any) -> StepResult:
        raise NotImplementedError

    def render(self) -> Any:
        """Optional renderer for human/debug visualization.

        Child environments can override this to provide custom rendering.
        """
        return None
