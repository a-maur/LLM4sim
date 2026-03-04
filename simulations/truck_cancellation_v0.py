from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from rl_core.base import BaseEnvironment, BaseSimulation, StepResult


@dataclass
class TruckCancellationParams:
    n_destinations: int
    time_steps_per_day: int
    decision_step: int
    truck_capacity: int
    trucks_per_day_by_destination: List[int]
    season: str
    day_of_week: int
    destination_base_mean_high: List[float]
    destination_base_mean_low: List[float]
    day_of_week_weights: List[float]
    season_weights: Dict[str, float]
    shared_shock_std: float
    destination_shock_std: float
    registration_curve: List[float]
    cancellation_saving: float
    penalty_unshipped_high: float
    penalty_unshipped_low: float


class TruckCancellationV0Simulation(BaseSimulation):
    """Single-day simulation for route-level last-truck cancellation decisions."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = (config or {}).get("params", {})
        self.params = TruckCancellationParams(
            n_destinations=int(cfg.get("n_destinations", 4)),
            time_steps_per_day=int(cfg.get("time_steps_per_day", 12)),
            decision_step=int(cfg.get("decision_step", 8)),
            truck_capacity=int(cfg.get("truck_capacity", 120)),
            trucks_per_day_by_destination=list(cfg.get("trucks_per_day_by_destination", [3, 2, 3, 2])),
            season=str(cfg.get("season", "high")),
            day_of_week=int(cfg.get("day_of_week", 1)),
            destination_base_mean_high=list(cfg.get("destination_base_mean_high", [95, 70, 90, 65])),
            destination_base_mean_low=list(cfg.get("destination_base_mean_low", [180, 130, 160, 120])),
            day_of_week_weights=list(cfg.get("day_of_week_weights", [0.9, 1.0, 1.05, 1.1, 1.15, 0.95, 0.85])),
            season_weights=dict(cfg.get("season_weights", {"low": 0.85, "high": 1.2})),
            shared_shock_std=float(cfg.get("shared_shock_std", 0.08)),
            destination_shock_std=float(cfg.get("destination_shock_std", 0.12)),
            registration_curve=list(cfg.get("registration_curve", [0.04, 0.10, 0.18, 0.28, 0.39, 0.50, 0.62, 0.73, 0.83, 0.91, 0.97, 1.00])),
            cancellation_saving=float(cfg.get("cancellation_saving", 70.0)),
            penalty_unshipped_high=float(cfg.get("penalty_unshipped_high", 9.0)),
            penalty_unshipped_low=float(cfg.get("penalty_unshipped_low", 2.0)),
        )
        self._validate_params()
        self.rng = random.Random()
        self._true_high: List[int] = [0] * self.params.n_destinations
        self._true_low: List[int] = [0] * self.params.n_destinations
        self._keep_last_truck: List[int] = [1] * self.params.n_destinations
        self._decision_locked = False

    def set_seed(self, seed: Optional[int]) -> None:
        if seed is not None:
            self.rng.seed(seed)

    def _validate_params(self) -> None:
        p = self.params
        if p.n_destinations <= 0:
            raise ValueError("n_destinations must be positive")
        if p.time_steps_per_day <= 1:
            raise ValueError("time_steps_per_day must be > 1")
        if not (0 <= p.decision_step < p.time_steps_per_day):
            raise ValueError("decision_step must be in [0, time_steps_per_day - 1]")
        if len(p.trucks_per_day_by_destination) != p.n_destinations:
            raise ValueError("trucks_per_day_by_destination length mismatch")
        if len(p.destination_base_mean_high) != p.n_destinations:
            raise ValueError("destination_base_mean_high length mismatch")
        if len(p.destination_base_mean_low) != p.n_destinations:
            raise ValueError("destination_base_mean_low length mismatch")
        if len(p.day_of_week_weights) != 7:
            raise ValueError("day_of_week_weights must have 7 entries")
        if p.day_of_week < 0 or p.day_of_week > 6:
            raise ValueError("day_of_week must be in [0, 6]")
        if p.season not in p.season_weights:
            raise ValueError(f"season '{p.season}' not found in season_weights")
        if len(p.registration_curve) != p.time_steps_per_day:
            raise ValueError("registration_curve length must match time_steps_per_day")
        if any(x < 0.0 or x > 1.0 for x in p.registration_curve):
            raise ValueError("registration_curve values must be in [0, 1]")
        if any(p.registration_curve[i] > p.registration_curve[i + 1] for i in range(len(p.registration_curve) - 1)):
            raise ValueError("registration_curve must be non-decreasing")
        if abs(p.registration_curve[-1] - 1.0) > 1e-9:
            raise ValueError("registration_curve last value must be 1.0")

    def _generate_true_daily_volumes(self) -> Tuple[List[int], List[int]]:
        p = self.params
        day_mult = p.day_of_week_weights[p.day_of_week]
        season_mult = p.season_weights[p.season]
        shared_shock = self.rng.gauss(0.0, p.shared_shock_std)

        high: List[int] = []
        low: List[int] = []
        for i in range(p.n_destinations):
            dest_shock = self.rng.gauss(0.0, p.destination_shock_std)
            mult = max(0.05, 1.0 + shared_shock + dest_shock)
            high_mean = p.destination_base_mean_high[i] * day_mult * season_mult * mult
            low_mean = p.destination_base_mean_low[i] * day_mult * season_mult * mult
            high.append(max(0, int(round(high_mean))))
            low.append(max(0, int(round(low_mean))))
        return high, low

    def initial_state(self) -> Dict[str, Any]:
        self._true_high, self._true_low = self._generate_true_daily_volumes()
        self._keep_last_truck = [1] * self.params.n_destinations
        self._decision_locked = False
        return {
            "t": 0,
            "day_of_week": self.params.day_of_week,
            "season": self.params.season,
            "registered_high_by_destination": [0] * self.params.n_destinations,
            "registered_low_by_destination": [0] * self.params.n_destinations,
            "decision_locked": False,
        }

    def _normalize_action(self, action: Any) -> List[int]:
        n = self.params.n_destinations
        if action is None:
            return [1] * n

        if isinstance(action, dict):
            action = action.get("keep_last_truck", [1] * n)

        if isinstance(action, (int, bool)):
            return [1 if int(action) != 0 else 0] * n

        if isinstance(action, list):
            if len(action) != n:
                raise ValueError(f"Action list must have {n} entries")
            return [1 if int(a) != 0 else 0 for a in action]

        raise TypeError("Action must be None, int/bool, list, or dict with keep_last_truck")

    def _registered_until(self, totals: List[int], t: int) -> List[int]:
        frac = self.params.registration_curve[t]
        return [int(round(total * frac)) for total in totals]

    def transition(
        self, state: Dict[str, Any], action: Any
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        p = self.params
        t = int(state["t"])
        
        if t == p.decision_step and not self._decision_locked:
            self._keep_last_truck = self._normalize_action(action)
            self._decision_locked = True

        reg_high = self._registered_until(self._true_high, t)
        reg_low = self._registered_until(self._true_low, t)

        next_t = t + 1
        done = next_t >= p.time_steps_per_day
        reward = 0.0

        next_state = {
            "t": next_t,
            "day_of_week": p.day_of_week,
            "season": p.season,
            "registered_high_by_destination": reg_high,
            "registered_low_by_destination": reg_low,
            "decision_locked": self._decision_locked,
        }

        info: Dict[str, Any] = {
            "decision_step": p.decision_step,
            "keep_last_truck": list(self._keep_last_truck),
        }

        if done:
            base_capacity = [(x - 1) * p.truck_capacity for x in p.trucks_per_day_by_destination]
            full_capacity = [x * p.truck_capacity for x in p.trucks_per_day_by_destination]
            applied_capacity = [
                full_capacity[i] if self._keep_last_truck[i] == 1 else base_capacity[i]
                for i in range(p.n_destinations)
            ]

            loaded_high = [min(self._true_high[i], applied_capacity[i]) for i in range(p.n_destinations)]
            remaining_capacity = [applied_capacity[i] - loaded_high[i] for i in range(p.n_destinations)]
            loaded_low = [min(self._true_low[i], remaining_capacity[i]) for i in range(p.n_destinations)]

            unshipped_high = [self._true_high[i] - loaded_high[i] for i in range(p.n_destinations)]
            unshipped_low = [self._true_low[i] - loaded_low[i] for i in range(p.n_destinations)]
            last_truck_needed = [
                1 if (self._true_high[i] + self._true_low[i]) > base_capacity[i] else 0
                for i in range(p.n_destinations)
            ]

            saving = sum(
                p.cancellation_saving for i in range(p.n_destinations) if self._keep_last_truck[i] == 0
            )
            high_penalty = p.penalty_unshipped_high * float(sum(unshipped_high))
            low_penalty = p.penalty_unshipped_low * float(sum(unshipped_low))
            reward = saving - high_penalty - low_penalty

            info.update(
                {
                    "true_daily_high_by_destination": list(self._true_high),
                    "true_daily_low_by_destination": list(self._true_low),
                    "loaded_high_by_destination": loaded_high,
                    "loaded_low_by_destination": loaded_low,
                    "unshipped_high_by_destination": unshipped_high,
                    "unshipped_low_by_destination": unshipped_low,
                    "carry_over_next_day_by_destination": [
                        unshipped_high[i] + unshipped_low[i] for i in range(p.n_destinations)
                    ],
                    "last_truck_needed_by_destination": last_truck_needed,
                    "base_capacity_without_last": base_capacity,
                    "applied_capacity": applied_capacity,
                    "reward_breakdown": {
                        "cancellation_saving": saving,
                        "unshipped_high_penalty": high_penalty,
                        "unshipped_low_penalty": low_penalty,
                    },
                }
            )

        return next_state, reward, done, info


class TruckCancellationV0Environment(BaseEnvironment):
    def __init__(self, simulation: TruckCancellationV0Simulation):
        self.sim = simulation
        self.state: Dict[str, Any] = {}
        self.step_idx = 0
        self.episode_reward = 0.0
        self.last_reward = 0.0
        self.last_info: Dict[str, Any] = {}

    def reset(self, seed: Optional[int] = None):
        self.sim.set_seed(seed)
        self.state = self.sim.initial_state()
        self.step_idx = 0
        self.episode_reward = 0.0
        self.last_reward = 0.0
        self.last_info = {}
        return self.state, {"reset": True}

    def step(self, action: Any) -> StepResult:
        next_state, reward, done, info = self.sim.transition(self.state, action)
        self.state = next_state
        self.step_idx += 1
        self.last_reward = reward
        self.episode_reward += reward
        self.last_info = info
        return StepResult(next_state=next_state, reward=reward, done=done, info=info)

    def render(self) -> Dict[str, Any]:
        return {
            "episode_step": self.step_idx,
            "episode_reward": self.episode_reward,
            "last_reward": self.last_reward,
            "t": self.state.get("t"),
            "registered_high_by_destination": self.state.get("registered_high_by_destination"),
            "registered_low_by_destination": self.state.get("registered_low_by_destination"),
            "decision_locked": self.state.get("decision_locked"),
            "last_transition": self.last_info,
        }
