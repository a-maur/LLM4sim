from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from rl_core.base import BaseEnvironment, BaseSimulation, StepResult


@dataclass
class PriorityDemandConfig:
    base_mean_by_destination: List[float]
    day_of_week_weights: List[float]
    season_weights: Dict[str, float]
    shared_shock_std: float
    destination_shock_std: float
    registration_curve: List[float]


@dataclass
class TruckCancellationV01Params:
    n_destinations: int
    time_steps_per_day: int
    episode_days: int
    start_day_of_week: int
    truck_capacity: int
    booking_service_level: float
    booking_mc_samples: int
    active_route_mode: str
    default_season: str
    season_cycle: List[str]
    decision_step: int
    cancellation_saving: float
    penalty_unshipped_high: float
    penalty_unshipped_low: float
    high: PriorityDemandConfig
    low: PriorityDemandConfig


def _quantile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = max(0, min(len(s) - 1, math.ceil(q * len(s)) - 1))
    return s[idx]


class TruckCancellationV01Simulation(BaseSimulation):
    """Route-level truck cancellation with day rollover and in-day departures."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = (config or {}).get("params", {})
        high_cfg = dict(cfg.get("high_priority", {}))
        low_cfg = dict(cfg.get("low_priority", {}))

        self.params = TruckCancellationV01Params(
            n_destinations=int(cfg.get("n_destinations", 4)),
            time_steps_per_day=int(cfg.get("time_steps_per_day", 16)),
            episode_days=int(cfg.get("episode_days", 14)),
            start_day_of_week=int(cfg.get("start_day_of_week", 0)),
            truck_capacity=int(cfg.get("truck_capacity", 120)),
            booking_service_level=float(cfg.get("booking_service_level", 0.95)),
            booking_mc_samples=int(cfg.get("booking_mc_samples", 500)),
            active_route_mode=str(cfg.get("active_route_mode", "cycle")),
            default_season=str(cfg.get("default_season", "high")),
            season_cycle=list(cfg.get("season_cycle", ["high", "high", "high", "high", "high", "low", "low"])),
            decision_step=int(cfg.get("decision_step", 8)),
            cancellation_saving=float(cfg.get("cancellation_saving", 75.0)),
            penalty_unshipped_high=float(cfg.get("penalty_unshipped_high", 10.0)),
            penalty_unshipped_low=float(cfg.get("penalty_unshipped_low", 2.0)),
            high=PriorityDemandConfig(
                base_mean_by_destination=list(high_cfg.get("base_mean_by_destination", [95, 80, 90, 75])),
                day_of_week_weights=list(high_cfg.get("day_of_week_weights", [1.10, 1.00, 0.95, 1.05, 1.15, 0.90, 0.85])),
                season_weights=dict(high_cfg.get("season_weights", {"low": 0.85, "high": 1.20})),
                shared_shock_std=float(high_cfg.get("shared_shock_std", 0.10)),
                destination_shock_std=float(high_cfg.get("destination_shock_std", 0.12)),
                registration_curve=list(high_cfg.get("registration_curve", [0.45, 0.62, 0.74, 0.82, 0.88, 0.92, 0.95, 0.97, 0.98, 0.99, 0.995, 1.0, 1.0, 1.0, 1.0, 1.0])),
            ),
            low=PriorityDemandConfig(
                base_mean_by_destination=list(low_cfg.get("base_mean_by_destination", [180, 150, 170, 145])),
                day_of_week_weights=list(low_cfg.get("day_of_week_weights", [1.05, 1.00, 0.98, 1.03, 1.10, 0.92, 0.88])),
                season_weights=dict(low_cfg.get("season_weights", {"low": 0.90, "high": 1.10})),
                shared_shock_std=float(low_cfg.get("shared_shock_std", 0.08)),
                destination_shock_std=float(low_cfg.get("destination_shock_std", 0.10)),
                registration_curve=list(low_cfg.get("registration_curve", [0.38, 0.55, 0.68, 0.78, 0.85, 0.90, 0.94, 0.96, 0.975, 0.985, 0.992, 0.997, 1.0, 1.0, 1.0, 1.0])),
            ),
        )
        self._validate_params()
        self.rng = random.Random()

        # Episode counters.
        self._day_idx = 0
        self._time_idx = 0

        # Hidden inventories and day context.
        self._queue_high = [0] * self.params.n_destinations
        self._queue_low = [0] * self.params.n_destinations
        self._carryover_total_start_day = 0
        self._active_route = 0
        self._day_of_week = self.params.start_day_of_week
        self._season = self.params.default_season

        # Hidden current-day generated arrivals.
        self._daily_arrivals_high = [0] * self.params.n_destinations
        self._daily_arrivals_low = [0] * self.params.n_destinations
        self._booked_trucks = [1] * self.params.n_destinations
        self._departure_times: List[List[int]] = [[] for _ in range(self.params.n_destinations)]
        self._decision_locked = False
        self._cancel_last_for_active_route = False
        self._last_truck_needed_active_route = 0
        self._last_truck_departure_time = self.params.time_steps_per_day - 1

        # Counterfactual track for target label on active route.
        self._cf_queue_high_route = 0
        self._cf_queue_low_route = 0

    def set_seed(self, seed: Optional[int]) -> None:
        if seed is not None:
            self.rng.seed(seed)

    def _validate_curve(self, curve: List[float], name: str) -> None:
        if len(curve) != self.params.time_steps_per_day:
            raise ValueError(f"{name}: length must match time_steps_per_day")
        if any(x < 0.0 or x > 1.0 for x in curve):
            raise ValueError(f"{name}: values must be in [0, 1]")
        if any(curve[i] > curve[i + 1] for i in range(len(curve) - 1)):
            raise ValueError(f"{name}: must be non-decreasing")
        if abs(curve[-1] - 1.0) > 1e-9:
            raise ValueError(f"{name}: last value must be 1.0")

    def _validate_params(self) -> None:
        p = self.params
        if p.n_destinations <= 0:
            raise ValueError("n_destinations must be positive")
        if p.time_steps_per_day <= 1:
            raise ValueError("time_steps_per_day must be > 1")
        if p.episode_days <= 0:
            raise ValueError("episode_days must be positive")
        if not (0 <= p.start_day_of_week <= 6):
            raise ValueError("start_day_of_week must be in [0, 6]")
        if not (0 <= p.decision_step < p.time_steps_per_day):
            raise ValueError("decision_step must be in [0, time_steps_per_day - 1]")
        if p.truck_capacity <= 0:
            raise ValueError("truck_capacity must be positive")
        if p.booking_mc_samples < 100:
            raise ValueError("booking_mc_samples must be >= 100")
        if not (0.5 <= p.booking_service_level < 1.0):
            raise ValueError("booking_service_level must be in [0.5, 1.0)")
        if p.active_route_mode not in {"cycle", "random"}:
            raise ValueError("active_route_mode must be 'cycle' or 'random'")
        if not p.season_cycle:
            raise ValueError("season_cycle must not be empty")
        if len(p.high.base_mean_by_destination) != p.n_destinations:
            raise ValueError("high_priority base_mean_by_destination length mismatch")
        if len(p.low.base_mean_by_destination) != p.n_destinations:
            raise ValueError("low_priority base_mean_by_destination length mismatch")
        if len(p.high.day_of_week_weights) != 7 or len(p.low.day_of_week_weights) != 7:
            raise ValueError("day_of_week_weights for both priorities must have 7 entries")
        self._validate_curve(p.high.registration_curve, "high_priority.registration_curve")
        self._validate_curve(p.low.registration_curve, "low_priority.registration_curve")
        if p.default_season not in p.high.season_weights or p.default_season not in p.low.season_weights:
            raise ValueError("default_season must exist in both high and low season_weights")
        for season in p.season_cycle:
            if season not in p.high.season_weights or season not in p.low.season_weights:
                raise ValueError(f"season '{season}' missing in high/low season_weights")

    def _season_for_day(self, day_idx: int) -> str:
        if self.params.season_cycle:
            return self.params.season_cycle[day_idx % len(self.params.season_cycle)]
        return self.params.default_season

    def _draw_priority_volume(self, cfg: PriorityDemandConfig, destination: int, day_of_week: int, season: str) -> int:
        mean = (
            cfg.base_mean_by_destination[destination]
            * cfg.day_of_week_weights[day_of_week]
            * cfg.season_weights[season]
        )
        shared = self.rng.gauss(0.0, cfg.shared_shock_std)
        dest = self.rng.gauss(0.0, cfg.destination_shock_std)
        mult = max(0.05, 1.0 + shared + dest)
        return max(0, int(round(mean * mult)))

    def _sample_daily_volumes(self, day_of_week: int, season: str) -> Tuple[List[int], List[int]]:
        high = [
            self._draw_priority_volume(self.params.high, d, day_of_week, season)
            for d in range(self.params.n_destinations)
        ]
        low = [
            self._draw_priority_volume(self.params.low, d, day_of_week, season)
            for d in range(self.params.n_destinations)
        ]
        return high, low

    def _booked_trucks_for_route(self, route: int, day_of_week: int, season: str) -> int:
        totals: List[float] = []
        for _ in range(self.params.booking_mc_samples):
            h = self._draw_priority_volume(self.params.high, route, day_of_week, season)
            l = self._draw_priority_volume(self.params.low, route, day_of_week, season)
            totals.append(float(h + l))
        qv = _quantile(totals, self.params.booking_service_level)
        return max(1, int(math.ceil(qv / float(self.params.truck_capacity))))

    def _uniform_departure_schedule(self, trucks: int) -> List[int]:
        tmax = self.params.time_steps_per_day - 1
        if trucks <= 1:
            return [tmax]
        return [int(round(i * tmax / (trucks - 1))) for i in range(trucks)]

    def _registered_increment(self, total: int, curve: List[float], t: int) -> int:
        prev_frac = 0.0 if t == 0 else curve[t - 1]
        curr_frac = curve[t]
        prev_count = int(round(total * prev_frac))
        curr_count = int(round(total * curr_frac))
        return max(0, curr_count - prev_count)

    def _select_active_route(self) -> int:
        if self.params.active_route_mode == "random":
            return self.rng.randrange(self.params.n_destinations)
        return self._day_idx % self.params.n_destinations

    def _start_new_day(self) -> None:
        self._day_of_week = (self.params.start_day_of_week + self._day_idx) % 7
        self._season = self._season_for_day(self._day_idx)
        self._active_route = self._select_active_route()
        self._time_idx = 0
        self._decision_locked = False
        self._cancel_last_for_active_route = False
        self._last_truck_needed_active_route = 0
        self._carryover_total_start_day = sum(self._queue_high) + sum(self._queue_low)

        self._daily_arrivals_high, self._daily_arrivals_low = self._sample_daily_volumes(
            day_of_week=self._day_of_week, season=self._season
        )
        self._booked_trucks = [
            self._booked_trucks_for_route(route=d, day_of_week=self._day_of_week, season=self._season)
            for d in range(self.params.n_destinations)
        ]
        self._departure_times = [
            self._uniform_departure_schedule(self._booked_trucks[d])
            for d in range(self.params.n_destinations)
        ]
        self._last_truck_departure_time = self._departure_times[self._active_route][-1]
        self._cf_queue_high_route = self._queue_high[self._active_route]
        self._cf_queue_low_route = self._queue_low[self._active_route]

    def _observable_state(self) -> Dict[str, Any]:
        total_registered_today = 0
        for d in range(self.params.n_destinations):
            total_registered_today += int(
                round(self._daily_arrivals_high[d] * self.params.high.registration_curve[self._time_idx])
            )
            total_registered_today += int(
                round(self._daily_arrivals_low[d] * self.params.low.registration_curve[self._time_idx])
            )
        return {
            "observed_incoming_volume_total": self._carryover_total_start_day + total_registered_today,
        }

    def initial_state(self) -> Dict[str, Any]:
        self._day_idx = 0
        self._queue_high = [0] * self.params.n_destinations
        self._queue_low = [0] * self.params.n_destinations
        self._start_new_day()
        return self._observable_state()

    def _normalize_action(self, action: Any) -> bool:
        # action semantics for active route only: 1 keep last truck, 0 cancel last truck
        if action is None:
            return False
        if isinstance(action, dict):
            action = action.get("keep_last_truck", 1)
        if isinstance(action, list):
            if not action:
                return False
            action = action[0]
        return int(action) == 0

    def _dispatch_one_truck(self, route: int) -> Tuple[int, int]:
        cap = self.params.truck_capacity
        load_high = min(self._queue_high[route], cap)
        cap_left = cap - load_high
        load_low = min(self._queue_low[route], cap_left)
        self._queue_high[route] -= load_high
        self._queue_low[route] -= load_low
        return load_high, load_low

    def transition(
        self, state: Dict[str, Any], action: Any
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        t = self._time_idx
        active = self._active_route

        if t == self.params.decision_step and not self._decision_locked:
            self._cancel_last_for_active_route = self._normalize_action(action)
            self._decision_locked = True

        # Register incoming increments (hidden per destination/prio).
        for d in range(self.params.n_destinations):
            inc_h = self._registered_increment(
                total=self._daily_arrivals_high[d], curve=self.params.high.registration_curve, t=t
            )
            inc_l = self._registered_increment(
                total=self._daily_arrivals_low[d], curve=self.params.low.registration_curve, t=t
            )
            self._queue_high[d] += inc_h
            self._queue_low[d] += inc_l
            if d == active:
                self._cf_queue_high_route += inc_h
                self._cf_queue_low_route += inc_l

        loaded_by_destination: List[Tuple[int, int]] = [(0, 0) for _ in range(self.params.n_destinations)]

        # Dispatch trucks scheduled at current time.
        for d in range(self.params.n_destinations):
            times = self._departure_times[d]
            for dep_idx, dep_time in enumerate(times):
                if dep_time != t:
                    continue

                is_active_last = d == active and dep_idx == (len(times) - 1)
                if is_active_last:
                    if (self._cf_queue_high_route + self._cf_queue_low_route) > 0:
                        self._last_truck_needed_active_route = 1

                    # Counterfactual always keeps last truck for label generation.
                    cf_h = min(self._cf_queue_high_route, self.params.truck_capacity)
                    cf_rem = self.params.truck_capacity - cf_h
                    cf_l = min(self._cf_queue_low_route, cf_rem)
                    self._cf_queue_high_route -= cf_h
                    self._cf_queue_low_route -= cf_l

                    # Actual system may cancel.
                    if self._cancel_last_for_active_route:
                        continue

                lh, ll = self._dispatch_one_truck(d)
                prev_h, prev_l = loaded_by_destination[d]
                loaded_by_destination[d] = (prev_h + lh, prev_l + ll)

        reward = 0.0
        day_done = (t + 1) >= self.params.time_steps_per_day
        episode_done = False

        info: Dict[str, Any] = {
            "active_route": active,
            "day_index": self._day_idx,
            "day_of_week": self._day_of_week,
            "season": self._season,
            "time_in_day": t,
            "decision_step": self.params.decision_step,
            "decision_locked": self._decision_locked,
            "cancel_last_truck_for_active_route": self._cancel_last_for_active_route,
            "booked_trucks_active_route": self._booked_trucks[active],
            "last_truck_departure_time_active_route": self._last_truck_departure_time,
        }

        if day_done:
            unshipped_high = self._queue_high[active]
            unshipped_low = self._queue_low[active]

            if self._cancel_last_for_active_route:
                reward += self.params.cancellation_saving
            reward -= self.params.penalty_unshipped_high * float(unshipped_high)
            reward -= self.params.penalty_unshipped_low * float(unshipped_low)

            info.update(
                {
                    "last_truck_needed_active_route": self._last_truck_needed_active_route,
                    "unshipped_high_active_route": unshipped_high,
                    "unshipped_low_active_route": unshipped_low,
                    "carryover_total_end_of_day": sum(self._queue_high) + sum(self._queue_low),
                    "true_daily_arrivals_high_by_destination": list(self._daily_arrivals_high),
                    "true_daily_arrivals_low_by_destination": list(self._daily_arrivals_low),
                    "booked_trucks_by_destination": list(self._booked_trucks),
                    "departure_times_by_destination": [list(x) for x in self._departure_times],
                    "loaded_this_step_by_destination": loaded_by_destination,
                    "reward_breakdown": {
                        "cancellation_saving": self.params.cancellation_saving
                        if self._cancel_last_for_active_route
                        else 0.0,
                        "unshipped_high_penalty": self.params.penalty_unshipped_high * float(unshipped_high),
                        "unshipped_low_penalty": self.params.penalty_unshipped_low * float(unshipped_low),
                    },
                }
            )

            self._day_idx += 1
            if self._day_idx >= self.params.episode_days:
                episode_done = True
            else:
                self._start_new_day()
        else:
            self._time_idx += 1

        next_state = self._observable_state() if not episode_done else {
            "observed_incoming_volume_total": sum(self._queue_high) + sum(self._queue_low)
        }
        return next_state, reward, episode_done, info


class TruckCancellationV01Environment(BaseEnvironment):
    def __init__(self, simulation: TruckCancellationV01Simulation):
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
            "observed_incoming_volume_total": self.state.get("observed_incoming_volume_total"),
            "last_transition": self.last_info,
        }
