from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from rl_framework_skeleton.rl_core.base import BaseEnvironment, BaseSimulation, StepResult
from rl_framework_skeleton.rl_core.factory import build_distribution


@dataclass
class MailFlowParams:
    max_steps: int = 200
    queue_capacity: int = 200
    start_queue: int = 20
    process_rate: int = 5
    backlog_penalty: float = 0.01


class MailFlowSimulation(BaseSimulation):
    """Core mail-flow dynamics; distributions injected from outside."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        params_cfg = config.get("params", {})

        self.params = MailFlowParams(
            max_steps=int(params_cfg.get("max_steps", 200)),
            queue_capacity=int(params_cfg.get("queue_capacity", 200)),
            start_queue=int(params_cfg.get("start_queue", 20)),
            process_rate=int(params_cfg.get("process_rate", 5)),
            backlog_penalty=float(params_cfg.get("backlog_penalty", 0.01)),
        )

        dists_cfg = config.get("distributions", {})
        self.arrival_dist = build_distribution(dists_cfg.get("arrivals"), default_dim=1)

    def initial_state(self) -> Dict[str, Any]:
        return {
            "queue_size": self.params.start_queue,
            "time": 0,
            "dropped_mail": 0,
        }

    def transition(
        self, state: Dict[str, Any], action: Any, step_idx: int
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        # action is an optional processing boost in {-1, 0, +1} for this skeleton
        boost = int(action) if action is not None else 0
        effective_process_rate = max(0, self.params.process_rate + boost)

        sampled = self.arrival_dist.sample()
        arrivals = max(0, int(sampled[0] if isinstance(sampled, list) else sampled))

        queue_before = int(state["queue_size"])
        queue_after_arrivals = queue_before + arrivals

        dropped = max(0, queue_after_arrivals - self.params.queue_capacity)
        queue_capped = min(self.params.queue_capacity, queue_after_arrivals)

        processed = min(queue_capped, effective_process_rate)
        queue_next = queue_capped - processed

        next_state = {
            "queue_size": queue_next,
            "time": step_idx + 1,
            "dropped_mail": int(state["dropped_mail"]) + dropped,
        }

        reward = float(processed) - self.params.backlog_penalty * float(queue_next)
        done = (step_idx + 1) >= self.params.max_steps
        info = {
            "arrivals": arrivals,
            "processed": processed,
            "dropped": dropped,
        }
        return next_state, reward, done, info


class MailFlowEnvironment(BaseEnvironment):
    def __init__(self, simulation: MailFlowSimulation):
        self.sim = simulation
        self.state: Dict[str, Any] = {}
        self.step_idx = 0
        self.episode_reward = 0.0
        self.last_reward = 0.0
        self.last_info: Dict[str, Any] = {}

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            import random

            random.seed(seed)
        self.state = self.sim.initial_state()
        self.step_idx = 0
        self.episode_reward = 0.0
        self.last_reward = 0.0
        self.last_info = {}
        return self.state, {"reset": True}

    def step(self, action: Any) -> StepResult:
        next_state, reward, done, info = self.sim.transition(self.state, action, self.step_idx)
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
            "queue_size": self.state.get("queue_size"),
            "dropped_mail_total": self.state.get("dropped_mail"),
            "time": self.state.get("time"),
            "last_transition": self.last_info,
        }
