from __future__ import annotations

import argparse

from rl_core.config_loader import load_config
from simulations.truck_cancellation_v0 import (
    TruckCancellationV0Environment,
    TruckCancellationV0Simulation,
)


def policy_keep_or_cancel(state: dict, decision_step: int, base_capacity_without_last: list[int]) -> list[int] | None:
    t = int(state["t"])
    if t != decision_step:
        return None

    reg_high = state["registered_high_by_destination"]
    reg_low = state["registered_low_by_destination"]
    keep_last = []
    for i, cap in enumerate(base_capacity_without_last):
        observed = reg_high[i] + reg_low[i]
        keep_last.append(1 if observed > 0.90 * cap else 0)
    return keep_last


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/truck_cancellation_v0.yaml")
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    cfg = load_config(args.config)
    sim = TruckCancellationV0Simulation(cfg)
    env = TruckCancellationV0Environment(sim)

    params = cfg["params"]
    decision_step = int(params["decision_step"])
    cap = int(params["truck_capacity"])
    base_capacity_without_last = [(x - 1) * cap for x in params["trucks_per_day_by_destination"]]

    for ep in range(args.episodes):
        state, _ = env.reset(seed=ep)
        done = False
        total_reward = 0.0
        while not done:
            action = policy_keep_or_cancel(state, decision_step, base_capacity_without_last)
            result = env.step(action=action)
            state = result.next_state
            total_reward += result.reward
            done = result.done

        final = result.info
        print(
            f"episode={ep} total_reward={total_reward:.2f} "
            f"keep_last={final['keep_last_truck']} "
            f"needed={final['last_truck_needed_by_destination']} "
            f"carry_over={final['carry_over_next_day_by_destination']}"
        )


if __name__ == "__main__":
    main()
