from __future__ import annotations

import argparse

from rl_core.config_loader import load_config
from simulations.truck_cancellation_v01 import (
    TruckCancellationV01Environment,
    TruckCancellationV01Simulation,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/truck_cancellation_v01.yaml")
    parser.add_argument("--episodes", type=int, default=1)
    args = parser.parse_args()

    cfg = load_config(args.config)
    sim = TruckCancellationV01Simulation(cfg)
    env = TruckCancellationV01Environment(sim)

    decision_step = int(cfg["params"]["decision_step"])
    steps_per_day = int(cfg["params"]["time_steps_per_day"])

    for ep in range(args.episodes):
        state, _ = env.reset(seed=ep)
        done = False
        while not done:
            # Simple heuristic: cancel at decision step when observed incoming seems low.
            action = None
            if (env.step_idx % steps_per_day) == decision_step:
                observed = state["observed_incoming_volume_total"]
                action = 0 if observed < 1200 else 1
            result = env.step(action=action)
            state = result.next_state
            done = result.done

            # Print one summary line per day end.
            if result.info.get("time_in_day") == steps_per_day - 1:
                print(
                    f"day={result.info['day_index']} dow={result.info['day_of_week']} "
                    f"route={result.info['active_route']} cancel={result.info['cancel_last_truck_for_active_route']} "
                    f"needed={result.info['last_truck_needed_active_route']} "
                    f"carry={result.info['carryover_total_end_of_day']} reward={result.reward:.2f}"
                )

        print(f"episode={ep} total_reward={env.episode_reward:.2f}")


if __name__ == "__main__":
    main()
