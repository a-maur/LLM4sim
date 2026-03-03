from __future__ import annotations

import argparse

from rl_core.config_loader import load_config
from simulations.mail_flow import MailFlowEnvironment, MailFlowSimulation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config/mail_flow_random_defaults.yaml",
    )
    parser.add_argument("--episodes", type=int, default=1)
    args = parser.parse_args()

    cfg = load_config(args.config)
    sim = MailFlowSimulation(cfg)
    env = MailFlowEnvironment(sim)

    for ep in range(args.episodes):
        state, info = env.reset(seed=ep)
        total_reward = 0.0
        done = False
        while not done:
            print(env.render())  # can print or log this info as needed
            # skeleton policy: always action=0
            result = env.step(action=0)
            state = result.next_state
            total_reward += result.reward
            done = result.done
        print(env.render())
        print(
            f"episode={ep} total_reward={total_reward:.2f} "
            f"final_queue={state['queue_size']} dropped={state['dropped_mail']}"
        )


if __name__ == "__main__":
    main()
