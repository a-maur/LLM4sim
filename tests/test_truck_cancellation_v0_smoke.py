from __future__ import annotations

from rl_core.config_loader import load_config
from simulations.truck_cancellation_v0 import (
    TruckCancellationV0Environment,
    TruckCancellationV0Simulation,
)


def test_truck_cancellation_v0_runs_to_terminal_step() -> None:
    cfg = load_config("config/truck_cancellation_v0.yaml")
    sim = TruckCancellationV0Simulation(cfg)
    env = TruckCancellationV0Environment(sim)

    state, _ = env.reset(seed=123)
    done = False
    result = None

    while not done:
        action = None
        if state["t"] == cfg["params"]["decision_step"]:
            action = [1] * cfg["params"]["n_destinations"]
        result = env.step(action=action)
        state = result.next_state
        done = result.done

    assert result is not None
    assert "last_truck_needed_by_destination" in result.info
    assert len(result.info["last_truck_needed_by_destination"]) == cfg["params"]["n_destinations"]
    assert "carry_over_next_day_by_destination" in result.info
