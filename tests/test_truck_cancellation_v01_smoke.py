from __future__ import annotations

from rl_core.config_loader import load_config
from simulations.truck_cancellation_v01 import (
    TruckCancellationV01Environment,
    TruckCancellationV01Simulation,
)


def test_truck_cancellation_v01_runs_and_rolls_days() -> None:
    cfg = load_config("config/truck_cancellation_v01.yaml")
    sim = TruckCancellationV01Simulation(cfg)
    env = TruckCancellationV01Environment(sim)

    state, _ = env.reset(seed=42)
    done = False
    result = None

    while not done:
        action = None
        if (env.step_idx % cfg["params"]["time_steps_per_day"]) == cfg["params"]["decision_step"]:
            action = 1
        result = env.step(action=action)
        state = result.next_state
        done = result.done

    assert result is not None
    assert "last_truck_needed_active_route" in result.info
    assert "carryover_total_end_of_day" in result.info
    assert "day_of_week" in result.info
    assert "observed_incoming_volume_total" in state
