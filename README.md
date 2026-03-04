# RL Framework Skeleton

A starter framework for building multiple RL environments with a shared structure.

## Design goals
- Reusable base environment for consistent APIs.
- Strict separation between:
  - **Core simulation logic** (causality/transition rules).
  - **Input distributions** (real or placeholder/random).
- Config-driven setup (YAML/JSON).
- Random multi-dimensional defaults when inputs are missing.

## Structure
- `rl_core/`: common environment and distribution utilities.
- `simulations/`: problem-specific simulation logic.
- `config/`: sample configuration files.
- `examples/`: small runnable examples.

## Quick start
```bash
python -m examples.run_demo --config config/mail_flow.yaml
```

If a distribution block is missing in config, a random distribution is used by default.

## Truck cancellation model (v0)
Two-step workflow:
1. Decision-model specification in `specs/truck_cancellation_v0.yaml`
2. Simulation/environment implementation in `simulations/truck_cancellation_v0.py`

Run example:
```bash
python -m examples.run_truck_cancellation_v0 --config config/truck_cancellation_v0.yaml
```

## Truck cancellation model (v01)
Adds:
- Day rollover with day-of-week progression.
- Intra-day truck departures based on schedule (uniform by default).
- Historical booking policy to target 95% service-level capacity.
- Route-level decision (one active route per day).
- Priority-specific demand and registration models (high/low can differ).
- Partial observability: state exposes only aggregated incoming volume.

Files:
1. `specs/truck_cancellation_v01.yaml`
2. `config/truck_cancellation_v01.yaml`
3. `simulations/truck_cancellation_v01.py`

Run example:
```bash
python -m examples.run_truck_cancellation_v01 --config config/truck_cancellation_v01.yaml
```
