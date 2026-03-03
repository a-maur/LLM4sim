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
