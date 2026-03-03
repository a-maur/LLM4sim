from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def load_config(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    suffix = p.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "PyYAML is required for YAML config files. Install with: pip install pyyaml"
            ) from exc
        with p.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    if suffix == ".json":
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)

    raise ValueError("Unsupported config type. Use YAML or JSON.")
