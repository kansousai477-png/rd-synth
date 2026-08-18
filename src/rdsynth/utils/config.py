from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping, got {type(data).__name__}.")
    return data


def require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping, got {type(value).__name__}.")
    return dict(value)


def require_section(cfg: Mapping[str, Any], section: str) -> dict[str, Any]:
    value = cfg.get(section)
    if value is None:
        raise ValueError(f"Config section '{section}' not found.")
    return require_mapping(value, f"Config section '{section}'")


def optional_section(cfg: Mapping[str, Any], section: str) -> dict[str, Any]:
    value = cfg.get(section)
    if value is None:
        return {}
    return require_mapping(value, f"Config section '{section}'")


def maybe_int(value: object, name: str = "", *, positive_only: bool = False) -> int | None:
    """Parse *value* as int or return None when empty/null.

    When *positive_only* is True, the parsed value must be > 0.
    *name* is used in error messages when provided.
    """
    if value is None or value == "":
        return None
    label = f" {name}" if name else ""
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Config key{label} must be an integer, got {value!r}.") from exc
    if positive_only and parsed <= 0:
        raise ValueError(f"Config key{label} must be positive when provided.")
    return parsed
