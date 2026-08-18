"""Shared bootstrap helpers for scripts/ entrypoints."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def config_path_from_env() -> Optional[str]:
    return os.environ.get("RDSYNTH_CONFIG")


def resolve_config(config_arg: Optional[str] = None) -> str:
    if config_arg:
        return config_arg
    env = config_path_from_env()
    if env:
        return env
    return str(ROOT / "configs" / "demo_fast.yaml")
