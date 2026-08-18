"""Shared formatting helpers for reporting modules (avoids circular imports)."""

from __future__ import annotations

import math


def fmt_value(val: object) -> str:
    if isinstance(val, float):
        if not math.isfinite(val):
            return ""
        return f"{val:.6f}"
    if isinstance(val, (int, bool)):
        return str(val)
    if val is None:
        return ""
    return str(val)


def maybe_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None
