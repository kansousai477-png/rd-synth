from __future__ import annotations

# ruff: noqa: E402,I001

import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from _bootstrap import run_from_env

from rdsynth.pipeline.stage3 import main


if __name__ == "__main__":
    try:
        run_from_env(main)
    except BaseException:
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
