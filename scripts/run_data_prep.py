from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from _bootstrap import run_from_env

from rdsynth.pipeline.data_prep import run_data_prep


if __name__ == "__main__":
    run_from_env(run_data_prep)
