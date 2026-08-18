from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from _bootstrap import config_path_from_env

from rdsynth.pipeline.runner import run_stage
from rdsynth.utils.config import load_yaml
from rdsynth.utils.pipeline_config import prepare_pipeline_config

def main(config_path: str) -> None:
    cfg = prepare_pipeline_config(load_yaml(config_path), config_path)
    for stage_name, script_name in (
        ("stage1", "run_stage1.py"),
        ("stage2", "run_stage2.py"),
        ("stage3", "run_stage3.py"),
    ):
        run_stage(script_name, Path(config_path), cfg["project"], stage_name=stage_name)


if __name__ == "__main__":
    main(config_path_from_env())
