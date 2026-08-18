from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from _bootstrap import config_path_from_env

from rdsynth.pipeline.runner import run_stage
from rdsynth.pipeline.data_prep import run_data_prep
from rdsynth.utils.config import load_yaml
from rdsynth.utils.pipeline_config import prepare_pipeline_config


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage3 only from existing Stage2 adversarial samples.")
    parser.add_argument("--config", default=config_path_from_env(), help="Base config path.")
    parser.add_argument("--adv-samples", default="", help="Override adv_samples.npz path for Stage3.")
    parser.add_argument("--out-dir", default="", help="Optional override for project.out_dir before running Stage3.")
    parser.add_argument("--prebuild-data", action="store_true", help="Run data prep before Stage3.")
    args = parser.parse_args()

    base_cfg = prepare_pipeline_config(load_yaml(args.config), args.config)
    cfg = dict(base_cfg)
    cfg["project"] = dict(base_cfg["project"])
    cfg["stage3"] = dict(base_cfg["stage3"])
    if args.adv_samples.strip():
        cfg["stage3"]["adv_samples_path"] = args.adv_samples.strip()
    if args.out_dir.strip():
        cfg["project"]["out_dir"] = args.out_dir.strip()

    out_dir = Path(cfg["project"]["out_dir"])
    pipeline_dir = out_dir / "pipeline"
    cfg_path = pipeline_dir / "stage3_from_stage2_config.yaml"
    _write_yaml(cfg_path, cfg)

    if args.prebuild_data:
        run_data_prep(cfg_path)
    run_stage("run_stage3.py", cfg_path, cfg["project"], stage_name="stage3")
    print(f"[Stage3Only] config={cfg_path}")
    print(f"[Stage3Only] out_dir={out_dir}")
    print(f"[Stage3Only] adv_samples={cfg['stage3'].get('adv_samples_path') or (out_dir / 'stage2' / 'adv_samples.npz')}")


if __name__ == "__main__":
    main()
