from __future__ import annotations

import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Mapping

from rdsynth.utils.artifacts import write_failure_record

ROOT = Path(__file__).resolve().parents[3]


def build_stage_env(cfg_path: Path, project_cfg: Mapping[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    env["RDSYNTH_CONFIG"] = cfg_path.as_posix()
    # Always force the configured seed so subprocess reproducibility is not
    # accidentally inherited from the caller's shell environment.
    env["PYTHONHASHSEED"] = str(int(project_cfg.get("seed", 0) or 0))

    num_threads = project_cfg.get("num_threads")
    if num_threads is not None:
        thread_value = str(int(num_threads))
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[key] = thread_value

    return env


def _resolve_stage_entrypoint(script: str) -> Callable[[str], None]:
    script_name = Path(script).name
    if script_name == "run_stage1.py":
        from rdsynth.pipeline.stage1 import run_stage1

        return run_stage1
    if script_name == "run_stage2.py":
        from rdsynth.pipeline.stage2 import main as run_stage2

        return run_stage2
    if script_name == "run_stage3.py":
        from rdsynth.pipeline.stage3 import main as run_stage3

        return run_stage3
    raise ValueError(f"Unsupported inline stage script: {script}")


@contextmanager
def stage_environment(cfg_path: Path, project_cfg: Mapping[str, Any]):
    env_updates = build_stage_env(cfg_path, project_cfg)
    original_env = {key: os.environ.get(key) for key in env_updates}
    for key, value in env_updates.items():
        os.environ[key] = value
    original_threads = {}
    try:
        import torch
    except Exception:  # pragma: no cover - torch is expected, but keep inline mode resilient.
        torch = None
    if torch is not None:
        original_threads["num_threads"] = torch.get_num_threads()
        original_threads["num_interop_threads"] = torch.get_num_interop_threads()
        num_threads = project_cfg.get("num_threads")
        if num_threads is not None:
            torch.set_num_threads(int(num_threads))
        num_interop_threads = project_cfg.get("num_interop_threads")
        if num_interop_threads is not None:
            try:
                torch.set_num_interop_threads(int(num_interop_threads))
            except RuntimeError:
                pass
    try:
        yield
    finally:
        for key, previous in original_env.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous
        if torch is not None:
            if "num_threads" in original_threads:
                torch.set_num_threads(int(original_threads["num_threads"]))
            if "num_interop_threads" in original_threads:
                try:
                    torch.set_num_interop_threads(int(original_threads["num_interop_threads"]))
                except RuntimeError:
                    pass


def run_stage(
    script: str,
    cfg_path: Path,
    project_cfg: Mapping[str, Any],
    *,
    stage_name: str | None = None,
    execution_mode: str = "inline",
) -> None:
    label = stage_name or Path(script).stem
    if execution_mode == "inline":
        entrypoint = _resolve_stage_entrypoint(script)
        try:
            with stage_environment(cfg_path, project_cfg):
                entrypoint(str(cfg_path))
        except Exception as exc:
            write_failure_record(
                project_cfg=project_cfg,
                config_path=cfg_path,
                stage_name=label,
                error=exc,
                extra_fields={"execution_mode": execution_mode},
            )
            raise
        return
    if execution_mode != "subprocess":
        raise ValueError(f"Unsupported execution_mode: {execution_mode}")
    cmd = [sys.executable, str(ROOT / "scripts" / script)]
    try:
        subprocess.run(
            cmd,
            check=True,
            env=build_stage_env(cfg_path, project_cfg),
            cwd=ROOT,
            timeout=project_cfg.get("stage_timeout_sec"),
        )
    except subprocess.TimeoutExpired as exc:
        write_failure_record(
            project_cfg=project_cfg,
            config_path=cfg_path,
            stage_name=label,
            error=exc,
            extra_fields={"execution_mode": execution_mode, "cmd": cmd},
        )
        raise TimeoutError(f"Stage '{label}' timed out after {exc.timeout} seconds.") from exc
    except subprocess.CalledProcessError as exc:
        write_failure_record(
            project_cfg=project_cfg,
            config_path=cfg_path,
            stage_name=label,
            error=exc,
            extra_fields={"execution_mode": execution_mode, "cmd": cmd, "returncode": exc.returncode},
        )
        raise RuntimeError(f"Stage '{label}' failed with exit code {exc.returncode}: {' '.join(cmd)}") from exc
