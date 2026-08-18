from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from rdsynth.utils.artifacts import ensure_dir
from rdsynth.utils.config import load_yaml, require_section
from rdsynth.utils.seed import set_seed


@dataclass(frozen=True)
class StageRuntime:
    config_path: Path
    stage_name: str
    cfg: Mapping[str, Any]
    project_cfg: Mapping[str, Any]
    stage_cfg: Mapping[str, Any]
    seed: int
    deterministic: bool
    device: torch.device
    out_dir: Path


def resolve_torch_device(project_cfg: Mapping[str, Any]) -> torch.device:
    requested = str(os.environ.get("RDSYNTH_DEVICE", project_cfg.get("device", "auto"))).strip().lower()
    if requested in {"", "auto"}:
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Config requested CUDA, but torch.cuda.is_available() is False.")
    if device.type == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not torch.backends.mps.is_available():
            raise RuntimeError("Config requested MPS, but torch.backends.mps.is_available() is False.")
    return device


def configure_torch_runtime(project_cfg: Mapping[str, Any]) -> None:
    strict_repro = bool(project_cfg.get("strict_repro", False))
    deterministic = bool(project_cfg.get("deterministic", True))
    num_threads = project_cfg.get("num_threads")
    if num_threads is not None:
        torch.set_num_threads(int(num_threads))

    num_interop_threads = project_cfg.get("num_interop_threads")
    if num_interop_threads is not None:
        try:
            torch.set_num_interop_threads(int(num_interop_threads))
        except RuntimeError:
            # Inline pipeline mode may already have initialized intra-op workers
            # for the current process before a later stage reloads runtime config.
            pass

    matmul_precision = str(project_cfg.get("matmul_precision", "high"))
    if strict_repro:
        matmul_precision = "highest"
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(matmul_precision)
    if torch.cuda.is_available():
        # Deterministic CUDA kernels require the CuBLAS workspace contract to
        # avoid repeated warn_only messages during training and metric passes.
        if strict_repro or deterministic:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        allow_tf32 = bool(project_cfg.get("allow_tf32", True)) and not strict_repro
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
        torch.backends.cudnn.benchmark = False if strict_repro else not deterministic


def load_stage_runtime(config_path: str | Path, stage_name: str) -> StageRuntime:
    resolved_config_path = Path(config_path).resolve()
    cfg = load_yaml(resolved_config_path)
    project_cfg = require_section(cfg, "project")
    stage_cfg = require_section(cfg, stage_name)

    seed = int(project_cfg["seed"])
    deterministic = bool(project_cfg.get("deterministic", True))
    configure_torch_runtime(project_cfg)
    set_seed(seed, deterministic=deterministic)

    return StageRuntime(
        config_path=resolved_config_path,
        stage_name=stage_name,
        cfg=cfg,
        project_cfg=project_cfg,
        stage_cfg=stage_cfg,
        seed=seed,
        deterministic=deterministic,
        device=resolve_torch_device(project_cfg),
        out_dir=ensure_dir(Path(project_cfg["out_dir"]) / stage_name),
    )
