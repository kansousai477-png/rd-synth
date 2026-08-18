from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from rdsynth.pipeline.stage2_eval import sanitize_feature_array
from rdsynth.stages.stage3_targets import build_remap_targets
from rdsynth.utils.constraints import infer_constraints


@dataclass(frozen=True)
class Stage2EvalInputs:
    x_mal_eval: np.ndarray
    x_ben_eval: np.ndarray
    denorm_mean: np.ndarray
    denorm_std: np.ndarray
    x_ben_norm: np.ndarray
    x_ben_pre: np.ndarray
    x_mal_norm: np.ndarray
    x_mal_pre: np.ndarray
    x_adv_denorm: np.ndarray | None
    x_ben_denorm: np.ndarray | None
    x_mal_denorm: np.ndarray | None
    eval_denorm: bool
    sample_denorm: bool
    pre_min: np.ndarray | None
    pre_max: np.ndarray | None
    pull_alpha: float
    pull_k: int
    moment_alpha: float
    moment_std_floor: float


@dataclass(frozen=True)
class Stage2ConstraintInputs:
    constraints_enabled: bool
    constraints_spec: Any
    constraints_cfg: Mapping[str, Any]
    deploy_enabled: bool
    port_policy: str
    flag_policy: str
    temporal_policy: str
    port_allowlist: list[int]
    norm_bounds_min: Any
    norm_bounds_max: Any
    norm_nonneg: np.ndarray
    x_ben_raw_full: np.ndarray | None
    x_ben_mod_targets: np.ndarray | None
    remap_target_mean: np.ndarray | None
    remap_target_scale: np.ndarray | None


def coerce_port_allowlist(raw_value: object) -> list[int]:
    if isinstance(raw_value, str):
        return [int(s.strip()) for s in raw_value.split(",") if s.strip().isdigit()]
    if isinstance(raw_value, (list, tuple)):
        values: list[int] = []
        for item in raw_value:
            try:
                values.append(int(item))
            except (TypeError, ValueError):
                continue
        return values
    return []


def prepare_stage2_eval_inputs(
    *,
    cfg: Mapping[str, Any],
    settings: Any,
    seed: int,
    x_ben: np.ndarray,
    x_mal: np.ndarray,
    diffusion_bundle: Any,
    preprocessor: Any,
) -> Stage2EvalInputs:
    eval_n = int(settings.eval_samples)
    rng = np.random.default_rng(seed)
    idx_m = rng.choice(len(x_mal), min(eval_n, len(x_mal)), replace=False)
    idx_b = rng.choice(len(x_ben), min(eval_n, len(x_ben)), replace=False)
    x_mal_eval = sanitize_feature_array(x_mal[idx_m])
    x_ben_eval = sanitize_feature_array(x_ben[idx_b])

    denorm_mean = diffusion_bundle.ben_stats.get("denorm_mean")
    denorm_std = diffusion_bundle.ben_stats.get("denorm_std")
    if denorm_mean is None or denorm_std is None:
        denorm_mean = np.zeros(x_ben_eval.shape[1], dtype=np.float32)
        denorm_std = np.ones(x_ben_eval.shape[1], dtype=np.float32)

    x_ben_norm = (x_ben_eval - denorm_mean) / (denorm_std + 1.0e-8)
    x_ben_pre = x_ben_eval
    x_mal_norm = (x_mal_eval - denorm_mean) / (denorm_std + 1.0e-8)
    x_mal_pre = x_mal_eval

    eval_denorm = bool(cfg["stage2"].get("eval_denorm_metrics", False))
    x_adv_denorm = None
    x_ben_denorm = None
    x_mal_denorm = None
    pre_min = None
    pre_max = None
    if eval_denorm:
        pre_min = np.min(x_ben_pre, axis=0)
        pre_max = np.max(x_ben_pre, axis=0)
        x_ben_denorm = preprocessor.inverse_transform(x_ben_pre)
        x_mal_denorm = preprocessor.inverse_transform(x_mal_pre)

    return Stage2EvalInputs(
        x_mal_eval=x_mal_eval,
        x_ben_eval=x_ben_eval,
        denorm_mean=denorm_mean,
        denorm_std=denorm_std,
        x_ben_norm=x_ben_norm,
        x_ben_pre=x_ben_pre,
        x_mal_norm=x_mal_norm,
        x_mal_pre=x_mal_pre,
        x_adv_denorm=x_adv_denorm,
        x_ben_denorm=x_ben_denorm,
        x_mal_denorm=x_mal_denorm,
        eval_denorm=eval_denorm,
        sample_denorm=bool(settings.sample_denorm_output),
        pre_min=pre_min,
        pre_max=pre_max,
        pull_alpha=float(cfg["stage2"].get("sample_pullback_alpha", 0.0)),
        pull_k=int(cfg["stage2"].get("sample_pullback_k", 1)),
        moment_alpha=float(cfg["stage2"].get("sample_moment_alpha", 0.0)),
        moment_std_floor=float(cfg["stage2"].get("sample_moment_std_floor", 1.0e-3)),
    )


def prepare_stage2_constraint_inputs(
    *,
    cfg: Mapping[str, Any],
    diffusion_bundle: Any,
    preprocessor: Any,
    feature_names: Sequence[str],
    x_ben: np.ndarray,
    x_ben_norm: np.ndarray,
    metrics_payload: dict[str, Any],
) -> Stage2ConstraintInputs:
    constraints_cfg = cfg["stage2"].get("constraints", {})
    constraints_enabled = bool(constraints_cfg.get("enable", True))
    x_ben_raw_full = None
    if constraints_enabled or cfg["stage2"].get("deployable_constraints", {}).get("enable", False):
        x_ben_raw_full = preprocessor.inverse_transform(sanitize_feature_array(x_ben))

    x_ben_mod_targets = None
    remap_target_mean = None
    remap_target_scale = None
    if x_ben_raw_full is not None:
        x_ben_mod_targets = build_remap_targets(x_ben_raw_full, list(feature_names))
        remap_target_mean = np.mean(x_ben_mod_targets, axis=0).astype(np.float32)
        remap_target_scale = np.maximum(np.std(x_ben_mod_targets, axis=0), 1.0e-3).astype(np.float32)

    constraints_spec = None
    if constraints_enabled:
        constraints_spec = infer_constraints(
            x_ben_raw_full,
            integer_tol=float(constraints_cfg.get("integer_tol", 0.05)),
            integer_frac=float(constraints_cfg.get("integer_frac", 0.95)),
            nonneg_tol=float(constraints_cfg.get("nonneg_tol", 1.0e-8)),
        )
        metrics_payload["constraints_enabled"] = True
    else:
        metrics_payload["constraints_enabled"] = False

    deploy_cfg = cfg["stage2"].get("deployable_constraints", {}) or {}
    deploy_enabled = bool(deploy_cfg.get("enable", False))
    port_policy = str(deploy_cfg.get("port_policy", "keep")).lower()
    flag_policy = str(deploy_cfg.get("flag_policy", "clip")).lower()
    temporal_policy = str(deploy_cfg.get("temporal_policy", "clip_benign")).lower()
    port_allowlist = coerce_port_allowlist(deploy_cfg.get("port_allowlist", []))
    norm_bounds_min = diffusion_bundle.ben_stats.get("min")
    norm_bounds_max = diffusion_bundle.ben_stats.get("max")
    norm_nonneg = np.zeros(x_ben_norm.shape[1], dtype=bool)
    metrics_payload["deployable_constraints_enabled"] = deploy_enabled
    if deploy_enabled:
        metrics_payload["deploy_port_policy"] = port_policy
        metrics_payload["deploy_flag_policy"] = flag_policy
        metrics_payload["deploy_temporal_policy"] = temporal_policy
        if port_allowlist:
            metrics_payload["deploy_port_allowlist"] = port_allowlist

    return Stage2ConstraintInputs(
        constraints_enabled=constraints_enabled,
        constraints_spec=constraints_spec,
        constraints_cfg=constraints_cfg,
        deploy_enabled=deploy_enabled,
        port_policy=port_policy,
        flag_policy=flag_policy,
        temporal_policy=temporal_policy,
        port_allowlist=port_allowlist,
        norm_bounds_min=norm_bounds_min,
        norm_bounds_max=norm_bounds_max,
        norm_nonneg=norm_nonneg,
        x_ben_raw_full=x_ben_raw_full,
        x_ben_mod_targets=x_ben_mod_targets,
        remap_target_mean=remap_target_mean,
        remap_target_scale=remap_target_scale,
    )
