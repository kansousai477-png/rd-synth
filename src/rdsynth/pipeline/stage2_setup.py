from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from rdsynth.pipeline.stage2_eval import Stage2EvalHelper
from rdsynth.pipeline.stage2_inference import Stage2Predictors, make_stage2_predictors
from rdsynth.pipeline.stage2_inputs import Stage2ConstraintInputs, Stage2EvalInputs
from rdsynth.utils.checkpoints import OracleRestoreData, load_stage1_artifacts
from rdsynth.utils.query_oracle import QueryOracle


@dataclass(frozen=True)
class Stage2Settings:
    oracle_name: str
    require_stage1: bool
    mode: str
    generator_backbone: str
    guidance_mode: str
    epochs: int
    batch_size: int
    lr: float
    eval_metrics: bool
    eval_samples: int
    sample_batch_size: int
    sample_clip_minmax: bool
    sample_denorm_output: bool
    sample_init_latent: str
    sample_init_editor: str
    latent_use_prior: bool
    guidance_scale: float
    latent_noise_scale: float
    residual_scale: float
    save_samples: bool
    save_intermediate_results: bool
    post_clip_norm_range: bool
    mal_anchor_alpha: float

    @classmethod
    def from_cfg(cls, stage2_cfg: Mapping[str, Any]) -> "Stage2Settings":
        return cls(
            oracle_name=str(stage2_cfg.get("oracle_name", "default")),
            require_stage1=bool(stage2_cfg.get("require_stage1", True)),
            mode=str(stage2_cfg.get("mode", "editor")),
            generator_backbone=str(stage2_cfg.get("generator_backbone", "ddpm")).lower(),
            guidance_mode=str(stage2_cfg.get("surrogate_guidance_mode", "embedding")).lower(),
            epochs=int(stage2_cfg["epochs"]),
            batch_size=int(stage2_cfg["batch_size"]),
            lr=float(stage2_cfg["lr"]),
            eval_metrics=bool(stage2_cfg.get("eval_metrics", False)),
            eval_samples=int(stage2_cfg.get("eval_samples", 2000)),
            sample_batch_size=int(stage2_cfg.get("sample_batch_size", 512)),
            sample_clip_minmax=bool(stage2_cfg.get("sample_clip_minmax", True)),
            sample_denorm_output=bool(stage2_cfg.get("sample_denorm_output", False)),
            sample_init_latent=str(stage2_cfg.get("sample_init", "benign_sample")),
            sample_init_editor=str(stage2_cfg.get("sample_init", "benign_mean")),
            latent_use_prior=bool(stage2_cfg.get("latent_use_prior", False)),
            guidance_scale=float(stage2_cfg.get("guidance_scale", 1.5)),
            latent_noise_scale=float(stage2_cfg.get("latent_noise_scale", 1.0)),
            residual_scale=float(stage2_cfg.get("residual_scale", 0.5)),
            save_samples=bool(stage2_cfg.get("save_samples", False)),
            save_intermediate_results=bool(stage2_cfg.get("save_intermediate_results", True)),
            post_clip_norm_range=bool(stage2_cfg.get("post_clip_norm_range", True)),
            mal_anchor_alpha=float(stage2_cfg.get("mal_anchor_alpha", 0.0)),
        )


def load_stage2_artifacts(
    *,
    cfg: Mapping[str, Any],
    oracle_name: str,
    require_stage1: bool,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
    device: Any,
    seed: int,
) -> Any:
    return load_stage1_artifacts(
        cfg=cfg,
        oracle_name=oracle_name,
        feature_dim=x_train.shape[1],
        n_classes=int(np.max(y_train)) + 1,
        surrogate_hidden_dims=cfg["stage1"]["sur_hidden"],
        feature_names=feature_names,
        device=device,
        require_checkpoint=require_stage1,
        oracle_restore_data=OracleRestoreData(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            seed=seed,
        ),
    )


@dataclass(frozen=True)
class Stage2PredictorSetup:
    predictors: Stage2Predictors
    attack_query_oracle: QueryOracle


def build_stage2_predictor_setup(
    *,
    stage2_cfg: dict[str, Any],
    surrogate: Any,
    oracle: Any,
    device: Any,
    batch_size: int,
    metrics_payload: dict[str, Any],
) -> Stage2PredictorSetup:
    predictors = make_stage2_predictors(
        surrogate=surrogate,
        oracle=oracle,
        device=device,
        batch_size=batch_size,
    )
    blackbox_cfg = stage2_cfg.get("blackbox_eval", {}) or {}
    attack_query_oracle = QueryOracle(
        predictors.attack_score_fn,
        max_queries=(
            None if blackbox_cfg.get("query_budget") in (None, "", "null") else int(blackbox_cfg.get("query_budget"))
        ),
        hard_label=bool(blackbox_cfg.get("hard_label_queries", False)),
        hard_label_threshold=float(blackbox_cfg.get("hard_label_threshold", 0.5)),
        exhausted_fill=float(blackbox_cfg.get("exhausted_fill", 1.0)),
    )
    metrics_payload["attack_score_hard_label_queries"] = bool(blackbox_cfg.get("hard_label_queries", False))
    if blackbox_cfg.get("query_budget") not in (None, "", "null"):
        metrics_payload["attack_score_query_budget"] = int(blackbox_cfg.get("query_budget"))
    return Stage2PredictorSetup(
        predictors=predictors,
        attack_query_oracle=attack_query_oracle,
    )


def build_stage2_eval_helper(
    *,
    cfg: dict[str, Any],
    seed: int,
    preprocessor: Any,
    bundle_feature_names: list[str],
    diffusion_bundle: Any,
    traffic_schema: Any,
    oracle: Any,
    eval_inputs: Stage2EvalInputs,
    constraint_inputs: Stage2ConstraintInputs,
    predictors: Stage2Predictors,
) -> Stage2EvalHelper:
    return Stage2EvalHelper(
        cfg=cfg,
        seed=seed,
        preprocessor=preprocessor,
        bundle_feature_names=bundle_feature_names,
        denorm_mean=eval_inputs.denorm_mean,
        denorm_std=eval_inputs.denorm_std,
        x_ben_norm=eval_inputs.x_ben_norm,
        x_ben_raw_full=constraint_inputs.x_ben_raw_full,
        constraints_enabled=constraint_inputs.constraints_enabled,
        constraints_spec=constraint_inputs.constraints_spec,
        constraints_cfg=constraint_inputs.constraints_cfg,
        deploy_enabled=constraint_inputs.deploy_enabled,
        traffic_schema=traffic_schema,
        port_policy=constraint_inputs.port_policy,
        flag_policy=constraint_inputs.flag_policy,
        temporal_policy=constraint_inputs.temporal_policy,
        port_allowlist=constraint_inputs.port_allowlist,
        diffusion_bundle=diffusion_bundle,
        norm_bounds_min=constraint_inputs.norm_bounds_min,
        norm_bounds_max=constraint_inputs.norm_bounds_max,
        norm_nonneg=constraint_inputs.norm_nonneg,
        x_ben_mod_targets=constraint_inputs.x_ben_mod_targets,
        remap_target_mean=constraint_inputs.remap_target_mean,
        remap_target_scale=constraint_inputs.remap_target_scale,
        pull_alpha=eval_inputs.pull_alpha,
        pull_k=eval_inputs.pull_k,
        moment_alpha=eval_inputs.moment_alpha,
        moment_std_floor=eval_inputs.moment_std_floor,
        surrogate_predict_probs=lambda x, _batch_size: predictors.surrogate_predict_probs(x),
        oracle_predict_probs=lambda x, _batch_size: predictors.oracle_predict_probs(x),
        oracle=oracle,
    )
