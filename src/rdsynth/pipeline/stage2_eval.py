from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from sklearn.neighbors import NearestNeighbors

from rdsynth.stages.stage3_targets import build_remap_targets, build_rule_based_modifications, clip_modifications
from rdsynth.utils.constraints import apply_constraints, infer_constraints
from rdsynth.utils.metrics_stage2 import compute_stage2_metrics, nearest_reference_distance, paired_sample_l2
from rdsynth.utils.traffic_schema import apply_schema_projection


def sanitize_feature_array(
    x: np.ndarray,
    *,
    clip_value: float = 1.0e4,
) -> np.ndarray:
    arr = np.nan_to_num(np.asarray(x, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if clip_value > 0.0:
        arr = np.clip(arr, -clip_value, clip_value)
    return arr


@dataclass
class Stage2EvalHelper:
    cfg: dict[str, Any]
    seed: int
    preprocessor: Any
    bundle_feature_names: list[str]
    denorm_mean: np.ndarray
    denorm_std: np.ndarray
    x_ben_norm: np.ndarray
    x_ben_raw_full: np.ndarray | None
    constraints_enabled: bool
    constraints_spec: Any
    constraints_cfg: dict[str, Any]
    deploy_enabled: bool
    traffic_schema: Any
    port_policy: str
    flag_policy: str
    temporal_policy: str
    port_allowlist: list[int]
    diffusion_bundle: Any
    norm_bounds_min: np.ndarray | None
    norm_bounds_max: np.ndarray | None
    norm_nonneg: np.ndarray
    x_ben_mod_targets: np.ndarray | None
    remap_target_mean: np.ndarray | None
    remap_target_scale: np.ndarray | None
    pull_alpha: float
    pull_k: int
    moment_alpha: float
    moment_std_floor: float
    surrogate_predict_probs: Callable[[np.ndarray, int], tuple[np.ndarray, np.ndarray]]
    oracle_predict_probs: Callable[[np.ndarray, int], tuple[np.ndarray, np.ndarray | None]]
    oracle: Any

    def _compute_remapability_proxy(
        self,
        adv_pre_local: np.ndarray,
    ) -> dict[str, object]:
        if (
            self.x_ben_raw_full is None
            or self.x_ben_mod_targets is None
            or self.remap_target_mean is None
            or self.remap_target_scale is None
        ):
            return {}
        adv_raw = self.preprocessor.inverse_transform(adv_pre_local)
        adv_mod_targets = build_remap_targets(adv_raw, self.bundle_feature_names)
        projected_mods = build_rule_based_modifications(
            x_adv_raw=adv_raw,
            x_ben_raw=self.x_ben_raw_full,
            feature_names=self.bundle_feature_names,
        )
        clipped_mods = clip_modifications(adv_mod_targets.copy())
        scale = np.maximum(np.asarray(self.remap_target_scale, dtype=np.float64), 1.0e-3)
        project_penalty = np.sqrt(np.mean(((projected_mods - adv_mod_targets) / scale) ** 2, axis=1))
        clip_penalty = np.sqrt(np.mean(((clipped_mods - adv_mod_targets) / scale) ** 2, axis=1))
        center_penalty = np.sqrt(np.mean(((adv_mod_targets - self.remap_target_mean[None, :]) / scale) ** 2, axis=1))
        total_penalty = project_penalty + 0.5 * clip_penalty + 0.25 * center_penalty
        remapability_score = 1.0 / (1.0 + total_penalty)
        target_l2 = np.sqrt(np.mean((projected_mods - adv_mod_targets) ** 2, axis=1))
        target_mae = np.mean(np.abs(projected_mods - adv_mod_targets), axis=1)
        return {
            "remap_projection_penalty": float(np.mean(project_penalty)),
            "remap_clip_penalty": float(np.mean(clip_penalty)),
            "remap_center_penalty": float(np.mean(center_penalty)),
            "remapability_score": float(np.mean(remapability_score)),
            "_per_sample_remap_penalty": total_penalty.astype(np.float64, copy=False),
            "_per_sample_stage3_target_l2": target_l2.astype(np.float64, copy=False),
            "_per_sample_stage3_target_mae": target_mae.astype(np.float64, copy=False),
        }

    def postprocess_adv(
        self,
        x_adv_local: np.ndarray,
        x_mal_local: np.ndarray,
        apply_constraints_local: bool = True,
        apply_deploy_local: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        sample_denorm = self.cfg["stage2"].get("sample_denorm_output", False)
        if sample_denorm:
            adv_pre = sanitize_feature_array(x_adv_local)
            adv_norm = (adv_pre - self.denorm_mean) / (self.denorm_std + 1.0e-8)
        else:
            adv_norm = sanitize_feature_array(x_adv_local)
            adv_pre = adv_norm * self.denorm_std + self.denorm_mean

        if self.norm_bounds_min is not None and self.norm_bounds_max is not None:
            adv_norm = np.clip(adv_norm, self.norm_bounds_min, self.norm_bounds_max)
            adv_pre = adv_norm * self.denorm_std + self.denorm_mean
        else:
            adv_norm = np.clip(adv_norm, -20.0, 20.0)
            adv_pre = sanitize_feature_array(adv_pre)

        if self.pull_alpha > 0.0 and self.pull_k > 0 and self.x_ben_norm.shape[0] > 0:
            pull_k = min(int(self.pull_k), int(self.x_ben_norm.shape[0]))
            nn = NearestNeighbors(n_neighbors=pull_k, metric="euclidean").fit(self.x_ben_norm)
            _, idx = nn.kneighbors(adv_norm, return_distance=True)
            ben_nn = np.mean(self.x_ben_norm[idx], axis=1)
            adv_norm = (1.0 - self.pull_alpha) * adv_norm + self.pull_alpha * ben_nn

        if self.moment_alpha > 0.0:
            adv_mean = np.mean(adv_norm, axis=0)
            adv_std = np.maximum(np.std(adv_norm, axis=0), self.moment_std_floor)
            ben_mean = np.mean(self.x_ben_norm, axis=0)
            ben_std = np.maximum(np.std(self.x_ben_norm, axis=0), self.moment_std_floor)
            adv_matched = (adv_norm - adv_mean) / adv_std * ben_std + ben_mean
            adv_norm = (1.0 - self.moment_alpha) * adv_norm + self.moment_alpha * adv_matched

        adv_pre = adv_norm * self.denorm_std + self.denorm_mean
        adv_pre = sanitize_feature_array(adv_pre)

        if self.constraints_enabled and apply_constraints_local:
            spec = self.constraints_spec
            if spec is None:
                spec = infer_constraints(
                    self.x_ben_raw_full,
                    integer_tol=float(self.constraints_cfg.get("integer_tol", 0.05)),
                    integer_frac=float(self.constraints_cfg.get("integer_frac", 0.95)),
                    nonneg_tol=float(self.constraints_cfg.get("nonneg_tol", 1.0e-8)),
                )
            adv_raw = self.preprocessor.inverse_transform(adv_pre)
            adv_raw = apply_constraints(
                adv_raw,
                spec,
                clip=bool(self.constraints_cfg.get("clip", True)),
                round_integer=bool(self.constraints_cfg.get("round_integer", True)),
            )
            adv_pre = self.preprocessor.transform(adv_raw)
            adv_norm = (adv_pre - self.denorm_mean) / (self.denorm_std + 1.0e-8)

        if self.deploy_enabled and apply_deploy_local and self.x_ben_raw_full is not None:
            adv_raw = self.preprocessor.inverse_transform(adv_pre)
            adv_raw = apply_schema_projection(
                x_adv=adv_raw,
                x_mal=self.preprocessor.inverse_transform(x_mal_local),
                x_ben=self.x_ben_raw_full,
                schema=self.traffic_schema,
                port_policy=self.port_policy,
                flag_policy=self.flag_policy,
                temporal_policy=self.temporal_policy,
                port_allowlist=self.port_allowlist,
            )
            adv_pre = self.preprocessor.transform(adv_raw)
            adv_norm = (adv_pre - self.denorm_mean) / (self.denorm_std + 1.0e-8)

        if self.cfg["stage2"].get("post_clip_norm_range", True):
            lo = self.diffusion_bundle.ben_stats.get("min")
            hi = self.diffusion_bundle.ben_stats.get("max")
            if lo is not None and hi is not None:
                adv_norm = np.maximum(np.minimum(adv_norm, hi), lo)
                adv_pre = adv_norm * self.denorm_std + self.denorm_mean

        return adv_pre, adv_norm

    def evaluate_candidate(
        self,
        x_adv_local: np.ndarray,
        x_mal_local: np.ndarray,
        *,
        x_ben_norm_ref: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
        adv_pre_local, adv_norm_local = self.postprocess_adv(x_adv_local, x_mal_local)
        ben_ref = self.x_ben_norm if x_ben_norm_ref is None else x_ben_norm_ref
        metrics_local = compute_stage2_metrics(
            ben_ref,
            adv_norm_local,
            feature_names=self.bundle_feature_names,
            max_real=self.cfg["stage2"].get("metrics_max_real", 2000),
            max_gen=self.cfg["stage2"].get("metrics_max_gen", 2000),
            seed=self.seed,
            bounds_min=self.norm_bounds_min,
            bounds_max=self.norm_bounds_max,
            nonneg_mask=self.norm_nonneg,
        ).as_dict()
        s_preds, s_probs = self.surrogate_predict_probs(adv_pre_local, self.cfg["stage2"].get("sample_batch_size", 512))
        o_preds = np.array([])
        o_probs = None
        if self.oracle is not None:
            o_preds, o_probs = self.oracle_predict_probs(
                adv_pre_local, self.cfg["stage2"].get("sample_batch_size", 512)
            )
        row = {
            "ffd": float(metrics_local.get("FFD", float("nan"))),
            "swd": float(metrics_local.get("SWD", float("nan"))),
            "energy": float(metrics_local.get("Energy", float("nan"))),
            "c2st_auc": float(metrics_local.get("C2ST-AUC", float("nan"))),
            "c2st_acc": float(metrics_local.get("C2ST-Acc", float("nan"))),
            "adv_to_ben_l2": nearest_reference_distance(adv_norm_local, ben_ref),
            "adv_to_mal_l2": paired_sample_l2(adv_pre_local, x_mal_local),
            "asr_surrogate": float(np.mean(s_preds == 0)) if s_preds.size else float("nan"),
            "adv_pmal_surrogate": float(np.mean(s_probs[:, 1])) if s_probs.size else float("nan"),
            "asr_oracle": float(np.mean(o_preds == 0)) if o_preds.size else float("nan"),
            "adv_pmal_oracle": float(np.mean(o_probs[:, 1])) if o_probs is not None and o_probs.size else float("nan"),
        }
        per_sample_support_penalty = np.min(
            np.linalg.norm(adv_norm_local[:, None, :] - ben_ref[None, :, :], axis=2), axis=1
        )
        row["support_score"] = float(np.mean(1.0 / (1.0 + per_sample_support_penalty)))
        row["_per_sample_support_penalty"] = per_sample_support_penalty.astype(np.float64, copy=False)
        row.update(self._compute_remapability_proxy(adv_pre_local))
        stage3_loop_metrics: dict[str, float] = {}
        row.update({})
        decision_score = row.get("asr_oracle")
        if isinstance(decision_score, float) and np.isfinite(decision_score):
            ffd_val = float(row.get("ffd", 0.0))
            row["selection_score"] = float(decision_score) - 0.01 * ffd_val
            row["stage3_closed_loop_score"] = float(decision_score)
        return adv_pre_local, adv_norm_local, row
