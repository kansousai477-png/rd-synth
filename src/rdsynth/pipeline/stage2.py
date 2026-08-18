from __future__ import annotations

import csv
import os
import time
from typing import Any

import numpy as np

from rdsynth.pipeline.data import load_data_context
from rdsynth.pipeline.preprocessing import DatasetPreprocessor
from rdsynth.pipeline.runtime import load_stage_runtime
from rdsynth.pipeline.stage2_analysis import (
    build_stage2_artifact_payload,
    print_stage2_metric_tables,
    save_pareto_front,
)
from rdsynth.pipeline.stage2_baselines import run_stage2_baselines
from rdsynth.pipeline.stage2_execution import execute_stage2_sample, run_stage2_candidate_selection
from rdsynth.pipeline.stage2_inputs import prepare_stage2_constraint_inputs, prepare_stage2_eval_inputs
from rdsynth.pipeline.stage2_outputs import save_stage2_manifest, save_stage2_state
from rdsynth.pipeline.stage2_runtime import (
    compute_stage2_distribution_metrics,
    persist_stage2_metrics,
    record_sample_runtime,
    run_stage2_pareto_eval,
    update_attack_metrics,
    update_sample_distribution_summary,
)
from rdsynth.pipeline.stage2_sampling import build_stage2_sampler
from rdsynth.pipeline.stage2_setup import (
    Stage2Settings,
    build_stage2_eval_helper,
    build_stage2_predictor_setup,
    load_stage2_artifacts,
)
from rdsynth.pipeline.stage2_training import train_stage2_generator
from rdsynth.utils.artifacts import save_config, save_training_log_csv
from rdsynth.utils.traffic_schema import infer_traffic_feature_schema


def _schema_counts(traffic_schema: Any) -> dict[str, int]:
    return {
        "schema_port_features": int(traffic_schema.port_idx.size),
        "schema_flag_features": int(traffic_schema.flag_idx.size),
        "schema_temporal_features": int(traffic_schema.temporal_idx.size),
        "schema_ratio_features": int(traffic_schema.ratio_idx.size),
        "schema_count_features": int(traffic_schema.count_idx.size),
    }


def _resolve_eval_attack_label(cfg: dict[str, Any]) -> str:
    project_cfg = cfg.get("project") or {}
    data_cfg = cfg.get("data") or {}
    for key in ("eval_attack_label", "attack_type"):
        value = project_cfg.get(key)
        if str(value or "").strip():
            return str(value).strip()
    for key in ("eval_attack_label", "attack_type", "attack", "attack_label"):
        value = data_cfg.get(key)
        if str(value or "").strip():
            return str(value).strip()
    include_labels = list(data_cfg.get("include_labels") or [])
    benign = {str(v).strip() for v in list(data_cfg.get("benign_labels") or [])}
    candidates = [str(v).strip() for v in include_labels if str(v).strip() and str(v).strip() not in benign]
    if candidates:
        label = candidates[-1]
        print(f"[Stage2] eval_attack_label not set; falling back to last non-benign label: '{label}'")
        return label
    return ""


def _discover_eval_attack_labels(bundle: Any) -> list[str]:
    raw_test = getattr(bundle, "raw_y_test", None)
    y_test = np.asarray(getattr(bundle, "y_test", np.empty((0,), dtype=np.int64)))
    if raw_test is None or len(raw_test) != len(y_test):
        return []
    labels: list[str] = []
    seen: set[str] = set()
    for raw_value, y_value in zip(np.asarray(raw_test, dtype=object), y_test):
        if int(y_value) != 1:
            continue
        attack = str(raw_value).strip()
        if not attack or attack in seen:
            continue
        seen.add(attack)
        labels.append(attack)
    return labels


def _write_stage2_attack_eval_index(out_dir: Any, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path = out_dir / "attack_eval_index.csv"
    fieldnames = [
        "attack_type",
        "out_dir",
        "stage2_eval_split",
        "stage2_eval_attack_rows",
        "stage2_eval_attack_filtered",
        "asr_surrogate",
        "asr_oracle",
        "norm_FFD",
        "norm_SWD",
        "norm_AdvToMal_L2",
        "sample_end_to_end_time_sec",
        "sample_end_to_end_samples_per_sec",
        "attack_score_query_count",
        "attack_score_query_time_sec",
        "attack_score_queries_per_success_oracle",
        "adv_samples_path",
        "metrics_path",
        "baseline_summary_path",
        "x_adv_pre_rows",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _run_stage2_attack_slice_evals(
    *,
    cfg: dict[str, Any],
    settings: Stage2Settings,
    seed: int,
    device: Any,
    out_dir: Any,
    preprocessor: DatasetPreprocessor,
    traffic_schema: Any,
    bundle: Any,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_ben: np.ndarray,
    x_mal: np.ndarray,
    diffusion_bundle: Any,
    surrogate: Any,
    oracle: Any,
    train_runtime_sec: float,
    stage2_mode: str,
    generator_backbone: str,
    guidance_mode: str,
    feature_names: list[str],
) -> list[dict[str, Any]]:
    stage2_cfg = cfg.get("stage2", {}) or {}
    if not bool(stage2_cfg.get("attack_slice_eval_enabled", True)):
        return []
    attack_labels = _discover_eval_attack_labels(bundle)
    max_labels = int(stage2_cfg.get("attack_slice_eval_max_labels", 0) or 0)
    if max_labels > 0:
        attack_labels = attack_labels[:max_labels]
    if not attack_labels:
        return []

    rows: list[dict[str, Any]] = []
    for attack_label in attack_labels:
        attack_out_dir = out_dir / "attack_eval" / attack_label.replace("/", "_").replace("\\", "_").replace(" ", "_")
        attack_out_dir.mkdir(parents=True, exist_ok=True)
        attack_metrics_payload: dict[str, Any] = {
            "attack_eval_scope": "posthoc_attack_slice",
            "attack_type": attack_label,
        }
        x_ben_eval_pool, x_mal_eval_pool = _select_stage2_eval_pools(
            bundle=bundle,
            x_train=x_train,
            y_train=y_train,
            metrics_payload=attack_metrics_payload,
            eval_attack_label=attack_label,
        )
        eval_result = _run_stage2_eval_metrics(
            cfg=cfg,
            stage2_cfg=cfg["stage2"],
            settings=settings,
            seed=seed,
            device=device,
            out_dir=attack_out_dir,
            preprocessor=preprocessor,
            traffic_schema=traffic_schema,
            x_train=x_train,
            y_train=y_train,
            x_ben=x_ben,
            x_mal=x_mal,
            x_ben_eval_pool=x_ben_eval_pool,
            x_mal_eval_pool=x_mal_eval_pool,
            diffusion_bundle=diffusion_bundle,
            surrogate=surrogate,
            oracle=oracle,
            metrics_payload=attack_metrics_payload,
            train_runtime_sec=train_runtime_sec,
            stage2_mode=stage2_mode,
            generator_backbone=generator_backbone,
            guidance_mode=guidance_mode,
            feature_names=feature_names,
        )
        rows.append(
            {
                "attack_type": attack_label,
                "out_dir": str(attack_out_dir),
                "stage2_eval_split": attack_metrics_payload.get("stage2_eval_split", ""),
                "stage2_eval_attack_rows": attack_metrics_payload.get("stage2_eval_attack_rows", ""),
                "stage2_eval_attack_filtered": attack_metrics_payload.get("stage2_eval_attack_filtered", ""),
                "asr_surrogate": attack_metrics_payload.get("asr_surrogate", ""),
                "asr_oracle": attack_metrics_payload.get("asr_oracle", ""),
                "norm_FFD": attack_metrics_payload.get("norm_FFD", ""),
                "norm_SWD": attack_metrics_payload.get("norm_SWD", ""),
                "norm_AdvToMal_L2": attack_metrics_payload.get("norm_AdvToMal_L2", ""),
                "sample_end_to_end_time_sec": attack_metrics_payload.get("sample_end_to_end_time_sec", ""),
                "sample_end_to_end_samples_per_sec": attack_metrics_payload.get(
                    "sample_end_to_end_samples_per_sec", ""
                ),
                "attack_score_query_count": attack_metrics_payload.get("attack_score_query_count", ""),
                "attack_score_query_time_sec": attack_metrics_payload.get("attack_score_query_time_sec", ""),
                "attack_score_queries_per_success_oracle": attack_metrics_payload.get(
                    "attack_score_queries_per_success_oracle",
                    "",
                ),
                "adv_samples_path": str(attack_out_dir / "adv_samples.npz") if settings.save_samples else "",
                "metrics_path": str(attack_out_dir / "metrics.json"),
                "baseline_summary_path": str(attack_out_dir / "baseline_summary.csv"),
                "x_adv_pre_rows": int(eval_result["x_adv_pre"].shape[0]),
            }
        )

    _write_stage2_attack_eval_index(out_dir, rows)
    return rows


def _select_stage2_eval_pools(
    *,
    bundle: Any,
    x_train: np.ndarray,
    y_train: np.ndarray,
    metrics_payload: dict[str, Any],
    eval_attack_label: str,
) -> tuple[np.ndarray, np.ndarray]:
    x_test = np.asarray(getattr(bundle, "x_test", np.empty((0, x_train.shape[1]), dtype=x_train.dtype)))
    y_test = np.asarray(getattr(bundle, "y_test", np.empty((0,), dtype=y_train.dtype)))
    has_test = x_test.shape[0] > 0 and y_test.shape[0] == x_test.shape[0]
    x_eval_source = x_test if has_test else x_train
    y_eval_source = y_test if has_test else y_train
    raw_eval_labels = getattr(bundle, "raw_y_test", None) if has_test else getattr(bundle, "raw_y_train", None)

    x_ben_eval_pool = x_eval_source[y_eval_source == 0]
    x_mal_eval_pool = x_eval_source[y_eval_source == 1]
    metrics_payload["stage2_eval_split"] = "test" if has_test else "train_fallback"
    metrics_payload["stage2_eval_attack_label"] = eval_attack_label

    if raw_eval_labels is not None and len(raw_eval_labels) == len(y_eval_source) and eval_attack_label:
        raw_eval_labels = np.asarray(raw_eval_labels, dtype=object)
        label_mask = np.array([str(value).strip() == eval_attack_label for value in raw_eval_labels], dtype=bool)
        attack_mask = np.logical_and(y_eval_source == 1, label_mask)
        if np.any(attack_mask):
            x_mal_eval_pool = x_eval_source[attack_mask]
            metrics_payload["stage2_eval_attack_rows"] = int(x_mal_eval_pool.shape[0])
            metrics_payload["stage2_eval_attack_filtered"] = True
        else:
            metrics_payload["stage2_eval_attack_rows"] = 0
            metrics_payload["stage2_eval_attack_filtered"] = False
            metrics_payload["stage2_eval_attack_filter_fallback"] = "no_matching_test_rows"
    else:
        metrics_payload["stage2_eval_attack_filtered"] = False

    if x_ben_eval_pool.shape[0] == 0:
        x_ben_eval_pool = x_train[y_train == 0]
        metrics_payload["stage2_eval_benign_fallback"] = True
    if x_mal_eval_pool.shape[0] == 0:
        x_mal_eval_pool = x_train[y_train == 1]
        metrics_payload["stage2_eval_malicious_fallback"] = True
    return x_ben_eval_pool, x_mal_eval_pool


def _build_training_metrics_payload(
    *,
    diffusion_bundle: Any,
    generator_backbone: str,
    guidance_mode: str,
    train_runtime_sec: float,
    schema_counts: dict[str, int],
) -> tuple[dict[str, Any], list[dict[str, Any]] | None]:
    metrics_payload: dict[str, Any] = {
        "groups": diffusion_bundle.groups,
        "generator_backbone": generator_backbone,
        "surrogate_guidance_mode": guidance_mode,
        "train_time_sec": float(train_runtime_sec),
        **schema_counts,
    }
    if getattr(diffusion_bundle, "best_epoch", None) is not None:
        metrics_payload["train_selection_best_epoch"] = int(diffusion_bundle.best_epoch)
    if getattr(diffusion_bundle, "best_score", None) is not None:
        metrics_payload["train_selection_best_score"] = float(diffusion_bundle.best_score)
    return metrics_payload, getattr(diffusion_bundle, "train_log", None)


def _persist_training_log(
    *,
    out_dir: Any,
    train_log: list[dict[str, Any]] | None,
    metrics_payload: dict[str, Any],
) -> None:
    if not train_log:
        return
    train_csv = out_dir / "stage2_train_metrics.csv"
    save_training_log_csv(train_csv, train_log)
    metrics_payload["train_selection_log_path"] = str(train_csv)


def _finalize_stage2_outputs(
    *,
    config_path: str,
    out_dir: Any,
    oracle_name: str,
    stage2_mode: str,
    generator_backbone: str,
    diffusion_bundle: Any,
    feature_names: list[str],
    x_train: np.ndarray,
    x_ben: np.ndarray,
    x_mal: np.ndarray,
    train_log: list[dict[str, Any]] | None,
    metrics_payload: dict[str, Any],
    settings: Stage2Settings,
    x_adv_pre: np.ndarray | None,
    x_adv_norm: np.ndarray | None,
    x_ben_pre: np.ndarray | None,
    x_mal_pre: np.ndarray | None,
) -> None:
    save_stage2_state(
        stage2_mode=stage2_mode,
        generator_backbone=generator_backbone,
        diffusion_bundle=diffusion_bundle,
        feature_names=feature_names,
        out_path=out_dir / "stage2.pt",
    )
    print(f"[Stage2] saved to {out_dir}")
    save_stage2_manifest(
        out_dir=out_dir,
        config_path=config_path,
        oracle_name=oracle_name,
        x_train=x_train,
        x_ben=x_ben,
        x_mal=x_mal,
        stage2_mode=stage2_mode,
        train_log=train_log,
        metrics_payload=metrics_payload,
        settings=settings,
        x_adv_pre=x_adv_pre,
        x_adv_norm=x_adv_norm,
        x_ben_pre=x_ben_pre,
        x_mal_pre=x_mal_pre,
    )


def _run_stage2_eval_metrics(
    *,
    cfg: dict[str, Any],
    stage2_cfg: dict[str, Any],
    settings: Stage2Settings,
    seed: int,
    device: Any,
    out_dir: Any,
    preprocessor: DatasetPreprocessor,
    traffic_schema: Any,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_ben: np.ndarray,
    x_mal: np.ndarray,
    x_ben_eval_pool: np.ndarray | None = None,
    x_mal_eval_pool: np.ndarray | None = None,
    diffusion_bundle: Any,
    surrogate: Any,
    oracle: Any,
    metrics_payload: dict[str, Any],
    train_runtime_sec: float,
    stage2_mode: str,
    generator_backbone: str,
    guidance_mode: str,
    feature_names: list[str],
) -> dict[str, Any]:
    predictor_setup = build_stage2_predictor_setup(
        stage2_cfg=stage2_cfg,
        surrogate=surrogate,
        oracle=oracle,
        device=device,
        batch_size=settings.sample_batch_size,
        metrics_payload=metrics_payload,
    )
    predictors = predictor_setup.predictors
    attack_query_oracle = predictor_setup.attack_query_oracle

    eval_inputs = prepare_stage2_eval_inputs(
        cfg=cfg,
        settings=settings,
        seed=seed,
        x_ben=x_ben if x_ben_eval_pool is None else x_ben_eval_pool,
        x_mal=x_mal if x_mal_eval_pool is None else x_mal_eval_pool,
        diffusion_bundle=diffusion_bundle,
        preprocessor=preprocessor,
    )
    x_mal_eval = eval_inputs.x_mal_eval
    x_ben_eval = eval_inputs.x_ben_eval
    denorm_mean = eval_inputs.denorm_mean
    denorm_std = eval_inputs.denorm_std
    sample_denorm = eval_inputs.sample_denorm
    x_ben_norm = eval_inputs.x_ben_norm
    x_ben_pre = eval_inputs.x_ben_pre
    x_mal_norm = eval_inputs.x_mal_norm
    x_mal_pre = eval_inputs.x_mal_pre
    x_adv_denorm = eval_inputs.x_adv_denorm
    x_ben_denorm = eval_inputs.x_ben_denorm
    x_mal_denorm = eval_inputs.x_mal_denorm
    eval_denorm = eval_inputs.eval_denorm
    pre_min = eval_inputs.pre_min
    pre_max = eval_inputs.pre_max

    constraint_inputs = prepare_stage2_constraint_inputs(
        cfg=cfg,
        diffusion_bundle=diffusion_bundle,
        preprocessor=preprocessor,
        feature_names=feature_names,
        x_ben=x_ben,
        x_ben_norm=x_ben_norm,
        metrics_payload=metrics_payload,
    )
    norm_bounds_min = constraint_inputs.norm_bounds_min
    norm_bounds_max = constraint_inputs.norm_bounds_max
    norm_nonneg = constraint_inputs.norm_nonneg

    sample_with_alpha = build_stage2_sampler(
        stage2_mode=stage2_mode,
        generator_backbone=generator_backbone,
        settings=settings,
        diffusion_bundle=diffusion_bundle,
        surrogate=surrogate,
        device=device,
        guidance_mode=guidance_mode,
        benign_pool=x_ben,
    )
    eval_helper = build_stage2_eval_helper(
        cfg=cfg,
        seed=seed,
        preprocessor=preprocessor,
        bundle_feature_names=feature_names,
        diffusion_bundle=diffusion_bundle,
        traffic_schema=traffic_schema,
        oracle=oracle,
        eval_inputs=eval_inputs,
        constraint_inputs=constraint_inputs,
        predictors=predictors,
    )

    pareto_cfg = cfg["stage2"].get("pareto_eval", {}) or {}
    selection = run_stage2_candidate_selection(
        pareto_cfg=pareto_cfg,
        eval_helper=eval_helper,
        sample_with_alpha=sample_with_alpha,
        x_mal_eval=x_mal_eval,
        x_ben_norm=x_ben_norm,
        attack_query_oracle=attack_query_oracle,
        metrics_payload=metrics_payload,
        default_alpha=settings.mal_anchor_alpha,
    )
    sample_execution = execute_stage2_sample(
        sample_with_alpha=sample_with_alpha,
        eval_helper=eval_helper,
        attack_query_oracle=attack_query_oracle,
        metrics_payload=metrics_payload,
        x_mal_eval=x_mal_eval,
        selection=selection,
    )
    x_adv_pre = sample_execution.x_adv_pre
    x_adv_norm = sample_execution.x_adv_norm
    query_stats = sample_execution.query_stats
    total_runtime_sec = float(train_runtime_sec + selection.selection_runtime_sec + sample_execution.sample_runtime_sec)
    metrics_payload["sample_end_to_end_time_sec"] = total_runtime_sec
    metrics_payload["sample_end_to_end_samples_per_sec"] = (
        float(x_mal_eval.shape[0] / total_runtime_sec) if total_runtime_sec > 0.0 else float("nan")
    )
    record_sample_runtime(
        metrics_payload=metrics_payload,
        sample_count=int(x_mal_eval.shape[0]),
        runtime_sec=sample_execution.sample_runtime_sec,
        pull_alpha=eval_inputs.pull_alpha,
        pull_k=eval_inputs.pull_k,
        moment_alpha=eval_inputs.moment_alpha,
        moment_std_floor=eval_inputs.moment_std_floor,
        post_clip_norm_range=settings.post_clip_norm_range,
    )
    if eval_denorm:
        x_adv_pre = np.maximum(np.minimum(x_adv_pre, pre_max), pre_min)
        x_adv_denorm = preprocessor.inverse_transform(x_adv_pre)

    update_attack_metrics(
        metrics_payload=metrics_payload,
        surrogate=surrogate,
        oracle=oracle,
        surrogate_predict_probs=lambda x, _batch_size: predictors.surrogate_predict_probs(x),
        oracle_predict_probs=lambda x, _batch_size: predictors.oracle_predict_probs(x),
        x_adv_pre=x_adv_pre,
        x_mal_pre=x_mal_pre,
        batch_size=settings.sample_batch_size,
        sample_runtime_sec=total_runtime_sec,
    )
    success_orc = float(metrics_payload.get("asr_oracle", float("nan"))) * max(1, int(x_mal_eval.shape[0]))
    success_sur = float(metrics_payload.get("asr_surrogate", float("nan"))) * max(1, int(x_mal_eval.shape[0]))
    qcnt = float(query_stats.query_count)
    metrics_payload["attack_score_queries_per_success_oracle"] = (
        qcnt / success_orc if np.isfinite(success_orc) and success_orc > 1.0e-12 else float("nan")
    )
    metrics_payload["attack_score_queries_per_success_surrogate"] = (
        qcnt / success_sur if np.isfinite(success_sur) and success_sur > 1.0e-12 else float("nan")
    )
    update_sample_distribution_summary(
        metrics_payload=metrics_payload,
        feature_names=feature_names,
        x_adv_norm=x_adv_norm,
        x_ben_norm=x_ben_norm,
        x_mal_norm=x_mal_norm,
        lo=diffusion_bundle.ben_stats.get("min"),
        hi=diffusion_bundle.ben_stats.get("max"),
    )
    metrics_result = compute_stage2_distribution_metrics(
        metrics_payload=metrics_payload,
        cfg=cfg,
        feature_names=feature_names,
        seed=seed,
        x_ben_norm=x_ben_norm,
        x_adv_norm=x_adv_norm,
        x_adv_pre=x_adv_pre,
        x_mal_pre=x_mal_pre,
        x_ben_denorm=x_ben_denorm if eval_denorm else None,
        x_adv_denorm=x_adv_denorm if eval_denorm else None,
        x_mal_denorm=x_mal_denorm if eval_denorm else None,
        norm_bounds_min=norm_bounds_min,
        norm_bounds_max=norm_bounds_max,
        norm_nonneg=norm_nonneg,
    )
    print_stage2_metric_tables(
        metrics_payload=metrics_payload,
        metrics_norm=metrics_result.metrics_norm,
        adv_ben_l2=metrics_result.adv_ben_l2,
        adv_mal_l2=metrics_result.adv_mal_l2,
        eval_denorm=eval_denorm,
        metrics_denorm=metrics_result.metrics_denorm if eval_denorm else None,
        adv_ben_l2_denorm=metrics_result.adv_ben_l2_denorm if eval_denorm else None,
        adv_mal_l2_denorm=metrics_result.adv_mal_l2_denorm if eval_denorm else None,
    )
    run_stage2_baselines(
        cfg=cfg,
        bundle_feature_names=feature_names,
        out_dir=out_dir,
        device=device,
        seed=seed,
        surrogate=surrogate,
        oracle=oracle,
        y_train=y_train,
        x_train=x_train,
        x_ben_pre=x_ben_pre,
        x_mal_pre=x_mal_pre,
        x_ben_norm=x_ben_norm,
        x_mal_norm=x_mal_norm,
        denorm_mean=denorm_mean,
        denorm_std=denorm_std,
        norm_bounds_min=norm_bounds_min,
        norm_bounds_max=norm_bounds_max,
        norm_nonneg=norm_nonneg,
        metrics_payload=metrics_payload,
        sample_denorm=sample_denorm,
        mal_benign_rate=metrics_payload.get("mal_benign_rate"),
        mal_benign_rate_oracle=metrics_payload.get("mal_benign_rate_oracle"),
        attack_score_fn=predictors.attack_score_fn,
        surrogate_predict_probs=lambda x, _batch_size: predictors.surrogate_predict_probs(x),
        oracle_predict_probs=lambda x, _batch_size: predictors.oracle_predict_probs(x),
        postprocess_adv=eval_helper.postprocess_adv,
    )
    run_stage2_pareto_eval(
        cfg=cfg,
        seed=seed,
        out_dir=out_dir,
        x_mal_eval=x_mal_eval,
        x_ben_eval=x_ben_eval,
        denorm_mean=denorm_mean,
        denorm_std=denorm_std,
        eval_helper=eval_helper,
        sample_with_alpha=sample_with_alpha,
        metrics_payload=metrics_payload,
        save_pareto_front=save_pareto_front,
    )
    payload = build_stage2_artifact_payload(
        x_adv_pre=x_adv_pre,
        x_adv_norm=x_adv_norm,
        x_ben_norm=x_ben_norm,
        x_mal_norm=x_mal_norm,
        x_ben_pre=x_ben_pre,
        x_mal_pre=x_mal_pre,
        denorm_mean=denorm_mean,
        denorm_std=denorm_std,
        feature_names=feature_names,
        x_adv_denorm=x_adv_denorm,
        x_ben_denorm=x_ben_denorm,
        x_mal_denorm=x_mal_denorm,
    )
    if settings.save_samples:
        np.savez_compressed(out_dir / "adv_samples.npz", **payload)
    if settings.save_intermediate_results:
        np.savez_compressed(out_dir / "intermediate_results.npz", **payload)
    persist_stage2_metrics(metrics_payload, out_dir)
    return {
        "x_adv_pre": x_adv_pre,
        "x_adv_norm": x_adv_norm,
        "x_ben_pre": x_ben_pre,
        "x_mal_pre": x_mal_pre,
    }


def main(config_path: str) -> None:
    runtime = load_stage_runtime(config_path, "stage2")
    cfg = runtime.cfg
    seed = runtime.seed
    device = runtime.device
    out_dir = runtime.out_dir
    stage2_cfg = runtime.stage_cfg
    settings = Stage2Settings.from_cfg(stage2_cfg)

    data_ctx = load_data_context(cfg, seed)
    bundle = data_ctx.bundle
    preprocessor = DatasetPreprocessor.from_bundle(bundle)
    traffic_schema = infer_traffic_feature_schema(
        bundle.feature_names,
        preprocessor.inverse_transform(data_ctx.bundle.x_train),
    )

    x_train = bundle.x_train
    y_train = bundle.y_train
    x_ben = x_train[y_train == 0]
    x_mal = x_train[y_train == 1]
    eval_attack_label = _resolve_eval_attack_label(cfg)

    schema_counts = _schema_counts(traffic_schema)

    # Load surrogate from Stage1; training without it makes Stage2 constraints unreliable.
    oracle_name = settings.oracle_name
    require_stage1 = settings.require_stage1
    artifacts = load_stage2_artifacts(
        cfg=cfg,
        oracle_name=oracle_name,
        require_stage1=bool(require_stage1),
        x_train=bundle.x_train,
        y_train=bundle.y_train,
        x_val=bundle.x_val,
        y_val=bundle.y_val,
        feature_names=list(bundle.feature_names),
        device=device,
        seed=seed,
    )
    surrogate = artifacts.surrogate
    oracle = artifacts.oracle
    if not artifacts.checkpoint_path.exists():
        if require_stage1:
            raise RuntimeError(
                f"[Stage2] Stage1 checkpoint required but not found at {artifacts.checkpoint_path}. "
                "Set require_stage1=false to use an untrained surrogate (not recommended for paper results)."
            )
        print("[Stage2][Warn] Stage1 checkpoint missing; using an untrained surrogate.")

    stage2_mode = settings.mode
    generator_backbone = settings.generator_backbone
    guidance_mode = settings.guidance_mode
    train_start = time.perf_counter()
    diffusion_bundle = train_stage2_generator(
        cfg=cfg,
        settings=settings,
        x_ben=x_ben,
        x_mal=x_mal,
        feature_names=list(bundle.feature_names),
        surrogate=surrogate,
        device=device,
    )
    train_runtime_sec = time.perf_counter() - train_start

    save_config(str(runtime.config_path), out_dir)
    metrics_payload, train_log = _build_training_metrics_payload(
        diffusion_bundle=diffusion_bundle,
        generator_backbone=generator_backbone,
        guidance_mode=guidance_mode,
        train_runtime_sec=train_runtime_sec,
        schema_counts=schema_counts,
    )
    _persist_training_log(
        out_dir=out_dir,
        train_log=train_log,
        metrics_payload=metrics_payload,
    )

    x_adv_pre: np.ndarray | None = None
    x_adv_norm: np.ndarray | None = None
    x_ben_pre: np.ndarray | None = None
    x_mal_pre: np.ndarray | None = None

    if settings.eval_metrics:
        x_ben_eval_pool, x_mal_eval_pool = _select_stage2_eval_pools(
            bundle=bundle,
            x_train=x_train,
            y_train=y_train,
            metrics_payload=metrics_payload,
            eval_attack_label=eval_attack_label,
        )
        eval_result = _run_stage2_eval_metrics(
            cfg=cfg,
            stage2_cfg=stage2_cfg,
            settings=settings,
            seed=seed,
            device=device,
            out_dir=out_dir,
            preprocessor=preprocessor,
            traffic_schema=traffic_schema,
            x_train=x_train,
            y_train=y_train,
            x_ben=x_ben,
            x_mal=x_mal,
            x_ben_eval_pool=x_ben_eval_pool,
            x_mal_eval_pool=x_mal_eval_pool,
            diffusion_bundle=diffusion_bundle,
            surrogate=surrogate,
            oracle=oracle,
            metrics_payload=metrics_payload,
            train_runtime_sec=train_runtime_sec,
            stage2_mode=stage2_mode,
            generator_backbone=generator_backbone,
            guidance_mode=guidance_mode,
            feature_names=list(bundle.feature_names),
        )
        x_adv_pre = eval_result["x_adv_pre"]
        x_adv_norm = eval_result["x_adv_norm"]
        x_ben_pre = eval_result["x_ben_pre"]
        x_mal_pre = eval_result["x_mal_pre"]
        attack_eval_rows = _run_stage2_attack_slice_evals(
            cfg=cfg,
            settings=settings,
            seed=seed,
            device=device,
            out_dir=out_dir,
            preprocessor=preprocessor,
            traffic_schema=traffic_schema,
            bundle=bundle,
            x_train=x_train,
            y_train=y_train,
            x_ben=x_ben,
            x_mal=x_mal,
            diffusion_bundle=diffusion_bundle,
            surrogate=surrogate,
            oracle=oracle,
            train_runtime_sec=train_runtime_sec,
            stage2_mode=stage2_mode,
            generator_backbone=generator_backbone,
            guidance_mode=guidance_mode,
            feature_names=list(bundle.feature_names),
        )
        if attack_eval_rows:
            metrics_payload["attack_eval_count"] = int(len(attack_eval_rows))
            metrics_payload["attack_eval_index_path"] = str(out_dir / "attack_eval_index.csv")
    _finalize_stage2_outputs(
        config_path=config_path,
        out_dir=out_dir,
        oracle_name=oracle_name,
        stage2_mode=stage2_mode,
        generator_backbone=generator_backbone,
        diffusion_bundle=diffusion_bundle,
        feature_names=list(bundle.feature_names),
        x_train=x_train,
        x_ben=x_ben,
        x_mal=x_mal,
        train_log=train_log,
        metrics_payload=metrics_payload,
        settings=settings,
        x_adv_pre=x_adv_pre,
        x_adv_norm=x_adv_norm,
        x_ben_pre=x_ben_pre,
        x_mal_pre=x_mal_pre,
    )


if __name__ == "__main__":
    cfg_path = os.environ.get("RDSYNTH_CONFIG", "configs/demo.yaml")
    main(cfg_path)
