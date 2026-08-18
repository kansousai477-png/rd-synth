from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from rdsynth.utils.artifacts import save_metrics, save_metrics_csv, save_records_csv
from rdsynth.utils.metrics_stage2 import compute_stage2_metrics, nearest_reference_distance, paired_sample_l2
from rdsynth.utils.paper_metrics import add_paper_attack_metrics

SELECTED_CANDIDATE_FIELDS = (
    "asr_oracle",
    "asr_surrogate",
    "selection_score",
    "remapability_score",
    "remap_projection_penalty",
    "remap_clip_penalty",
    "remap_center_penalty",
    "stage3_loop_decision_pcap_sanity_score",
    "stage3_closed_loop_score",
    "ffd",
    "swd",
    "c2st_auc",
    "adv_to_ben_l2",
    "adv_to_mal_l2",
    "asr_oracle",
    "asr_surrogate",
)


@dataclass(frozen=True)
class Stage2DistributionMetricsResult:
    metrics_norm: Any
    adv_ben_l2: float
    adv_mal_l2: float
    metrics_denorm: Any | None = None
    adv_ben_l2_denorm: float | None = None
    adv_mal_l2_denorm: float | None = None


def _safe_console_text(text: str) -> str:
    encoding = sys.stdout.encoding or "utf-8"
    return text.encode(encoding, errors="replace").decode(encoding, errors="replace")


def _safe_print(text: str) -> None:
    rendered = _safe_console_text(text)
    try:
        print(rendered)
    except OSError:
        fallback = getattr(sys, "__stdout__", None)
        if fallback is None:
            return
        try:
            fallback.write(rendered + "\n")
            fallback.flush()
        except OSError:
            pass


def _invoke_predictor(
    predictor: Callable[..., tuple[np.ndarray, np.ndarray | None]],
    x: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    try:
        return predictor(x, batch_size)
    except TypeError:
        return predictor(x)


def update_selected_candidate_metrics(
    metrics_payload: dict[str, Any],
    selected_row: dict[str, float] | None,
) -> None:
    if selected_row is None:
        return
    for key in SELECTED_CANDIDATE_FIELDS:
        if key in selected_row:
            metrics_payload[f"selected_candidate_{key}"] = float(selected_row[key])


def record_sample_statistics(
    *,
    metrics_payload: dict[str, Any],
    sample_for_stats: np.ndarray,
    selected_alpha: float,
) -> tuple[float, float]:
    nan_rate = float(np.mean(np.isnan(sample_for_stats)))
    inf_rate = float(np.mean(np.isinf(sample_for_stats)))
    metrics_payload["sample_nan_rate"] = nan_rate
    metrics_payload["sample_inf_rate"] = inf_rate
    metrics_payload["sample_min"] = float(np.nanmin(sample_for_stats))
    metrics_payload["sample_max"] = float(np.nanmax(sample_for_stats))
    metrics_payload["mal_anchor_alpha"] = float(selected_alpha)
    return nan_rate, inf_rate


def record_sample_runtime(
    *,
    metrics_payload: dict[str, Any],
    sample_count: int,
    runtime_sec: float,
    pull_alpha: float,
    pull_k: int,
    moment_alpha: float,
    moment_std_floor: float,
    post_clip_norm_range: bool,
) -> None:
    metrics_payload["sample_generation_time_sec"] = runtime_sec
    metrics_payload["sample_count"] = int(sample_count)
    metrics_payload["sample_generation_samples_per_sec"] = (
        float(sample_count / runtime_sec) if runtime_sec > 0.0 else float("nan")
    )
    if pull_alpha > 0.0 and pull_k > 0:
        metrics_payload["sample_pullback_alpha"] = float(pull_alpha)
        metrics_payload["sample_pullback_k"] = int(pull_k)
    if moment_alpha > 0.0:
        metrics_payload["sample_moment_alpha"] = float(moment_alpha)
        metrics_payload["sample_moment_std_floor"] = float(moment_std_floor)
    metrics_payload["post_clip_norm_range"] = bool(post_clip_norm_range)


def update_attack_metrics(
    *,
    metrics_payload: dict[str, Any],
    surrogate: Any,
    oracle: Any,
    surrogate_predict_probs: Callable[..., tuple[np.ndarray, np.ndarray]],
    oracle_predict_probs: Callable[..., tuple[np.ndarray, np.ndarray | None]],
    x_adv_pre: np.ndarray,
    x_mal_pre: np.ndarray,
    batch_size: int,
    sample_runtime_sec: float,
) -> None:
    if surrogate is not None:
        adv_preds, adv_probs = _invoke_predictor(surrogate_predict_probs, x_adv_pre, batch_size)
        mal_preds, mal_probs = _invoke_predictor(surrogate_predict_probs, x_mal_pre, batch_size)
        metrics_payload["asr_surrogate"] = float(np.mean(adv_preds == 0))
        metrics_payload["adv_prob_malicious_mean"] = float(np.mean(adv_probs[:, 1]))
        metrics_payload["mal_benign_rate"] = float(np.mean(mal_preds == 0))
        metrics_payload["mal_prob_malicious_mean"] = float(np.mean(mal_probs[:, 1]))
        add_paper_attack_metrics(
            metrics_payload,
            prefix="surrogate_",
            asr=metrics_payload["asr_surrogate"],
            orig_benign_rate=metrics_payload["mal_benign_rate"],
            adv_prob_malicious=metrics_payload["adv_prob_malicious_mean"],
            runtime_sec=sample_runtime_sec,
        )

    if oracle is None:
        return

    o_adv_preds, o_adv_probs = _invoke_predictor(oracle_predict_probs, x_adv_pre, batch_size)
    o_mal_preds, o_mal_probs = _invoke_predictor(oracle_predict_probs, x_mal_pre, batch_size)
    if o_adv_preds.size:
        metrics_payload["asr_oracle"] = float(np.mean(o_adv_preds == 0))
    if o_adv_probs is not None and o_adv_probs.size:
        metrics_payload["adv_prob_malicious_mean_oracle"] = float(np.mean(o_adv_probs[:, 1]))
    if o_mal_preds.size:
        metrics_payload["mal_benign_rate_oracle"] = float(np.mean(o_mal_preds == 0))
    if o_mal_probs is not None and o_mal_probs.size:
        metrics_payload["mal_prob_malicious_mean_oracle"] = float(np.mean(o_mal_probs[:, 1]))
    add_paper_attack_metrics(
        metrics_payload,
        prefix="oracle_",
        asr=metrics_payload.get("asr_oracle"),
        orig_benign_rate=metrics_payload.get("mal_benign_rate_oracle"),
        adv_prob_malicious=metrics_payload.get("adv_prob_malicious_mean_oracle"),
        runtime_sec=sample_runtime_sec,
    )


def update_sample_distribution_summary(
    *,
    metrics_payload: dict[str, Any],
    feature_names: list[str],
    x_adv_norm: np.ndarray,
    x_ben_norm: np.ndarray,
    x_mal_norm: np.ndarray,
    lo: np.ndarray | None,
    hi: np.ndarray | None,
) -> None:
    if lo is not None and hi is not None:
        out_of_range = np.logical_or(x_adv_norm < np.asarray(lo), x_adv_norm > np.asarray(hi))
        range_rate = float(np.mean(out_of_range))
        metrics_payload["sample_range_violation_rate"] = range_rate
        _safe_print(f"[Stage2] sample_range_violation_rate={range_rate:.6f}")

    adv_norm_mean = np.mean(x_adv_norm, axis=0)
    adv_norm_std = np.std(x_adv_norm, axis=0)
    metrics_payload["sample_norm_mean_abs"] = float(np.mean(np.abs(adv_norm_mean)))
    metrics_payload["sample_norm_std_mean"] = float(np.mean(adv_norm_std))
    metrics_payload["sample_norm_std_min"] = float(np.min(adv_norm_std))
    metrics_payload["sample_norm_std_max"] = float(np.max(adv_norm_std))
    _safe_print(
        f"[Stage2] sample_norm_mean_abs={metrics_payload['sample_norm_mean_abs']:.6f} "
        f"sample_norm_std_mean={metrics_payload['sample_norm_std_mean']:.6f} "
        f"sample_norm_std_min={metrics_payload['sample_norm_std_min']:.6f} "
        f"sample_norm_std_max={metrics_payload['sample_norm_std_max']:.6f}"
    )

    iat_idx = [i for i, name in enumerate(feature_names) if "iat" in name.lower()]
    if not iat_idx:
        return
    adv_iat_mean = np.mean(x_adv_norm[:, iat_idx], axis=0)
    ben_iat_mean = np.mean(x_ben_norm[:, iat_idx], axis=0)
    mal_iat_mean = np.mean(x_mal_norm[:, iat_idx], axis=0)
    adv_iat_std = np.std(x_adv_norm[:, iat_idx], axis=0)
    ben_iat_std = np.std(x_ben_norm[:, iat_idx], axis=0)
    mal_iat_std = np.std(x_mal_norm[:, iat_idx], axis=0)
    metrics_payload["iat_adv_ben_mean_abs"] = float(np.mean(np.abs(adv_iat_mean - ben_iat_mean)))
    metrics_payload["iat_adv_mal_mean_abs"] = float(np.mean(np.abs(adv_iat_mean - mal_iat_mean)))
    metrics_payload["iat_adv_ben_std_abs"] = float(np.mean(np.abs(adv_iat_std - ben_iat_std)))
    metrics_payload["iat_adv_mal_std_abs"] = float(np.mean(np.abs(adv_iat_std - mal_iat_std)))
    _safe_print(
        "[Stage2] iat_shift"
        f" adv_ben_mean_abs={metrics_payload['iat_adv_ben_mean_abs']:.6f}"
        f" adv_mal_mean_abs={metrics_payload['iat_adv_mal_mean_abs']:.6f}"
        f" adv_ben_std_abs={metrics_payload['iat_adv_ben_std_abs']:.6f}"
        f" adv_mal_std_abs={metrics_payload['iat_adv_mal_std_abs']:.6f}"
    )


def compute_stage2_distribution_metrics(
    *,
    metrics_payload: dict[str, Any],
    cfg: dict[str, Any],
    feature_names: list[str],
    seed: int,
    x_ben_norm: np.ndarray,
    x_adv_norm: np.ndarray,
    x_adv_pre: np.ndarray,
    x_mal_pre: np.ndarray,
    x_ben_denorm: np.ndarray | None = None,
    x_adv_denorm: np.ndarray | None = None,
    x_mal_denorm: np.ndarray | None = None,
    norm_bounds_min: np.ndarray | None = None,
    norm_bounds_max: np.ndarray | None = None,
    norm_nonneg: np.ndarray | None = None,
) -> Stage2DistributionMetricsResult:
    metrics_norm = compute_stage2_metrics(
        x_ben_norm,
        x_adv_norm,
        feature_names=feature_names,
        max_real=cfg["stage2"].get("metrics_max_real", 2000),
        max_gen=cfg["stage2"].get("metrics_max_gen", 2000),
        seed=seed,
        bounds_min=norm_bounds_min,
        bounds_max=norm_bounds_max,
        nonneg_mask=norm_nonneg,
    )
    adv_ben_l2 = nearest_reference_distance(x_adv_norm, x_ben_norm)
    adv_mal_l2 = paired_sample_l2(x_adv_pre, x_mal_pre)

    result = Stage2DistributionMetricsResult(
        metrics_norm=metrics_norm,
        adv_ben_l2=adv_ben_l2,
        adv_mal_l2=adv_mal_l2,
    )
    metrics_payload.update({f"norm_{k}": v for k, v in metrics_norm.as_dict().items()})
    metrics_payload["norm_AdvToBen_L2"] = adv_ben_l2
    metrics_payload["norm_AdvToMal_L2"] = adv_mal_l2

    add_paper_attack_metrics(
        metrics_payload,
        prefix="surrogate_",
        asr=metrics_payload.get("asr_surrogate"),
        orig_benign_rate=metrics_payload.get("mal_benign_rate"),
        adv_prob_malicious=metrics_payload.get("adv_prob_malicious_mean"),
        ffd=metrics_payload.get("norm_FFD"),
        swd=metrics_payload.get("norm_SWD"),
        c2st_auc=metrics_payload.get("norm_C2ST-AUC"),
        c2st_acc=metrics_payload.get("norm_C2ST-Acc"),
        adv_to_ben_l2=metrics_payload.get("norm_AdvToBen_L2"),
        adv_to_mal_l2=metrics_payload.get("norm_AdvToMal_L2"),
        runtime_sec=metrics_payload.get("sample_generation_time_sec"),
    )
    add_paper_attack_metrics(
        metrics_payload,
        prefix="oracle_",
        asr=metrics_payload.get("asr_oracle"),
        orig_benign_rate=metrics_payload.get("mal_benign_rate_oracle"),
        adv_prob_malicious=metrics_payload.get("adv_prob_malicious_mean_oracle"),
        ffd=metrics_payload.get("norm_FFD"),
        swd=metrics_payload.get("norm_SWD"),
        c2st_auc=metrics_payload.get("norm_C2ST-AUC"),
        c2st_acc=metrics_payload.get("norm_C2ST-Acc"),
        adv_to_ben_l2=metrics_payload.get("norm_AdvToBen_L2"),
        adv_to_mal_l2=metrics_payload.get("norm_AdvToMal_L2"),
        runtime_sec=metrics_payload.get("sample_generation_time_sec"),
    )

    if x_ben_denorm is None or x_adv_denorm is None or x_mal_denorm is None:
        return result

    denorm_nan_rate = float(np.mean(np.isnan(x_adv_denorm)))
    denorm_inf_rate = float(np.mean(np.isinf(x_adv_denorm)))
    metrics_payload["denorm_nan_rate"] = denorm_nan_rate
    metrics_payload["denorm_inf_rate"] = denorm_inf_rate
    if denorm_nan_rate > 0.0 or denorm_inf_rate > 0.0:
        _safe_print(f"[Stage2][Warn] denorm invalid: nan_rate={denorm_nan_rate:.6f} inf_rate={denorm_inf_rate:.6f}")

    ben_denorm_mean = np.nanmean(x_ben_denorm, axis=0)
    x_ben_denorm_clean = np.where(np.isfinite(x_ben_denorm), x_ben_denorm, ben_denorm_mean)
    x_adv_denorm_clean = np.where(np.isfinite(x_adv_denorm), x_adv_denorm, ben_denorm_mean)
    x_mal_denorm_clean = np.where(np.isfinite(x_mal_denorm), x_mal_denorm, ben_denorm_mean)
    denorm_bounds_min = np.min(x_ben_denorm_clean, axis=0)
    denorm_bounds_max = np.max(x_ben_denorm_clean, axis=0)
    metrics_denorm = compute_stage2_metrics(
        x_ben_denorm_clean,
        x_adv_denorm_clean,
        feature_names=feature_names,
        max_real=cfg["stage2"].get("metrics_max_real", 2000),
        max_gen=cfg["stage2"].get("metrics_max_gen", 2000),
        seed=seed,
        bounds_min=denorm_bounds_min,
        bounds_max=denorm_bounds_max,
    )
    adv_ben_l2_denorm = nearest_reference_distance(x_adv_denorm_clean, x_ben_denorm_clean)
    adv_mal_l2_denorm = paired_sample_l2(x_adv_denorm_clean, x_mal_denorm_clean)
    metrics_payload.update({f"denorm_{k}": v for k, v in metrics_denorm.as_dict().items()})
    metrics_payload["denorm_AdvToBen_L2"] = adv_ben_l2_denorm
    metrics_payload["denorm_AdvToMal_L2"] = adv_mal_l2_denorm
    return Stage2DistributionMetricsResult(
        metrics_norm=result.metrics_norm,
        adv_ben_l2=result.adv_ben_l2,
        adv_mal_l2=result.adv_mal_l2,
        metrics_denorm=metrics_denorm,
        adv_ben_l2_denorm=adv_ben_l2_denorm,
        adv_mal_l2_denorm=adv_mal_l2_denorm,
    )


def run_stage2_pareto_eval(
    *,
    cfg: dict[str, Any],
    seed: int,
    out_dir: Path,
    x_mal_eval: np.ndarray,
    x_ben_eval: np.ndarray,
    denorm_mean: np.ndarray,
    denorm_std: np.ndarray,
    eval_helper: Any,
    sample_with_alpha: Callable[[np.ndarray, float], np.ndarray],
    metrics_payload: dict[str, Any],
    save_pareto_front: Callable[[Path, list[dict[str, float]]], Path],
) -> None:
    def _safe_float(value: Any) -> float:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return float("nan")
        return out if np.isfinite(out) else float("nan")

    def _auc(points: list[tuple[float, float]]) -> float:
        clean = [(x, y) for x, y in points if np.isfinite(x) and np.isfinite(y)]
        if len(clean) < 2:
            return float("nan")
        clean.sort(key=lambda item: item[0])
        xs = np.asarray([item[0] for item in clean], dtype=np.float64)
        ys = np.asarray([item[1] for item in clean], dtype=np.float64)
        span = float(xs[-1] - xs[0])
        if span <= 1.0e-12:
            return float("nan")
        return float(np.trapezoid(ys, xs) / span)

    pareto_cfg = cfg["stage2"].get("pareto_eval", {}) or {}
    if not pareto_cfg.get("enable", False):
        return
    anchor_grid = pareto_cfg.get("anchor_grid", [0.0, 0.05, 0.1, 0.2, 0.3, 0.5])
    max_samples = int(pareto_cfg.get("max_samples", 1000))
    pareto_rows: list[dict[str, float]] = []
    if max_samples > 0 and x_mal_eval.shape[0] > max_samples:
        rng = np.random.default_rng(seed)
        sel = rng.choice(x_mal_eval.shape[0], max_samples, replace=False)
        pareto_mal = x_mal_eval[sel]
    else:
        pareto_mal = x_mal_eval
    if max_samples > 0 and x_ben_eval.shape[0] > pareto_mal.shape[0]:
        rng = np.random.default_rng(seed + 1)
        sel_b = rng.choice(x_ben_eval.shape[0], pareto_mal.shape[0], replace=False)
        pareto_ben = x_ben_eval[sel_b]
    else:
        pareto_ben = x_ben_eval[: pareto_mal.shape[0]]
    pareto_ben_norm = (pareto_ben - denorm_mean) / (denorm_std + 1.0e-8)
    for alpha in anchor_grid:
        eval_start = time.perf_counter()
        _, _, row = eval_helper.evaluate_candidate(
            sample_with_alpha(pareto_mal, float(alpha)),
            pareto_mal,
            x_ben_norm_ref=pareto_ben_norm,
        )
        row["candidate_eval_time_sec"] = float(time.perf_counter() - eval_start)
        row["mal_anchor_alpha"] = float(alpha)
        pareto_rows.append(row)

    pareto_path = out_dir / "pareto.csv"
    save_pareto_front(pareto_path, pareto_rows)
    metrics_payload["pareto_path"] = str(pareto_path)
    curve_rows: list[dict[str, float | str]] = []
    for row in pareto_rows:
        asr = _safe_float(row.get("asr_oracle", row.get("asr_surrogate")))
        curve_rows.append(
            {
                "axis": "distortion",
                "x": _safe_float(row.get("adv_to_mal_l2")),
                "asr": asr,
                "mal_anchor_alpha": _safe_float(row.get("mal_anchor_alpha")),
            }
        )
        curve_rows.append(
            {
                "axis": "queries",
                "x": _safe_float(row.get("query_count")),
                "asr": asr,
                "mal_anchor_alpha": _safe_float(row.get("mal_anchor_alpha")),
            }
        )
        curve_rows.append(
            {
                "axis": "latency",
                "x": _safe_float(row.get("candidate_eval_time_sec")),
                "asr": asr,
                "mal_anchor_alpha": _safe_float(row.get("mal_anchor_alpha")),
            }
        )
    curve_path = out_dir / "pareto_tradeoff_curves.csv"
    save_records_csv(curve_path, curve_rows, fieldnames=["axis", "x", "asr", "mal_anchor_alpha"])
    metrics_payload["pareto_tradeoff_curve_path"] = str(curve_path)
    distortion_pts = [(float(r["x"]), float(r["asr"])) for r in curve_rows if r["axis"] == "distortion"]
    query_pts = [(float(r["x"]), float(r["asr"])) for r in curve_rows if r["axis"] == "queries"]
    latency_pts = [(float(r["x"]), float(r["asr"])) for r in curve_rows if r["axis"] == "latency"]
    metrics_payload["pareto_auc_asr_vs_distortion"] = _auc(distortion_pts)
    metrics_payload["pareto_auc_asr_vs_queries"] = _auc(query_pts)
    metrics_payload["pareto_auc_asr_vs_latency"] = _auc(latency_pts)
    _safe_print(f"[Stage2] pareto saved to {pareto_path}")


def persist_stage2_metrics(metrics_payload: dict[str, Any], out_dir: Path) -> None:
    save_metrics(metrics_payload, out_dir)
    save_metrics_csv(metrics_payload, out_dir)
