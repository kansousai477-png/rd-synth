from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class Stage2SelectionResult:
    selected_alpha: float
    selected_adv_pre: np.ndarray | None
    selected_adv_norm: np.ndarray | None
    selected_row: dict[str, float] | None


def _finite_value(row: dict[str, float], key: str) -> float | None:
    value = row.get(key)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _passes_hard_constraints(row: dict[str, float], selection_cfg: dict[str, Any]) -> bool:
    checks = (
        ("min_asr_oracle", "asr_oracle", "min"),
        ("min_asr_surrogate", "asr_surrogate", "min"),
        ("max_adv_pmal_oracle", "adv_pmal_oracle", "max"),
        ("max_adv_pmal_surrogate", "adv_pmal_surrogate", "max"),
        ("max_ffd", "ffd", "max"),
        ("max_swd", "swd", "max"),
        ("max_adv_to_ben_l2", "adv_to_ben_l2", "max"),
        ("max_adv_to_mal_l2", "adv_to_mal_l2", "max"),
    )
    for cfg_key, row_key, mode in checks:
        threshold = selection_cfg.get(cfg_key)
        if threshold is None or threshold == "":
            continue
        try:
            threshold_value = float(threshold)
        except (TypeError, ValueError):
            continue
        metric_value = _finite_value(row, row_key)
        if metric_value is None:
            return False
        if mode == "min" and metric_value < threshold_value:
            return False
        if mode == "max" and metric_value > threshold_value:
            return False
    return True


def select_pareto_candidate(
    rows: list[dict[str, float]],
    *,
    prefer_oracle: bool,
    score_weights: dict[str, float],
) -> dict[str, float] | None:
    if not rows:
        return None
    asr_key = "asr_oracle" if prefer_oracle else "asr_surrogate"

    def score(row: dict[str, float]) -> float:
        selection_score = float(row.get("selection_score", float("nan")))
        if np.isfinite(selection_score):
            return selection_score
        asr = float(row.get(asr_key, float("nan")))
        if not np.isfinite(asr):
            asr = float(row.get("asr_surrogate", 0.0))
        ffd = float(row.get("ffd", 0.0))
        return asr - 0.01 * ffd

    return max(rows, key=lambda row: (score(row), float(row.get(asr_key, -1.0))))


def select_per_sample_candidates(
    candidate_cache: dict[float, tuple[np.ndarray, np.ndarray, dict[str, object]]],
    candidate_alphas: list[float],
    *,
    score_fn: Callable[[np.ndarray], np.ndarray],
    distance_weight: float = 0.0,
    benign_reference: np.ndarray | None = None,
    support_weight: float = 0.0,
    remapability_weight: float = 0.0,
    stage3_closed_loop_weight: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    if not candidate_alphas:
        raise ValueError("candidate_alphas must not be empty")
    selected = [candidate_cache[float(alpha)] for alpha in candidate_alphas]
    adv_pre_stack = np.stack([item[0] for item in selected], axis=0)
    adv_norm_stack = np.stack([item[1] for item in selected], axis=0)
    score_stack = np.stack([score_fn(item[0]) for item in selected], axis=0)
    if benign_reference is not None and distance_weight > 0.0:
        distance_stack = _nearest_reference_distance_stack(adv_norm_stack, benign_reference)
        score_stack = score_stack + distance_weight * distance_stack
    if support_weight > 0.0:
        support_penalties = []
        for _, _, row in selected:
            penalty = row.get("_per_sample_support_penalty")
            if isinstance(penalty, np.ndarray) and penalty.shape[0] == score_stack.shape[1]:
                support_penalties.append(np.asarray(penalty, dtype=np.float64))
            else:
                support_penalties.append(np.zeros((score_stack.shape[1],), dtype=np.float64))
        score_stack = score_stack + support_weight * np.stack(support_penalties, axis=0)
    if remapability_weight > 0.0:
        remap_penalties = []
        for _, _, row in selected:
            penalty = row.get("_per_sample_remap_penalty")
            if isinstance(penalty, np.ndarray) and penalty.shape[0] == score_stack.shape[1]:
                remap_penalties.append(np.asarray(penalty, dtype=np.float64))
            else:
                remap_penalties.append(np.zeros((score_stack.shape[1],), dtype=np.float64))
        score_stack = score_stack + remapability_weight * np.stack(remap_penalties, axis=0)
    if stage3_closed_loop_weight > 0.0:
        stage3_penalties = []
        for _, _, row in selected:
            l2_penalty = row.get("_per_sample_stage3_target_l2")
            mae_penalty = row.get("_per_sample_stage3_target_mae")
            if (
                isinstance(l2_penalty, np.ndarray)
                and isinstance(mae_penalty, np.ndarray)
                and l2_penalty.shape[0] == score_stack.shape[1]
                and mae_penalty.shape[0] == score_stack.shape[1]
            ):
                stage3_penalties.append(np.asarray(l2_penalty + 0.5 * mae_penalty, dtype=np.float64))
            else:
                stage3_penalties.append(np.zeros((score_stack.shape[1],), dtype=np.float64))
        score_stack = score_stack + stage3_closed_loop_weight * np.stack(stage3_penalties, axis=0)
    best_idx = np.argmin(score_stack, axis=0)
    sample_idx = np.arange(best_idx.shape[0])
    adv_pre = adv_pre_stack[best_idx, sample_idx]
    adv_norm = adv_norm_stack[best_idx, sample_idx]
    selected_alphas = np.asarray([candidate_alphas[int(i)] for i in best_idx], dtype=np.float64)
    summary = {
        "selected_mal_anchor_alpha_mean": float(np.mean(selected_alphas)),
        "selected_mal_anchor_alpha_min": float(np.min(selected_alphas)),
        "selected_mal_anchor_alpha_max": float(np.max(selected_alphas)),
    }
    for alpha in candidate_alphas:
        summary[f"selected_alpha_frac_{str(alpha).replace('.', 'p')}"] = float(
            np.mean(np.isclose(selected_alphas, alpha))
        )
    return adv_pre, adv_norm, summary


def _nearest_reference_distance_stack(adv_norm_stack: np.ndarray, benign_reference: np.ndarray) -> np.ndarray:
    """Return per-candidate/sample distance to the nearest benign reference row."""

    adv = np.asarray(adv_norm_stack, dtype=np.float64)
    ref = np.asarray(benign_reference, dtype=np.float64)
    if adv.ndim != 3:
        raise ValueError("adv_norm_stack must have shape (candidate, sample, feature)")
    if ref.ndim != 2:
        raise ValueError("benign_reference must have shape (reference, feature)")
    if adv.shape[2] != ref.shape[1]:
        raise ValueError(
            f"feature dimension mismatch: adv_norm_stack has {adv.shape[2]} features, "
            f"benign_reference has {ref.shape[1]} features"
        )
    if ref.shape[0] == 0:
        return np.zeros(adv.shape[:2], dtype=np.float64)

    ref_sq = np.sum(ref * ref, axis=1)[None, :]
    out = np.empty(adv.shape[:2], dtype=np.float64)
    for idx in range(adv.shape[0]):
        rows = adv[idx]
        dist_sq = np.sum(rows * rows, axis=1)[:, None] + ref_sq - 2.0 * rows @ ref.T
        out[idx] = np.sqrt(np.maximum(np.min(dist_sq, axis=1), 0.0))
    return out


def _unique_sorted(values: list[float]) -> list[float]:
    seen: set[float] = set()
    out: list[float] = []
    for value in values:
        key = round(float(value), 8)
        if key in seen:
            continue
        seen.add(key)
        out.append(float(value))
    return sorted(out)


def _local_refinement_grid(
    *,
    center: float,
    radius: float,
    points: int,
    alpha_min: float,
    alpha_max: float,
    evaluated: set[float],
) -> list[float]:
    if points <= 1 or radius <= 0.0:
        return []
    values = np.linspace(float(center) - float(radius), float(center) + float(radius), int(points))
    candidates = [
        float(np.clip(value, float(alpha_min), float(alpha_max)))
        for value in values
        if round(float(np.clip(value, float(alpha_min), float(alpha_max))), 8) not in evaluated
    ]
    return _unique_sorted(candidates)


def auto_select_stage2_candidates(
    *,
    pareto_cfg: dict[str, Any],
    eval_helper: Any,
    sample_with_alpha: Callable[[np.ndarray, float], np.ndarray],
    x_mal_eval: np.ndarray,
    x_ben_norm: np.ndarray,
    attack_score_fn: Callable[[np.ndarray], np.ndarray],
    metrics_payload: dict[str, Any],
) -> Stage2SelectionResult:
    selected_alpha = 0.0
    selected_adv_pre: np.ndarray | None = None
    selected_adv_norm: np.ndarray | None = None
    selected_row: dict[str, float] | None = None
    if not (pareto_cfg.get("enable", False) and bool(pareto_cfg.get("auto_select", False))):
        return Stage2SelectionResult(selected_alpha, selected_adv_pre, selected_adv_norm, selected_row)

    selection_cfg = pareto_cfg.get("selection", {}) or {}
    selection_mode = str(selection_cfg.get("candidate_selection", "global_pareto")).lower()
    score_weights = {
        "asr": float(selection_cfg.get("asr_weight", 1.0)),
        "c2st_auc": float(selection_cfg.get("c2st_auc_weight", 0.35)),
        "adv_to_ben_l2": float(selection_cfg.get("adv_to_ben_l2_weight", 0.10)),
        "ffd": float(selection_cfg.get("ffd_weight", 0.02)),
        "adv_pmal": float(selection_cfg.get("adv_pmal_weight", 0.25)),
    }
    anchor_grid = [float(alpha) for alpha in pareto_cfg.get("anchor_grid", [0.0, 0.05, 0.1, 0.2, 0.3, 0.5])]
    candidate_rows: list[dict[str, float]] = []
    candidate_cache: dict[float, tuple[np.ndarray, np.ndarray, dict[str, float]]] = {}

    def evaluate_alpha(alpha: float) -> None:
        alpha = float(alpha)
        if round(alpha, 8) in {round(float(existing), 8) for existing in candidate_cache}:
            return
        evaluated = eval_helper.evaluate_candidate(sample_with_alpha(x_mal_eval, float(alpha)), x_mal_eval)
        _, _, row = evaluated
        row["mal_anchor_alpha"] = float(alpha)
        candidate_rows.append(row)
        candidate_cache[float(alpha)] = evaluated

    def feasible_or_all_rows() -> tuple[list[dict[str, float]], list[dict[str, float]], bool]:
        feasible = [row for row in candidate_rows if _passes_hard_constraints(row, selection_cfg)]
        if feasible:
            return feasible, feasible, False
        return candidate_rows, feasible, bool(candidate_rows)

    def best_global_row(rows: list[dict[str, float]]) -> dict[str, float] | None:
        if selection_mode == "stage3_closed_loop":
            return max(
                rows,
                key=lambda row: (
                    float(row.get("stage3_closed_loop_score", float("-inf"))),
                    float(row.get("selection_score", float("-inf"))),
                    float(row.get("asr_oracle", float("-inf"))),
                ),
            )
        return select_pareto_candidate(
            rows,
            prefer_oracle=bool(selection_cfg.get("prefer_oracle", True)),
            score_weights=score_weights,
        )

    for alpha in anchor_grid:
        evaluate_alpha(alpha)

    iterative_rounds = max(1, int(selection_cfg.get("iterative_rounds", pareto_cfg.get("iterative_rounds", 1))))
    if selection_mode in {"global_pareto", "stage3_closed_loop"} and iterative_rounds > 1:
        alpha_min = float(selection_cfg.get("alpha_min", pareto_cfg.get("alpha_min", min(anchor_grid))))
        alpha_max = float(selection_cfg.get("alpha_max", pareto_cfg.get("alpha_max", max(anchor_grid))))
        refinement_points = max(3, int(selection_cfg.get("iterative_points", 3)))
        radius_decay = float(selection_cfg.get("iterative_radius_decay", 0.5))
        sorted_grid = _unique_sorted(anchor_grid)
        radius = float(selection_cfg.get("iterative_radius", 0.0))
        if radius <= 0.0 and len(sorted_grid) > 1:
            radius = 0.5 * max(b - a for a, b in zip(sorted_grid[:-1], sorted_grid[1:]))
        elif radius <= 0.0:
            radius = 0.1
        rounds_used = 1
        for round_idx in range(1, iterative_rounds):
            selection_rows, _, _ = feasible_or_all_rows()
            best_row = best_global_row(selection_rows)
            if best_row is None:
                break
            evaluated_keys = {round(float(alpha), 8) for alpha in candidate_cache}
            local_grid = _local_refinement_grid(
                center=float(best_row["mal_anchor_alpha"]),
                radius=radius,
                points=refinement_points,
                alpha_min=alpha_min,
                alpha_max=alpha_max,
                evaluated=evaluated_keys,
            )
            if not local_grid:
                break
            for alpha in local_grid:
                evaluate_alpha(alpha)
            rounds_used = round_idx + 1
            radius *= radius_decay
        metrics_payload["candidate_selection_iterative_rounds"] = int(iterative_rounds)
        metrics_payload["candidate_selection_iterative_rounds_used"] = int(rounds_used)

    feasible_rows = [row for row in candidate_rows if _passes_hard_constraints(row, selection_cfg)]
    metrics_payload["candidate_count_total"] = int(len(candidate_rows))
    metrics_payload["candidate_count_feasible"] = int(len(feasible_rows))
    if feasible_rows:
        selection_rows = feasible_rows
        metrics_payload["selected_candidate_constraints_fallback"] = False
    else:
        selection_rows = candidate_rows
        metrics_payload["selected_candidate_constraints_fallback"] = bool(candidate_rows)

    if selection_mode == "per_sample_attack_score":
        effective_alphas = [float(row["mal_anchor_alpha"]) for row in selection_rows]
        selected_adv_pre, selected_adv_norm, selection_summary = select_per_sample_candidates(
            candidate_cache,
            effective_alphas,
            score_fn=attack_score_fn,
            distance_weight=float(selection_cfg.get("per_sample_distance_weight", 0.0)),
            benign_reference=x_ben_norm[: x_mal_eval.shape[0]],
            support_weight=float(selection_cfg.get("per_sample_support_weight", 0.0)),
            remapability_weight=float(selection_cfg.get("per_sample_remapability_weight", 0.0)),
            stage3_closed_loop_weight=float(selection_cfg.get("per_sample_stage3_closed_loop_weight", 0.0)),
        )
        selected_alpha = float(selection_summary["selected_mal_anchor_alpha_mean"])
        metrics_payload["selected_candidate_mode"] = selection_mode
        metrics_payload.update(selection_summary)
        return Stage2SelectionResult(selected_alpha, selected_adv_pre, selected_adv_norm, None)

    if selection_mode == "stage3_closed_loop":
        best_row = max(
            selection_rows,
            key=lambda row: (
                float(row.get("stage3_closed_loop_score", float("-inf"))),
                float(row.get("selection_score", float("-inf"))),
                float(row.get("asr_oracle", float("-inf"))),
            ),
        )
        selected_alpha = float(best_row["mal_anchor_alpha"])
        selected_adv_pre, selected_adv_norm, _ = candidate_cache[selected_alpha]
        metrics_payload["selected_candidate_mode"] = selection_mode
        metrics_payload["selected_mal_anchor_alpha"] = selected_alpha
        metrics_payload["selected_candidate_stage3_closed_loop_score"] = float(
            best_row.get("stage3_closed_loop_score", float("nan"))
        )
        return Stage2SelectionResult(selected_alpha, selected_adv_pre, selected_adv_norm, best_row)

    best_row = select_pareto_candidate(
        selection_rows,
        prefer_oracle=bool(selection_cfg.get("prefer_oracle", True)),
        score_weights=score_weights,
    )
    if best_row is not None:
        selected_alpha = float(best_row["mal_anchor_alpha"])
        selected_adv_pre, selected_adv_norm, _ = candidate_cache[selected_alpha]
        metrics_payload["selected_candidate_mode"] = selection_mode
        metrics_payload["selected_mal_anchor_alpha"] = selected_alpha
        metrics_payload["selected_candidate_asr_oracle"] = float(best_row.get("asr_oracle", float("nan")))
        selected_row = best_row
    return Stage2SelectionResult(selected_alpha, selected_adv_pre, selected_adv_norm, selected_row)
