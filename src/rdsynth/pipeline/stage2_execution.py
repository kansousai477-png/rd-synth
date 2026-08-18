from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from rdsynth.pipeline.stage2_runtime import (
    record_sample_statistics,
    update_selected_candidate_metrics,
)
from rdsynth.pipeline.stage2_selection import auto_select_stage2_candidates
from rdsynth.utils.query_oracle import QueryOracle, QueryOracleStats


@dataclass(frozen=True)
class Stage2ExecutionSelection:
    selected_alpha: float
    selected_adv_pre: np.ndarray | None
    selected_adv_norm: np.ndarray | None
    selected_row: dict[str, float] | None
    selection_runtime_sec: float


@dataclass(frozen=True)
class Stage2SampleExecution:
    x_adv_pre: np.ndarray
    x_adv_norm: np.ndarray
    sample_runtime_sec: float
    nan_rate: float
    inf_rate: float
    query_stats: QueryOracleStats


def run_stage2_candidate_selection(
    *,
    pareto_cfg: dict[str, Any],
    eval_helper: Any,
    sample_with_alpha: Callable[[np.ndarray, float], np.ndarray],
    x_mal_eval: np.ndarray,
    x_ben_norm: np.ndarray,
    attack_query_oracle: QueryOracle,
    metrics_payload: dict[str, Any],
    default_alpha: float,
) -> Stage2ExecutionSelection:
    selection_start = time.perf_counter()
    selection_result = auto_select_stage2_candidates(
        pareto_cfg=pareto_cfg,
        eval_helper=eval_helper,
        sample_with_alpha=sample_with_alpha,
        x_mal_eval=x_mal_eval,
        x_ben_norm=x_ben_norm,
        attack_score_fn=attack_query_oracle,
        metrics_payload=metrics_payload,
    )
    selection_runtime_sec = time.perf_counter() - selection_start
    metrics_payload["candidate_selection_time_sec"] = float(selection_runtime_sec)

    selected_alpha = float(default_alpha)
    selected_adv_pre = None
    selected_adv_norm = None
    selected_row = None
    if selection_result.selected_alpha != 0.0 or selection_result.selected_adv_pre is not None:
        selected_alpha = float(selection_result.selected_alpha)
        selected_adv_pre = selection_result.selected_adv_pre
        selected_adv_norm = selection_result.selected_adv_norm
        selected_row = selection_result.selected_row

    return Stage2ExecutionSelection(
        selected_alpha=selected_alpha,
        selected_adv_pre=selected_adv_pre,
        selected_adv_norm=selected_adv_norm,
        selected_row=selected_row,
        selection_runtime_sec=float(selection_runtime_sec),
    )


def execute_stage2_sample(
    *,
    sample_with_alpha: Callable[[np.ndarray, float], np.ndarray],
    eval_helper: Any,
    attack_query_oracle: QueryOracle,
    metrics_payload: dict[str, Any],
    x_mal_eval: np.ndarray,
    selection: Stage2ExecutionSelection,
    warn_fn: Callable[[str], None] = print,
) -> Stage2SampleExecution:
    sample_start = time.perf_counter()
    x_adv = sample_with_alpha(x_mal_eval, selection.selected_alpha) if selection.selected_adv_pre is None else None
    sample_for_stats = x_adv if x_adv is not None else selection.selected_adv_pre
    nan_rate, inf_rate = record_sample_statistics(
        metrics_payload=metrics_payload,
        sample_for_stats=sample_for_stats,
        selected_alpha=selection.selected_alpha,
    )
    update_selected_candidate_metrics(metrics_payload, selection.selected_row)
    if nan_rate > 0.0 or inf_rate > 0.0:
        warn_fn(f"[Stage2][Warn] invalid samples: nan_rate={nan_rate:.6f} inf_rate={inf_rate:.6f}")

    if selection.selected_adv_pre is not None and selection.selected_adv_norm is not None:
        x_adv_pre, x_adv_norm = selection.selected_adv_pre, selection.selected_adv_norm
    else:
        x_adv_pre, x_adv_norm = eval_helper.postprocess_adv(x_adv, x_mal_eval)
    sample_runtime_sec = time.perf_counter() - sample_start
    query_stats = attack_query_oracle.stats()
    _record_query_stats(metrics_payload, query_stats)
    return Stage2SampleExecution(
        x_adv_pre=x_adv_pre,
        x_adv_norm=x_adv_norm,
        sample_runtime_sec=float(sample_runtime_sec),
        nan_rate=float(nan_rate),
        inf_rate=float(inf_rate),
        query_stats=query_stats,
    )


def _record_query_stats(metrics_payload: dict[str, Any], query_stats: QueryOracleStats) -> None:
    metrics_payload["attack_score_query_count"] = int(query_stats.query_count)
    metrics_payload["attack_score_query_calls"] = int(query_stats.query_calls)
    metrics_payload["attack_score_query_time_sec"] = float(query_stats.query_time_sec)
    metrics_payload["attack_score_query_over_budget_count"] = int(query_stats.query_over_budget_count)
    metrics_payload["attack_score_query_budget_exhausted"] = bool(query_stats.budget_exhausted)
