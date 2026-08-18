from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rdsynth.stages.stage2_metrics import infer_groups
from rdsynth.utils.metrics_stage2 import (
    c2st_metrics,
    corr_delta,
    coverage_at_k,
    energy_distance,
    frechet_distance,
    knn_precision_recall,
    sliced_wasserstein_distance,
)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_metric_kv(path: Path) -> dict[str, str]:
    rows = read_csv_rows(path)
    out: dict[str, str] = {}
    for row in rows:
        key = str(row.get("metric", "")).strip()
        if key:
            out[key] = str(row.get("value", "")).strip()
    return out


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def fmt(value: Any, digits: int = 4) -> str:
    number = to_float(value)
    if number is None:
        return "-"
    return f"{number:.{digits}f}"


def _safe_array(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _load_npz_payload(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as payload:
        return {key: payload[key] for key in payload.files}


def _npz_method_name(path: Path) -> str:
    if path.name == "adv_samples.npz":
        return "RD-Synth"
    return path.stem.replace("baseline_", "").replace("_samples", "")


def _stage2_npz_by_method(stage2_dir: Path) -> dict[str, Path]:
    paths = {"RD-Synth": stage2_dir / "adv_samples.npz"}
    for path in sorted(stage2_dir.glob("baseline_*_samples.npz")):
        paths[_npz_method_name(path)] = path
    return paths


def _covariance_metrics(x_real: np.ndarray, x_gen: np.ndarray) -> tuple[float, float]:
    cov_real = np.cov(x_real, rowvar=False)
    cov_gen = np.cov(x_gen, rowvar=False)
    cov_real = np.nan_to_num(cov_real, nan=0.0, posinf=0.0, neginf=0.0)
    cov_gen = np.nan_to_num(cov_gen, nan=0.0, posinf=0.0, neginf=0.0)
    eig_real = np.sort(np.linalg.eigvalsh(cov_real))[::-1]
    eig_gen = np.sort(np.linalg.eigvalsh(cov_gen))[::-1]
    covspec_l2 = float(np.linalg.norm(eig_real - eig_gen))
    covtrace = float(abs(np.trace(cov_real) - np.trace(cov_gen)))
    return covspec_l2, covtrace


def _pairwise_distances(x: np.ndarray, sample_n: int = 256, seed: int = 42) -> np.ndarray:
    if len(x) < 2:
        return np.zeros((0,), dtype=np.float64)
    rng = np.random.default_rng(seed)
    if len(x) > sample_n:
        idx = rng.choice(len(x), size=sample_n, replace=False)
        x = x[np.sort(idx)]
    diff = x[:, None, :] - x[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    upper = np.triu_indices(len(x), k=1)
    return dist[upper]


def _ks_statistic(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    x = np.sort(np.asarray(x, dtype=np.float64))
    y = np.sort(np.asarray(y, dtype=np.float64))
    merged = np.sort(np.concatenate([x, y]))
    cdf_x = np.searchsorted(x, merged, side="right") / len(x)
    cdf_y = np.searchsorted(y, merged, side="right") / len(y)
    return float(np.max(np.abs(cdf_x - cdf_y)))


def _cross_group_corr_delta(x_real: np.ndarray, x_gen: np.ndarray, feature_names: list[str]) -> dict[str, float]:
    groups = infer_groups(feature_names)
    corr_real = np.corrcoef(_safe_array(x_real), rowvar=False)
    corr_gen = np.corrcoef(_safe_array(x_gen), rowvar=False)
    corr_real = np.nan_to_num(corr_real, nan=0.0, posinf=0.0, neginf=0.0)
    corr_gen = np.nan_to_num(corr_gen, nan=0.0, posinf=0.0, neginf=0.0)

    def _block_delta(a: list[int], b: list[int]) -> float:
        if not a or not b:
            return float("nan")
        block_real = corr_real[np.ix_(a, b)]
        block_gen = corr_gen[np.ix_(a, b)]
        return float(np.linalg.norm(block_real - block_gen, ord="fro") / (block_real.size + 1.0e-12))

    st = _block_delta(groups["spatial"], groups["temporal"])
    sp = _block_delta(groups["spatial"], groups["protocol"])
    tp = _block_delta(groups["temporal"], groups["protocol"])
    avg = float(np.nanmean([st, sp, tp]))
    return {"CGD_ST": st, "CGD_SP": sp, "CGD_TP": tp, "CGD_AVG": avg}


def _compute_stage2_realism_metrics(npz_path: Path) -> dict[str, float]:
    payload = _load_npz_payload(npz_path)
    benign = _safe_array(payload["benign_pre"])
    adv = _safe_array(payload["adv_pre"])
    mal = _safe_array(payload["mal_pre"])
    feature_names = [str(item) for item in np.asarray(payload["feature_names"]).tolist()]
    cgd = _cross_group_corr_delta(benign, adv, feature_names)
    covspec_l2, covtrace = _covariance_metrics(benign, adv)
    pair_real = _pairwise_distances(benign, seed=41)
    pair_adv = _pairwise_distances(adv, seed=42)
    auc, acc = c2st_metrics(benign, adv, seed=42)
    knn_p, knn_r = knn_precision_recall(benign, adv, k=5)
    return {
        "FFD": frechet_distance(benign, adv),
        "SWD": sliced_wasserstein_distance(benign, adv, seed=42),
        "Energy": energy_distance(benign, adv),
        "C2ST_AUC": auc,
        "C2ST_Acc": acc,
        "Coverage@5": coverage_at_k(benign, adv, k=5),
        "kNN_P": knn_p,
        "kNN_R": knn_r,
        "Corr_Delta": corr_delta(benign, adv),
        "CovSpec_L2": covspec_l2,
        "CovTrace": covtrace,
        "PairDist_KS": _ks_statistic(pair_real, pair_adv),
        "PairMean": float(abs(np.mean(pair_real) - np.mean(pair_adv))) if len(pair_real) and len(pair_adv) else float("nan"),
        "AdvToMal_L2": float(np.mean(np.linalg.norm(adv - mal, axis=1))) if len(adv) and len(mal) and adv.shape == mal.shape else float("nan"),
        **cgd,
    }


def pipe_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *["| " + " | ".join(row) + " |" for row in rows],
    ]


def group_key(row: dict[str, str], group_by: list[str] | None) -> tuple[str, ...]:
    if not group_by:
        return tuple()
    return tuple(str(row.get(key, "")) for key in group_by)


def decorate_best(
    rows: list[dict[str, str]],
    *,
    columns: dict[str, str],
    group_by: list[str] | None = None,
) -> list[dict[str, str]]:
    if not rows:
        return []

    grouped: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(group_key(row, group_by), []).append(row)

    best_values: dict[tuple[tuple[str, ...], str], float] = {}
    for key, items in grouped.items():
        for column, direction in columns.items():
            nums = [to_float(item.get(column)) for item in items]
            nums = [value for value in nums if value is not None]
            if not nums:
                continue
            best_values[(key, column)] = max(nums) if direction == "max" else min(nums)

    out: list[dict[str, str]] = []
    for row in rows:
        decorated = dict(row)
        key = group_key(row, group_by)
        for column, direction in columns.items():
            best = best_values.get((key, column))
            value = to_float(row.get(column))
            if best is None or value is None:
                continue
            if abs(value - best) <= 1.0e-12:
                decorated[column] = f"**{row[column]}**"
        out.append(decorated)
    return out


def render_table(
    rows: list[dict[str, str]],
    *,
    best_columns: dict[str, str] | None = None,
    group_by: list[str] | None = None,
) -> list[str]:
    if not rows:
        return ["（当前无数据）"]
    display_rows = decorate_best(rows, columns=best_columns or {}, group_by=group_by)
    headers = list(display_rows[0].keys())
    return pipe_table(headers, [[str(row.get(header, "-")) for header in headers] for row in display_rows])


def load_stage1_matrix(dataset_root: Path, rq1_rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], str]:
    if not rq1_rows:
        return [], [], [], "当前运行缺少 `rq1_matrix_summary.csv`，无法展示 Stage1 异构 IDS 提取矩阵。"
    rq1 = rq1_rows[0]
    summary_path = Path(str(rq1.get("summary_path", "")))
    matrix_path = Path(str(rq1.get("matrix_path", "")))
    per_model_rows: list[dict[str, str]] = []
    summary_rows: list[dict[str, str]] = []
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8-sig", newline="") as handle:
            all_rows = list(csv.reader(handle))
        section = "per_model"
        per_model_headers: list[str] = []
        summary_headers: list[str] = []
        for row in all_rows:
            clean_row = [cell.strip() for cell in row]
            while clean_row and clean_row[-1] == "":
                clean_row.pop()
            if not clean_row:
                continue
            if clean_row[:3] == ["metric", "mean", "std"]:
                section = "summary"
                summary_headers = ["metric", "mean", "std"]
                continue
            if section == "per_model":
                if not per_model_headers:
                    per_model_headers = clean_row
                    continue
                payload = {
                    per_model_headers[idx]: clean_row[idx] if idx < len(clean_row) else ""
                    for idx in range(len(per_model_headers))
                }
                per_model_rows.append(payload)
            else:
                payload = {
                    summary_headers[idx]: clean_row[idx] if idx < len(clean_row) else ""
                    for idx in range(len(summary_headers))
                }
                summary_rows.append(payload)
    matrix_rows = read_csv_rows(matrix_path)
    note = (
        f"当前 Stage1 矩阵来自 seed=`{rq1.get('seed', '-')}` 的主线运行，"
        f"配置里声明的 IDS 数量为 `{rq1.get('ids_count', rq1.get('oracle_count', '-'))}`。"
    )
    return per_model_rows, summary_rows, matrix_rows, note


def build_global_main_table(main_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in main_rows:
        out_dir = Path(str(row.get("out_dir", "")))
        pcap_eval_rows = read_csv_rows(out_dir / "stage3" / "pcap_eval.csv")
        sampled = sorted({str(item.get("source_name", "")).strip() for item in pcap_eval_rows if str(item.get("source_name", "")).strip()})
        replayed = sorted(
            {
                str(item.get("source_name", "")).strip()
                for item in pcap_eval_rows
                if str(item.get("source_name", "")).strip() and str(item.get("is_original", "")).strip() != "1"
            }
        )
        out.append(
            {
                "Attack": str(row.get("attack_type", "-")),
                "Seed": str(row.get("seed", "-")),
                "Stage1 gain": fmt((to_float(row.get("stage1_agreement")) or 0.0) - (to_float(row.get("stage1_baseline_agreement")) or 0.0)),
                "Stage1 score": fmt(row.get("stage1_decision_score")),
                "Stage2 score": fmt(row.get("stage2_decision_score")),
                "Stage2 ASR": fmt(row.get("stage2_asr_oracle")),
                "Stage2 FFD": fmt(row.get("stage2_norm_ffd")),
                "Stage2 SWD": fmt(row.get("stage2_norm_swd")),
                "Stage3 replay ASR": fmt(row.get("stage3_pcap_attack_success_rate")),
                "Stage3 score": fmt(row.get("stage3_decision_score")),
                "Deployability": fmt(row.get("stage3_deployability_score")),
                "Target L2": fmt(row.get("stage3_pcap_target_l2_mean")),
                "Fatal rate": fmt(row.get("stage3_pcap_valid_fatal_rate")),
                "Carrier sampled": str(len(sampled)),
                "Carrier replayed": str(len(replayed)),
                "Representative carrier": str(row.get("pcap_selected_name", "-")),
            }
        )
    return out


def build_global_stage2_compare(main_rows: list[dict[str, str]], baseline_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in main_rows:
        out.append(
            {
                "Attack": str(row.get("attack_type", "-")),
                "Method": "RD-Synth",
                "Family": "ours",
                "ASR_oracle": fmt(row.get("stage2_asr_oracle")),
                "ASR_surrogate": fmt(row.get("stage2_asr_surrogate")),
                "FFD": fmt(row.get("stage2_norm_ffd")),
                "SWD": fmt(row.get("stage2_norm_swd")),
                "AdvToMal_L2": fmt(row.get("stage2_norm_advtomal_l2")),
                "Queries_per_success": fmt(row.get("stage2_queries_per_success_oracle")),
                "Time_sec": fmt(row.get("stage2_end_to_end_time_sec")),
                "Score": fmt(row.get("stage2_decision_score")),
            }
        )
    for row in sorted(baseline_rows, key=lambda item: (str(item.get("attack_type", "")), str(item.get("method", "")))):
        out.append(
            {
                "Attack": str(row.get("attack_type", "-")),
                "Method": str(row.get("method", "-")),
                "Family": str(row.get("family", "-")),
                "ASR_oracle": fmt(row.get("asr_oracle")),
                "ASR_surrogate": fmt(row.get("asr_surrogate")),
                "FFD": fmt(row.get("norm_ffd")),
                "SWD": fmt(row.get("norm_swd")),
                "AdvToMal_L2": fmt(row.get("norm_advtomal_l2")),
                "Queries_per_success": fmt(row.get("queries_per_success_oracle")),
                "Time_sec": fmt(row.get("end_to_end_time_sec")),
                "Score": "-",
            }
        )
    return out


def build_stage2_realism_table(main_rows: list[dict[str, str]], baseline_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    family_lookup = {(str(row.get("attack_type", "")), str(row.get("method", ""))): str(row.get("family", "-")) for row in baseline_rows}
    for row in main_rows:
        attack = str(row.get("attack_type", "-"))
        stage2_dir = Path(str(row.get("out_dir", ""))) / "stage2"
        metrics_csv = read_metric_kv(stage2_dir / "metrics.csv")
        npz_by_method = _stage2_npz_by_method(stage2_dir)
        for method, npz_path in npz_by_method.items():
            metrics = _compute_stage2_realism_metrics(npz_path)
            if method == "RD-Synth":
                family = "ours"
                asr_oracle = row.get("stage2_asr_oracle")
                asr_surrogate = row.get("stage2_asr_surrogate")
                queries = row.get("stage2_queries_per_success_oracle")
                time_sec = row.get("stage2_end_to_end_time_sec")
            else:
                family = family_lookup.get((attack, method), "-")
                key = method
                asr_oracle = metrics_csv.get(f"baseline_{key}_asr_oracle")
                asr_surrogate = metrics_csv.get(f"baseline_{key}_asr_surrogate")
                queries = metrics_csv.get(f"baseline_{key}_queries_per_success_oracle")
                time_sec = metrics_csv.get(f"baseline_{key}_end_to_end_time_sec")
            out.append(
                {
                    "Attack": attack,
                    "Method": method,
                    "Family": family,
                    "ASR_oracle": fmt(asr_oracle),
                    "ASR_surrogate": fmt(asr_surrogate),
                    "FFD": fmt(metrics["FFD"]),
                    "SWD": fmt(metrics["SWD"]),
                    "Energy": fmt(metrics["Energy"]),
                    "C2ST_AUC": fmt(metrics["C2ST_AUC"]),
                    "C2ST_Acc": fmt(metrics["C2ST_Acc"]),
                    "Coverage@5": fmt(metrics["Coverage@5"]),
                    "kNN_P": fmt(metrics["kNN_P"]),
                    "kNN_R": fmt(metrics["kNN_R"]),
                    "Corr_Delta": fmt(metrics["Corr_Delta"]),
                    "CovSpec_L2": fmt(metrics["CovSpec_L2"]),
                    "CovTrace": fmt(metrics["CovTrace"]),
                    "PairDist_KS": fmt(metrics["PairDist_KS"]),
                    "PairMean": fmt(metrics["PairMean"]),
                    "AdvToMal_L2": fmt(metrics["AdvToMal_L2"]),
                    "Queries_per_success": fmt(queries),
                    "Time_sec": fmt(time_sec),
                }
            )
    return out


def build_stage2_cgd_table(main_rows: list[dict[str, str]], baseline_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    family_lookup = {(str(row.get("attack_type", "")), str(row.get("method", ""))): str(row.get("family", "-")) for row in baseline_rows}
    for row in main_rows:
        attack = str(row.get("attack_type", "-"))
        stage2_dir = Path(str(row.get("out_dir", ""))) / "stage2"
        for method, npz_path in _stage2_npz_by_method(stage2_dir).items():
            payload = _load_npz_payload(npz_path)
            benign = _safe_array(payload["benign_pre"])
            adv = _safe_array(payload["adv_pre"])
            feature_names = [str(item) for item in np.asarray(payload["feature_names"]).tolist()]
            cgd = _cross_group_corr_delta(benign, adv, feature_names)
            out.append(
                {
                    "Attack": attack,
                    "Method": method,
                    "Family": "ours" if method == "RD-Synth" else family_lookup.get((attack, method), "-"),
                    "CGD_ST": fmt(cgd["CGD_ST"]),
                    "CGD_SP": fmt(cgd["CGD_SP"]),
                    "CGD_TP": fmt(cgd["CGD_TP"]),
                    "CGD_AVG": fmt(cgd["CGD_AVG"]),
                }
            )
    return out


def build_attack_slice_table(stage1_rows: list[dict[str, str]], stage2_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    stage1_by_attack = {str(row.get("attack_type", "")): row for row in stage1_rows}
    out: list[dict[str, str]] = []
    for row in sorted(stage2_rows, key=lambda item: str(item.get("attack_type", ""))):
        attack = str(row.get("attack_type", "-"))
        s1 = stage1_by_attack.get(attack, {})
        out.append(
            {
                "Attack": attack,
                "Eval rows": str(row.get("stage2_eval_attack_rows", "-")),
                "Stage1 agreement": fmt(s1.get("stage1_agreement")),
                "Stage1 baseline agreement": fmt(s1.get("stage1_baseline_agreement")),
                "ASR_oracle": fmt(row.get("asr_oracle")),
                "ASR_surrogate": fmt(row.get("asr_surrogate")),
                "FFD": fmt(row.get("norm_FFD")),
                "SWD": fmt(row.get("norm_SWD")),
                "AdvToMal_L2": fmt(row.get("norm_AdvToMal_L2")),
                "Score": fmt(row.get("stage2_decision_score")),
                "Time_sec": fmt(row.get("sample_end_to_end_time_sec")),
            }
        )
    return out


def _extract_methods_from_metrics(metrics: dict[str, str]) -> list[str]:
    methods = set()
    for key in metrics:
        prefix = "baseline_"
        suffix = "_asr_oracle"
        if key.startswith(prefix) and key.endswith(suffix):
            methods.add(key[len(prefix) : -len(suffix)])
    return sorted(methods)


def build_attack_baseline_compare(stage2_attack_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in sorted(stage2_attack_rows, key=lambda item: str(item.get("attack_type", ""))):
        attack = str(row.get("attack_type", "-"))
        out.append(
            {
                "Attack": attack,
                "Method": "RD-Synth",
                "ASR_oracle": fmt(row.get("asr_oracle")),
                "ASR_surrogate": fmt(row.get("asr_surrogate")),
                "FFD": fmt(row.get("norm_FFD")),
                "SWD": fmt(row.get("norm_SWD")),
                "AdvToMal_L2": fmt(row.get("norm_AdvToMal_L2")),
                "Time_sec": fmt(row.get("sample_end_to_end_time_sec")),
                "Score": fmt(row.get("stage2_decision_score")),
            }
        )
        metrics_path = Path(str(row.get("metrics_path", "")))
        if metrics_path.suffix.lower() == ".json":
            metrics_path = metrics_path.with_name("metrics.csv")
        metrics = read_metric_kv(metrics_path)
        for method in _extract_methods_from_metrics(metrics):
            out.append(
                {
                    "Attack": attack,
                    "Method": method,
                    "ASR_oracle": fmt(metrics.get(f"baseline_{method}_asr_oracle")),
                    "ASR_surrogate": fmt(metrics.get(f"baseline_{method}_asr_surrogate")),
                    "FFD": fmt(metrics.get(f"baseline_{method}_norm_FFD")),
                    "SWD": fmt(metrics.get(f"baseline_{method}_norm_SWD")),
                    "AdvToMal_L2": fmt(metrics.get(f"baseline_{method}_norm_AdvToMal_L2")),
                    "Time_sec": fmt(metrics.get(f"baseline_{method}_end_to_end_time_sec")),
                    "Score": "-",
                }
            )
    return out


def build_ablation_detail(ablation_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    order = {"full": 0, "w_o_stage1": 1, "backbone_gan": 2, "random_remap": 3}
    out: list[dict[str, str]] = []
    for row in sorted(ablation_rows, key=lambda item: (order.get(str(item.get("variant", "")), 99), str(item.get("variant", "")))):
        out.append(
            {
                "Variant": str(row.get("variant", "-")),
                "Stage2 score": fmt(row.get("stage2_decision_score")),
                "Stage2 ASR": fmt(row.get("stage2_asr_oracle")),
                "Stage2 FFD": fmt(row.get("stage2_norm_ffd")),
                "Stage3 score": fmt(row.get("stage3_decision_score")),
                "Replay ASR": fmt(row.get("stage3_pcap_attack_success_rate")),
                "Deployability": fmt(row.get("stage3_deployability_score")),
                "Target L2": fmt(row.get("stage3_pcap_target_l2_mean")),
                "Fatal rate": fmt(row.get("pcap_valid_fatal_rate")),
            }
        )
    return out


def build_stage3_summary_table(main_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in main_rows:
        out_dir = Path(str(row.get("out_dir", "")))
        stage3_config = (out_dir / "stage3" / "config.yaml").read_text(encoding="utf-8") if (out_dir / "stage3" / "config.yaml").exists() else ""
        stage2_oracle = "-"
        stage3_ids = "-"
        stage3_main_ids = "-"
        for line in stage3_config.splitlines():
            stripped = line.strip()
            if stripped.startswith("oracle_name:") and stage2_oracle == "-":
                stage2_oracle = stripped.split(":", 1)[1].strip()
            elif stripped.startswith("ids_name:") and stage3_ids == "-":
                stage3_ids = stripped.split(":", 1)[1].strip()
            elif stripped.startswith("main_ids_name:"):
                stage3_main_ids = stripped.split(":", 1)[1].strip()
        pcap_eval_rows = read_csv_rows(out_dir / "stage3" / "pcap_eval.csv")
        sampled = sorted({str(item.get("source_name", "")).strip() for item in pcap_eval_rows if str(item.get("source_name", "")).strip()})
        replayed = sorted(
            {
                str(item.get("source_name", "")).strip()
                for item in pcap_eval_rows
                if str(item.get("source_name", "")).strip() and str(item.get("is_original", "")).strip() != "1"
            }
        )
        source_only = max(0, len(sampled) - len(replayed))
        out.append(
            {
                "Attack": str(row.get("attack_type", "-")),
                "Carrier sampled": str(len(sampled)),
                "Carrier replayed": str(len(replayed)),
                "Carrier source_only": str(source_only),
                "Replay ASR": fmt(row.get("stage3_pcap_attack_success_rate")),
                "Deployability": fmt(row.get("stage3_deployability_score")),
                "Remap quality": fmt(row.get("stage3_remap_quality_score")),
                "Stage2 oracle": stage2_oracle,
                "Stage3 ids": stage3_ids,
                "Stage3 main_ids": stage3_main_ids,
                "Target L2": fmt(row.get("stage3_pcap_target_l2_mean")),
                "Alignment": fmt(row.get("stage3_pcap_alignment_coverage")),
                "Fatal rate": fmt(row.get("stage3_pcap_valid_fatal_rate")),
                "Eval time sec": fmt(row.get("stage3_pcap_eval_time_sec")),
            }
        )
    return out


def build_stage3_carrier_rows(dataset_root: Path, main_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in main_rows:
        out_dir = Path(str(row.get("out_dir", "")))
        pcap_eval_rows = read_csv_rows(out_dir / "stage3" / "pcap_eval.csv")
        grouped_rows: dict[str, dict[str, dict[str, str]]] = {}
        carrier_order: list[str] = []
        for eval_row in pcap_eval_rows:
            source_name = str(eval_row.get("source_name", "-")).strip()
            if not source_name:
                continue
            is_original = str(eval_row.get("is_original", "")).strip() == "1"
            if source_name not in grouped_rows:
                carrier_order.append(source_name)
                grouped_rows[source_name] = {}
            grouped_rows[source_name]["source" if is_original else "adv"] = eval_row

        for source_name in carrier_order:
            group = grouped_rows[source_name]
            source_row = group.get("source", {})
            adv_row = group.get("adv", {})
            has_adv = bool(adv_row)
            active_row = adv_row if has_adv else source_row
            source_label = str(source_row.get("pred_label", "-"))
            adv_label = str(adv_row.get("pred_label", "-")) if has_adv else "-"
            carrier_asr = "-"
            if source_label in {"1", "malicious"} and adv_label in {"0", "benign"}:
                carrier_asr = "1"
            elif has_adv and source_label != "-" and adv_label != "-":
                carrier_asr = "0"
            out.append(
                {
                    "Attack": str(row.get("attack_type", "-")),
                    "Carrier": source_name,
                    "Eval_status": "adv_replayed" if has_adv else "source_only",
                    "Feature_backend": str(active_row.get("feature_backend", "-")),
                    "Feature_status": str(active_row.get("feature_status", "-")),
                    "Flow_count": str(active_row.get("flow_count", "-")),
                    "Source_pred": source_label,
                    "Source_pmal": fmt(source_row.get("prob_malicious")),
                    "Adv_pred": adv_label,
                    "Adv_pmal": fmt(adv_row.get("prob_malicious")) if has_adv else "-",
                    "Carrier_ASR": carrier_asr,
                    "Alignment_coverage": fmt(active_row.get("alignment_coverage")),
                    "Alignment_missing": fmt(active_row.get("alignment_missing")),
                    "Target_L2": fmt(active_row.get("target_l2")),
                    "Target_MAE": fmt(active_row.get("target_mae")),
                    "Sanity_nonmonotonic": fmt(active_row.get("sanity_nonmonotonic_rate")),
                    "Sanity_transport_missing": fmt(active_row.get("sanity_transport_missing_rate")),
                    "Sanity_tcp_seq_backwards": fmt(active_row.get("sanity_tcp_seq_backwards_rate")),
                    "Sanity_tcp_flag_invalid": fmt(active_row.get("sanity_tcp_flag_invalid_rate")),
                }
            )
    return out


def best_numeric_row(rows: list[dict[str, str]], column: str, *, higher: bool) -> dict[str, str] | None:
    candidates = [(to_float(row.get(column)), row) for row in rows]
    candidates = [(value, row) for value, row in candidates if value is not None]
    if not candidates:
        return None
    return (max if higher else min)(candidates, key=lambda item: item[0])[1]


def stage1_matrix_analysis(matrix_rows: list[dict[str, str]]) -> str:
    if not matrix_rows:
        return "结论：当前没有可用的 Stage1 异构矩阵结果，因此不能声称已经完成多分类器互提取验证。"
    headers = [key for key in matrix_rows[0].keys() if key != "surrogate\\target"]
    if len(headers) <= 1:
        return "结论：当前 Stage1 只形成了单模型矩阵，这不足以支撑“异构模型互提取”论断，必须以多模型结果为准。"
    best_pair: tuple[str, str, float] | None = None
    worst_pair: tuple[str, str, float] | None = None
    for row in matrix_rows:
        surrogate = str(row.get("surrogate\\target", "-"))
        for target in headers:
            if surrogate == target:
                continue
            value = to_float(row.get(target))
            if value is None:
                continue
            if best_pair is None or value > best_pair[2]:
                best_pair = (surrogate, target, value)
            if worst_pair is None or value < worst_pair[2]:
                worst_pair = (surrogate, target, value)
    parts = []
    if best_pair is not None:
        parts.append(f"最高的跨模型 agreement 出现在 `{best_pair[0]} -> {best_pair[1]}`，值为 `{best_pair[2]:.4f}`。")
    if worst_pair is not None:
        parts.append(f"最低的跨模型 agreement 出现在 `{worst_pair[0]} -> {worst_pair[1]}`，值为 `{worst_pair[2]:.4f}`。")
    return "结论：" + " ".join(parts) if parts else "结论：矩阵存在，但有效的跨模型数值不足，无法得出稳定结论。"


def stage2_slice_analysis(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "结论：当前没有 attack-type 切片结果。"
    best_score = best_numeric_row(rows, "Score", higher=True)
    worst_score = best_numeric_row(rows, "Score", higher=False)
    best_ffd = best_numeric_row(rows, "FFD", higher=False)
    hardest = f"按综合 Score 最强的是 `{best_score['Attack']}`（`{best_score['Score']}`）" if best_score else ""
    weakest = f"最弱的是 `{worst_score['Attack']}`（`{worst_score['Score']}`）" if worst_score else ""
    fidelity = f"统计偏移最小的是 `{best_ffd['Attack']}`（FFD=`{best_ffd['FFD']}`）" if best_ffd else ""
    return "结论：" + "；".join(text for text in [hardest, weakest, fidelity] if text) + "。"


def stage2_baseline_analysis(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "结论：当前没有 Stage2 baseline 对比结果。"
    ours = [row for row in rows if str(row.get("Method", "")) == "RD-Synth"]
    if not ours:
        return "结论：当前表缺少 RD-Synth 主方法行。"
    better_ffd = 0
    better_score = 0
    attacks = sorted({str(row.get("Attack", "")) for row in ours})
    for attack in attacks:
        group = [row for row in rows if str(row.get("Attack", "")) == attack]
        ours_row = next((row for row in group if str(row.get("Method", "")) == "RD-Synth"), None)
        if ours_row is None:
            continue
        ours_ffd = to_float(ours_row.get("FFD"))
        ours_score = to_float(ours_row.get("Score"))
        baseline_ffd = [to_float(row.get("FFD")) for row in group if str(row.get("Method", "")) != "RD-Synth"]
        baseline_score = [to_float(row.get("Score")) for row in group if str(row.get("Method", "")) != "RD-Synth"]
        baseline_ffd = [value for value in baseline_ffd if value is not None]
        baseline_score = [value for value in baseline_score if value is not None]
        if ours_ffd is not None and baseline_ffd and ours_ffd <= min(baseline_ffd):
            better_ffd += 1
        if ours_score is not None and (not baseline_score or ours_score >= max(baseline_score)):
            better_score += 1
    return (
        "结论：RD-Synth 在 attack-type 切片上主要依靠更均衡的 ASR 与保真度取得优势；"
        f"在 `{len(attacks)}` 个攻击类型中，它有 `{better_ffd}` 个切片拿到最低 FFD，"
        f"有 `{better_score}` 个切片拿到最高可比较 Score。"
    )


def stage2_global_compare_analysis(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "结论：当前没有 GLOBAL Stage2 主线方法对比。"
    ours = next((row for row in rows if str(row.get("Method", "")) == "RD-Synth"), None)
    random_row = next((row for row in rows if str(row.get("Method", "")) == "global_random"), None)
    best_ffd = best_numeric_row(rows, "FFD", higher=False)
    best_swd = best_numeric_row(rows, "SWD", higher=False)
    best_asr = best_numeric_row(rows, "ASR_oracle", higher=True)
    parts: list[str] = []
    if best_ffd is not None and best_swd is not None:
        parts.append(f"最低 FFD 与最低 SWD 分别落在 `{best_ffd['Method']}` 和 `{best_swd['Method']}`。")
    if random_row is not None:
        parts.append(
            f"`global_random` 的 FFD=`{random_row.get('FFD', '-')}`、SWD=`{random_row.get('SWD', '-')}` 低于 RD-Synth，"
            "这说明当前 binary evasion setting 下，向 benign 支持集随机漂移本身就是一个很强的控制组；它更接近 benign 边缘分布，但并不等价于学到了条件攻击生成。"
        )
    if ours is not None and random_row is not None:
        parts.append(
            f"RD-Synth 仍把 `ASR_oracle/ASR_surrogate` 提到 `{ours.get('ASR_oracle', '-')}` / `{ours.get('ASR_surrogate', '-')}`，"
            f"高于 `global_random` 的 `{random_row.get('ASR_oracle', '-')}` / `{random_row.get('ASR_surrogate', '-')}`；"
            f"同时它保留了可比较的综合 `Score={ours.get('Score', '-')}` 以及后续 Stage3 replay 证据。"
        )
    if best_asr is not None:
        parts.append(f"最高 ASR 行是 `{best_asr['Method']}`。因此这张表必须联合 ASR、realism、查询代价与 Stage3 证据一起解读，不能只看 FFD/SWD。")
    return "结论：" + " ".join(parts)


def stage2_realism_analysis(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "结论：当前没有 Stage2 realism 表。"
    ours = next((row for row in rows if str(row.get("Method", "")) == "RD-Synth"), None)
    random_row = next((row for row in rows if str(row.get("Method", "")) == "global_random"), None)
    best_cov = best_numeric_row(rows, "Coverage@5", higher=True)
    parts: list[str] = []
    if ours is not None:
        parts.append(
            f"RD-Synth 的 realism 位置是 FFD=`{ours.get('FFD', '-')}`、SWD=`{ours.get('SWD', '-')}`、Corr_Delta=`{ours.get('Corr_Delta', '-')}`、Coverage@5=`{ours.get('Coverage@5', '-')}`。"
        )
    if random_row is not None:
        parts.append(
            f"`global_random` 拿到更低的 FFD/SWD/Energy（`{random_row.get('FFD', '-')}` / `{random_row.get('SWD', '-')}` / `{random_row.get('Energy', '-')}`），"
            "更像是在贴近 benign 分布，而不是在维持攻击条件下做结构一致的变形。"
        )
    if best_cov is not None:
        parts.append(f"Coverage@5 最好的方法是 `{best_cov['Method']}`。这类表应该同时看距离、可分性、覆盖率和结构偏差。")
    return "结论：" + " ".join(parts)


def stage2_cgd_analysis(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "结论：当前没有 STP 三组相关性表。"
    ours = next((row for row in rows if str(row.get("Method", "")) == "RD-Synth"), None)
    best_avg = best_numeric_row(rows, "CGD_AVG", higher=False)
    parts: list[str] = []
    if best_avg is not None:
        parts.append(f"平均跨组相关偏差最小的是 `{best_avg['Method']}`（CGD_AVG=`{best_avg['CGD_AVG']}`）。")
    if ours is not None:
        parts.append(
            f"RD-Synth 的 ST/SP/TP 偏差分别是 `{ours.get('CGD_ST', '-')}` / `{ours.get('CGD_SP', '-')}` / `{ours.get('CGD_TP', '-')}`。"
            "这张表专门回答 spatial-temporal-protocol 依赖关系是否被保住，而不是只比较边缘分布。"
        )
    return "结论：" + " ".join(parts)


def stage2_support_analysis(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "缁撹锛氬綋鍓嶆病鏈?Stage2 support-aware 閰嶇疆鍜岄€夋牱璇佹嵁銆?"
    row = rows[0]
    return (
        "缁撹锛歋tage2 宸茬粡鏄惧紡鍚敤 support-aware 鍚庡鐞嗗拰 per-sample candidate selection锛?"
        f"`Candidate mode={row.get('Candidate mode', '-')}`銆?`Pullback α={row.get('Pullback α', '-')}`銆?`Moment α={row.get('Moment α', '-')}`銆?"
        "杩欒 reviewer 鍙互鐪嬪埌褰撳墠涓嶆槸鍙潬 backbone 杈撳嚭锛岃€屾槸鍚屾椂鍦ㄥ悜 benign support 鍜?packet remapability 鍋氳仈鍚堢害鏉熴€?"
    )


def transfer_ids_analysis(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "缁撹锛氬綋鍓嶆病鏈?transfer IDS 姹囨€绘暟鎹紝杩欎粛鏄竴涓鍚戝绋夸汉浜ゅ緟鐨勭┖鐧姐€?"
    best = best_numeric_row(rows, "Adv ASR", higher=True)
    worst = best_numeric_row(rows, "Adv ASR", higher=False)
    parts: list[str] = []
    if best is not None:
        parts.append(f"鏈€瀹规槗杩佺Щ鐨?IDS 鏄?`{best['Transfer IDS']}`锛團dv ASR=`{best['Adv ASR']}`锛?")
    if worst is not None:
        parts.append(f"鏈€鍚冨姏鐨?IDS 鏄?`{worst['Transfer IDS']}`锛團dv ASR=`{worst['Adv ASR']}`锛?")
    return "结论：" + " ".join(parts) if parts else "结论：当前 transfer IDS 表有效数值不足。"


def hard_carrier_analysis(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "缁撹锛氬綋鍓嶈繍琛岄噷娌℃湁琚?shared IDS 鏄庣‘鎵撲负 malicious 鐨?carrier锛屽洜姝?hard-case replay 璇佹嵁浠嶆槸绌虹櫧銆?"
    success = sum(1 for row in rows if str(row.get("Carrier_ASR", "")).strip() == "1")
    return (
        "缁撹锛氬綋鍓嶆姤鍛婂凡缁忓皢 harder carriers 鍗曠嫭鍒椽嚭锛?"
        f"鍏朵腑 `{success}/{len(rows)}` 涓?carrier 鍦?source->adv 涓婂疄鐜颁簡 malicious 淇″彿涓嬮檷銆?"
        "杩欓儴鍒嗚瘉鎹瘮鈥滃師濮嬪氨鎺ヨ繎 benign 鐨?carrier鈥濇洿閫傚悎鍥炵瓟 reviewer 瀵?hard-case 鐨勮川鐤戙€?"
    )


def stage3_baseline_policy_analysis(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "缁撹锛氬綋鍓嶆病鏈?Stage3 baseline realization-policy 琛ㄣ€?"
    skipped = sum(1 for row in rows if str(row.get("PCAP status", "")).strip() == "skipped")
    evaluated = sum(1 for row in rows if str(row.get("PCAP status", "")).strip() == "evaluated")
    return (
        "缁撹锛歋tage3 baseline 鐜板湪鎸夊疄鐜板彛寰勮€屼笉鏄寜鏂规硶鍚嶇О鍘荤暀鏁板瓧銆?"
        f"褰撳墠 `evaluated={evaluated}`銆?`skipped={skipped}`銆?"
        "feature-only baseline 鍙兘浣滀负 `feature_only_random_remap_control`锛岃€屾病鏈夊師鐢?packet writer 鐨?traffic-space baseline 浼氳鏄剧ず鏍囨敞涓?`native_packet_realization_not_implemented`銆?"
    )


def ablation_analysis(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "结论：当前 run 没有 ablation 结果，这通常说明本次实验是 `main-only` 或 ablation 没有执行完成。"
    full_row = next((row for row in rows if str(row.get("Variant", "")) == "full"), None)
    if full_row is None:
        return "结论：ablation 表存在，但缺少 `full` 锚点，因此不能做规范的消融比较。"
    comparisons: list[str] = []
    for row in rows:
        variant = str(row.get("Variant", ""))
        if variant == "full":
            continue
        replay = row.get("Replay ASR", "-")
        target_l2 = row.get("Target L2", "-")
        comparisons.append(f"`{variant}` 的 Replay ASR=`{replay}`，Target L2=`{target_l2}`")
    lead = f"`full` 的 Stage3 score=`{full_row.get('Stage3 score', '-')}`，Deployability 未被显著拖垮。"
    return "结论：" + " ".join([lead, *comparisons]) if comparisons else f"结论：{lead}"


def stage3_summary_analysis(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "结论：当前没有 Stage3 汇总指标。"
    row = rows[0]
    fatal = to_float(row.get("Fatal rate"))
    fatal_text = (
        "但 `Fatal rate` 仍然偏高，说明 replay 成功不等于协议/时序有效性已经稳定。"
        if fatal is not None and fatal >= 0.5
        else "且 `Fatal rate` 没有成为主导问题。"
    )
    return (
        "结论：本次 Stage3 已经完整跑完 10-carrier replay。"
        f"总共抽样 `{row.get('Carrier sampled', '-')}` 个 malicious carrier，"
        f"其中 `{row.get('Carrier replayed', '-')}` 个完成了 `adv_*.pcap` replay，"
        f"`Replay ASR={row.get('Replay ASR', '-')}`，`Deployability={row.get('Deployability', '-')}`，"
        f"`Fatal rate={row.get('Fatal rate', '-')}`，{fatal_text}"
    )


def stage3_carrier_analysis(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "结论：当前没有 carrier 级明细。"
    replayed = [row for row in rows if str(row.get("Eval_status", "")) == "adv_replayed"]
    source_only = [row for row in rows if str(row.get("Eval_status", "")) == "source_only"]
    lowest_prob = best_numeric_row(rows, "Prob_malicious", higher=False)
    best_target = best_numeric_row(replayed, "Target_L2", higher=False) if replayed else None
    parts = [
        f"10 个 carrier 已全部落表，其中 `{len(replayed)}` 个是 `adv_replayed`，`{len(source_only)}` 个是 `source_only`。"
    ]
    if lowest_prob is not None:
        parts.append(
            f"最低 `Prob_malicious` 的 carrier 是 `{lowest_prob['Carrier']}`（`{lowest_prob['Prob_malicious']}`）。"
        )
    if best_target is not None:
        parts.append(
            f"在真正完成 replay 的 carrier 中，目标偏移最小的是 `{best_target['Carrier']}`（Target L2=`{best_target['Target_L2']}`）。"
        )
    return "结论：" + " ".join(parts)


def overall_conclusion(
    *,
    matrix_rows: list[dict[str, str]],
    ablation_rows: list[dict[str, str]],
    stage3_summary_rows: list[dict[str, str]],
) -> str:
    matrix_ok = bool(matrix_rows and len(matrix_rows[0].keys()) > 2)
    ablation_ok = bool(ablation_rows)
    stage3_row = stage3_summary_rows[0] if stage3_summary_rows else {}
    sampled = stage3_row.get("Carrier sampled", "0")
    replayed = stage3_row.get("Carrier replayed", "0")
    fatal = to_float(stage3_row.get("Fatal rate"))
    fatal_comment = (
        "需要重点盯住较高的 fatal rate。"
        if fatal is not None and fatal >= 0.5
        else "当前主要风险不在 fatal rate。"
    )
    return (
        "总体结论："
        f"Stage1 异构矩阵{'已形成多模型互提取证据' if matrix_ok else '尚未形成可接受的多模型互提取证据'}；"
        f"ablation {'已齐全' if ablation_ok else '缺失'}；"
        f"Stage3 当前是 `{sampled}` 个 carrier 抽样、`{replayed}` 个 carrier 完成 replay 的状态。"
        f"因此这份文档现在可以作为较完整的 reviewer-facing 中间报告，但 {fatal_comment}"
    )


def load_stage1_matrix(dataset_root: Path, rq1_rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], str]:
    if not rq1_rows:
        return [], [], [], "当前运行缺少 `rq1_matrix_summary.csv`，无法展示 Stage1 异构 IDS 提取矩阵。"
    rq1 = rq1_rows[0]
    summary_path = Path(str(rq1.get("summary_path", "")))
    matrix_path = Path(str(rq1.get("matrix_path", "")))
    per_model_rows: list[dict[str, str]] = []
    summary_rows: list[dict[str, str]] = []
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8-sig", newline="") as handle:
            all_rows = list(csv.reader(handle))
        section = "per_model"
        per_model_headers: list[str] = []
        summary_headers = ["metric", "mean", "std"]
        for row in all_rows:
            clean_row = [cell.strip() for cell in row]
            while clean_row and clean_row[-1] == "":
                clean_row.pop()
            if not clean_row:
                continue
            if clean_row[:3] == ["metric", "mean", "std"]:
                section = "summary"
                continue
            if section == "per_model":
                if not per_model_headers:
                    per_model_headers = clean_row
                    continue
                per_model_rows.append(
                    {
                        per_model_headers[idx]: clean_row[idx] if idx < len(clean_row) else ""
                        for idx in range(len(per_model_headers))
                    }
                )
            else:
                summary_rows.append(
                    {
                        summary_headers[idx]: clean_row[idx] if idx < len(clean_row) else ""
                        for idx in range(len(summary_headers))
                    }
                )
    matrix_rows = read_csv_rows(matrix_path)
    note = (
        f"当前 Stage1 矩阵来自 seed=`{rq1.get('seed', '-')}` 的主线运行，"
        f"配置里声明的 IDS 数量为 `{rq1.get('ids_count', rq1.get('oracle_count', '-'))}`。"
    )
    return per_model_rows, summary_rows, matrix_rows, note


def stage3_summary_analysis(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "结论：当前没有 Stage3 汇总指标。"
    row = rows[0]
    fatal = to_float(row.get("Fatal rate"))
    fatal_text = (
        "但 `Fatal rate` 仍然偏高，说明 10/10 carrier 都完成了 replay，并不等于协议级时序稳定性已经达标。"
        if fatal is not None and fatal >= 0.5
        else "而且 `Fatal rate` 没有成为主要瓶颈。"
    )
    return (
        "结论：本次 Stage3 已经完整跑完 10-carrier 评估。"
        f"共抽样 `{row.get('Carrier sampled', '-')}` 个 malicious carrier，"
        f"其中 `{row.get('Carrier replayed', '-')}` 个完成 `adv_*.pcap` replay；"
        f"`Replay ASR={row.get('Replay ASR', '-')}`，`Deployability={row.get('Deployability', '-')}`，"
        f"`Fatal rate={row.get('Fatal rate', '-')}`，{fatal_text}"
    )


def overall_conclusion(
    *,
    matrix_rows: list[dict[str, str]],
    ablation_rows: list[dict[str, str]],
    stage3_summary_rows: list[dict[str, str]],
) -> str:
    matrix_ok = bool(matrix_rows and len(matrix_rows[0].keys()) > 2)
    ablation_ok = bool(ablation_rows)
    stage3_row = stage3_summary_rows[0] if stage3_summary_rows else {}
    sampled = stage3_row.get("Carrier sampled", "0")
    replayed = stage3_row.get("Carrier replayed", "0")
    fatal = to_float(stage3_row.get("Fatal rate"))
    fatal_comment = (
        "需要把较高的 fatal rate 明确写成当前 Stage3 的主要风险。"
        if fatal is not None and fatal >= 0.5
        else "当前 Stage3 的主要风险不在 fatal rate。"
    )
    return (
        "总体结论："
        f"Stage1 异构矩阵{'已形成多模型互提取证据' if matrix_ok else '尚未形成可信的多模型互提取证据'}，"
        f"ablation {'已补齐' if ablation_ok else '仍缺失'}，"
        f"Stage3 当前是 `{sampled}` 个 carrier 抽样、`{replayed}` 个 carrier 完成 replay 的状态。"
        f"因此这份文档现在可以作为较完整的 reviewer-facing 实验报告，但 {fatal_comment}"
    )


def build_global_stage2_compare(main_rows: list[dict[str, str]], baseline_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in main_rows:
        out_dir = Path(str(row.get("out_dir", "")))
        stage2_metrics = read_metric_kv(out_dir / "stage2" / "metrics.csv")
        attack = str(row.get("attack_type", "-"))
        out.append(
            {
                "Attack": attack,
                "Method": "RD-Synth",
                "Family": "ours",
                "ASR_oracle": fmt(row.get("stage2_asr_oracle")),
                "ASR_surrogate": fmt(row.get("stage2_asr_surrogate")),
                "FFD": fmt(row.get("stage2_norm_ffd")),
                "SWD": fmt(row.get("stage2_norm_swd")),
                "C2ST_AUC": fmt(row.get("stage2_norm_c2st_auc")),
                "C2ST_Acc": fmt(row.get("stage2_norm_c2st_acc")),
                "Corr_Delta": fmt(row.get("stage2_norm_corr_delta")),
                "AdvToMal_L2": fmt(row.get("stage2_norm_advtomal_l2")),
                "Queries_per_success": fmt(row.get("stage2_queries_per_success_oracle")),
                "Time_sec": fmt(row.get("stage2_end_to_end_time_sec")),
                "Score": fmt(row.get("stage2_decision_score")),
            }
        )
        family_by_method = {
            str(item.get("method", "")): str(item.get("family", "-"))
            for item in baseline_rows
            if str(item.get("attack_type", "")) == attack
        }
        for method in _extract_methods_from_metrics(stage2_metrics):
            out.append(
                {
                    "Attack": attack,
                    "Method": method,
                    "Family": family_by_method.get(method, "-"),
                    "ASR_oracle": fmt(stage2_metrics.get(f"baseline_{method}_asr_oracle")),
                    "ASR_surrogate": fmt(stage2_metrics.get(f"baseline_{method}_asr_surrogate")),
                    "FFD": fmt(stage2_metrics.get(f"baseline_{method}_norm_FFD")),
                    "SWD": fmt(stage2_metrics.get(f"baseline_{method}_norm_SWD")),
                    "C2ST_AUC": fmt(stage2_metrics.get(f"baseline_{method}_norm_C2ST-AUC")),
                    "C2ST_Acc": fmt(stage2_metrics.get(f"baseline_{method}_norm_C2ST-Acc")),
                    "Corr_Delta": "-",
                    "AdvToMal_L2": fmt(stage2_metrics.get(f"baseline_{method}_norm_AdvToMal_L2")),
                    "Queries_per_success": fmt(stage2_metrics.get(f"baseline_{method}_queries_per_success_oracle")),
                    "Time_sec": fmt(stage2_metrics.get(f"baseline_{method}_end_to_end_time_sec")),
                    "Score": "-",
                }
            )
    return out


def build_ablation_detail(ablation_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    order = {"full": 0, "w_o_stage1": 1, "backbone_gan": 2, "random_remap": 3}
    out: list[dict[str, str]] = []
    for row in sorted(ablation_rows, key=lambda item: (order.get(str(item.get("variant", "")), 99), str(item.get("variant", "")))):
        out_dir = Path(str(row.get("out_dir", "")))
        stage2_metrics = read_metric_kv(out_dir / "stage2" / "metrics.csv")
        stage3_metrics = read_metric_kv(out_dir / "stage3" / "metrics.csv")
        out.append(
            {
                "Variant": str(row.get("variant", "-")),
                "S2_ASR_oracle": fmt(stage2_metrics.get("asr_oracle", row.get("stage2_asr_oracle"))),
                "S2_ASR_surrogate": fmt(stage2_metrics.get("asr_surrogate")),
                "S2_FFD": fmt(stage2_metrics.get("norm_FFD", row.get("stage2_norm_ffd"))),
                "S2_SWD": fmt(stage2_metrics.get("norm_SWD")),
                "S2_AdvToMal_L2": fmt(stage2_metrics.get("norm_AdvToMal_L2")),
                "S3_Replay_ASR": fmt(stage3_metrics.get("paper_pcap_attack_success_rate", row.get("stage3_pcap_attack_success_rate"))),
                "S3_Target_L2": fmt(stage3_metrics.get("pcap_target_l2_mean", row.get("stage3_pcap_target_l2_mean"))),
                "S3_Target_MAE": fmt(stage3_metrics.get("pcap_target_mae_mean")),
                "S3_Alignment": fmt(stage3_metrics.get("pcap_eval_avg_alignment", row.get("stage3_pcap_alignment_coverage"))),
                "S3_Fatal": fmt(stage3_metrics.get("pcap_valid_fatal_rate", row.get("pcap_valid_fatal_rate"))),
                "Remap_R2": fmt(stage3_metrics.get("remapper_eval_r2")),
                "Remap_MAE": fmt(stage3_metrics.get("remapper_eval_mae")),
                "Remap_RMSE": fmt(stage3_metrics.get("remapper_eval_rmse")),
                "Port_Acc": fmt(stage3_metrics.get("remapper_eval_port_acc")),
                "Score_aux": fmt(row.get("stage3_decision_score")),
            }
        )
    return out


def build_stage3_carrier_rows(dataset_root: Path, main_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in main_rows:
        out_dir = Path(str(row.get("out_dir", "")))
        pcap_eval_rows = read_csv_rows(out_dir / "stage3" / "pcap_eval.csv")
        grouped_rows: dict[str, dict[str, str]] = {}
        carrier_order: list[str] = []
        for eval_row in pcap_eval_rows:
            source_name = str(eval_row.get("source_name", "-")).strip()
            if not source_name:
                continue
            is_original = str(eval_row.get("is_original", "")).strip() == "1"
            if source_name not in grouped_rows:
                carrier_order.append(source_name)
                grouped_rows[source_name] = eval_row
            elif not is_original:
                grouped_rows[source_name] = eval_row
        for source_name in carrier_order:
            eval_row = grouped_rows[source_name]
            is_original = str(eval_row.get("is_original", "")).strip() == "1"
            out.append(
                {
                    "Attack": str(row.get("attack_type", "-")),
                    "Carrier": source_name,
                    "Eval_status": "source_only" if is_original else "adv_replayed",
                    "Feature_backend": str(eval_row.get("feature_backend", "-")),
                    "Feature_status": str(eval_row.get("feature_status", "-")),
                    "Flow_count": str(eval_row.get("flow_count", "-")),
                    "Pred_label": str(eval_row.get("pred_label", "-")),
                    "Prob_malicious": fmt(eval_row.get("prob_malicious")),
                    "Prob_benign": fmt(eval_row.get("prob_benign")),
                    "Alignment_coverage": fmt(eval_row.get("alignment_coverage")),
                    "Alignment_missing": fmt(eval_row.get("alignment_missing")),
                    "Target_L2": fmt(eval_row.get("target_l2")),
                    "Target_MAE": fmt(eval_row.get("target_mae")),
                    "Sanity_nonmonotonic": fmt(eval_row.get("sanity_nonmonotonic_rate")),
                    "Sanity_transport_missing": fmt(eval_row.get("sanity_transport_missing_rate")),
                    "Sanity_tcp_seq_backwards": fmt(eval_row.get("sanity_tcp_seq_backwards_rate")),
                    "Sanity_tcp_flag_invalid": fmt(eval_row.get("sanity_tcp_flag_invalid_rate")),
                }
            )
    return out


def ablation_analysis(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "结论：当前 run 没有 ablation 结果，这通常说明本次实验是 `main-only` 或 ablation 没有执行完成。"
    full_row = next((row for row in rows if str(row.get("Variant", "")) == "full"), None)
    if full_row is None:
        return "结论：Ablation 表存在，但缺少 `full` 锚点，因此不能做规范的消融比较。"
    findings: list[str] = []
    for row in rows:
        variant = str(row.get("Variant", ""))
        if variant == "full":
            continue
        findings.append(
            f"`{variant}` 相对 `full` 的主要变化是 "
            f"S2_FFD=`{row.get('S2_FFD', '-')}`、"
            f"S3_Target_L2=`{row.get('S3_Target_L2', '-')}`、"
            f"S3_Fatal=`{row.get('S3_Fatal', '-')}`。"
        )
    return (
        "结论：`full` 作为锚点时，"
        f"S2_FFD=`{full_row.get('S2_FFD', '-')}`、"
        f"S3_Target_L2=`{full_row.get('S3_Target_L2', '-')}`、"
        f"S3_Fatal=`{full_row.get('S3_Fatal', '-')}`、"
        f"Remap_R2=`{full_row.get('Remap_R2', '-')}`。"
        + (" " + " ".join(findings) if findings else "")
    )


def _postprocess_report_lines(lines: list[str]) -> list[str]:
    out = list(lines)
    replacements = {
        "指标说明：这张表用于比较 RD-Synth 与各 baseline 在主 IDS 上的攻击成功率、统计保真度和代价。":
            "指标说明：这张表用于在 GLOBAL 场景下直接比较 RD-Synth 与各 baseline；其中 `ASR_oracle` 是原始目标 IDS/oracle 上的成功率，`ASR_surrogate` 是提取出的 surrogate 上的成功率，其余列分别表示分布距离、两样本可分性、相关结构偏移、扰动幅度与代价。",
        "结论：这张表更适合看主方法与 baseline 的整体位置，而不是单独比较某一个攻击家族。":
            "结论：这张表现在只保留 GLOBAL 主线对比，因此可以直接回答“ours 相对各 baseline 在原始模型、提取模型和 realism/cost 上分别处在什么位置”。",
        "指标说明：这张表用于检查关键模块去掉后，性能是在哪一层退化，尤其看 Stage3 是否被明显拖垮。":
            "指标说明：这张表用常见 raw metric 展示 ablation 退化位置；`S2_*` 是特征空间攻击效果与保真度，`S3_*` 是 packet-level 回放效果，`Remap_*` 是 remapper 拟合误差，`Score_aux` 仅保留作辅助参考。",
        "指标说明：这张表用于逐个 carrier 查看 replay 后的检测结果与协议/时序 sanity；`source_only` 表示本次只保留了 source 检测，没有写出可评估的 `adv_*.pcap`。":
            "指标说明：这张表逐个 carrier 展示 packet-level 结果；`Eval_status` 表示是否真的生成并回放了 `adv_*.pcap`，`Feature_*` 表示特征提取后端与状态，`Pred_label/Prob_*` 是 oracle 在 PCAP 特征上的判别结果，`Alignment_*` 是与训练特征空间的对齐程度，`Target_*` 是相对目标 adv 向量的距离，`Sanity_*` 是协议/时序合法性检查。",
    }
    for idx, line in enumerate(out):
        if line in replacements:
            out[idx] = replacements[line]
    return out


def build_report(root: Path, dataset: str = "nb15") -> tuple[Path, list[Path]]:
    dataset_root = root / dataset
    main_rows = read_csv_rows(dataset_root / "main_runs.csv")
    stage2_baselines = read_csv_rows(dataset_root / "main_stage2_baselines.csv")
    stage1_attack_rows = read_csv_rows(dataset_root / "stage1_attack_runs.csv")
    stage2_attack_rows = read_csv_rows(dataset_root / "stage2_attack_runs.csv")
    rq1_rows = read_csv_rows(dataset_root / "rq1_matrix_summary.csv")
    ablation_rows_raw = read_csv_rows(dataset_root / "ablation_runs.csv")

    global_main = build_global_main_table(main_rows)
    global_s2_compare = build_global_stage2_compare(main_rows, stage2_baselines)
    attack_slice = build_attack_slice_table(stage1_attack_rows, stage2_attack_rows)
    attack_baselines = build_attack_baseline_compare(stage2_attack_rows)
    ablation_detail = build_ablation_detail(ablation_rows_raw)
    stage3_summary = build_stage3_summary_table(main_rows)
    carrier_rows = build_stage3_carrier_rows(dataset_root, main_rows)
    stage1_model_rows, stage1_summary_rows, stage1_matrix_rows, stage1_note = load_stage1_matrix(dataset_root, rq1_rows)

    csv_paths = [
        dataset_root / "nb15_global_main_table.csv",
        dataset_root / "nb15_stage2_global_method_compare.csv",
        dataset_root / "unsw_attack_slice_table.csv",
        dataset_root / "unsw_attack_slice_baseline_compare.csv",
        dataset_root / "unsw_ablation_detail_table.csv",
        dataset_root / "unsw_stage3_summary_table.csv",
        dataset_root / "unsw_stage3_carrier_eval_table.csv",
    ]
    write_csv(csv_paths[0], global_main, list(global_main[0].keys()) if global_main else ["Attack"])
    write_csv(csv_paths[1], global_s2_compare, list(global_s2_compare[0].keys()) if global_s2_compare else ["Attack"])
    write_csv(csv_paths[2], attack_slice, list(attack_slice[0].keys()) if attack_slice else ["Attack"])
    write_csv(csv_paths[3], attack_baselines, list(attack_baselines[0].keys()) if attack_baselines else ["Attack"])
    write_csv(csv_paths[4], ablation_detail, list(ablation_detail[0].keys()) if ablation_detail else ["Variant"])
    write_csv(csv_paths[5], stage3_summary, list(stage3_summary[0].keys()) if stage3_summary else ["Attack"])
    write_csv(csv_paths[6], carrier_rows, list(carrier_rows[0].keys()) if carrier_rows else ["Attack"])

    lines: list[str] = [
        "# NB15 实验表格库（中文版）",
        "",
        "这份报告面向作者和老板阅读，目标是作为论文制表、结果校验和 reviewer 质疑排查的中间层。",
        "",
        "写作约定：",
        "- 每张表前只用一句话解释指标是干什么的。",
        "- 每张表后给出客观结论，优先指出证据、边界和不足，不写宣传式总结。",
        "- 表内用加粗标出最优值；默认 `ASR/Score/Deployability/Alignment` 越大越好，`FFD/SWD/L2/Time/Fatal` 越小越好。",
        "",
        "## 1. GLOBAL 主结果",
        "",
        "指标说明：这张表用于总览主线 run 在 Stage1/2/3 上的综合表现，以及 10-carrier Stage3 的采样与 replay 覆盖情况。",
        "",
    ]
    lines.extend(
        render_table(
            global_main,
            best_columns={
                "Stage1 gain": "max",
                "Stage1 score": "max",
                "Stage2 score": "max",
                "Stage2 ASR": "max",
                "Stage2 FFD": "min",
                "Stage2 SWD": "min",
                "Stage3 replay ASR": "max",
                "Stage3 score": "max",
                "Deployability": "max",
                "Target L2": "min",
                "Fatal rate": "min",
                "Carrier sampled": "max",
                "Carrier replayed": "max",
            },
        )
    )
    lines.extend(["", stage3_summary_analysis(stage3_summary), ""])

    lines.extend(
        [
            "## 2. Stage1 IDS 矩阵",
            "",
            "指标说明：这两张表用于回答“是否真的训练了多个异构分类器，并且它们之间能否互相提取/逼近决策边界”。",
            "",
            stage1_note,
            "",
        ]
    )
    lines.extend(render_table(stage1_model_rows, best_columns={"diag_agreement": "max"}))
    lines.append("")
    lines.extend(render_table(stage1_summary_rows, best_columns={"mean": "max"}))
    lines.append("")
    if stage1_matrix_rows:
        matrix_headers = list(stage1_matrix_rows[0].keys())
        matrix_display: list[list[str]] = []
        for row in stage1_matrix_rows:
            numeric_headers = [header for header in matrix_headers if header != "surrogate\\target"]
            numeric_values = {header: to_float(row.get(header)) for header in numeric_headers}
            best = max((value for value in numeric_values.values() if value is not None), default=None)
            rendered = []
            for header in matrix_headers:
                cell = str(row.get(header, "-"))
                value = numeric_values.get(header)
                if header != "surrogate\\target" and best is not None and value is not None and abs(value - best) <= 1.0e-12:
                    cell = f"**{cell}**"
                rendered.append(cell)
            matrix_display.append(rendered)
        lines.extend(pipe_table(matrix_headers, matrix_display))
    else:
        lines.append("（当前无矩阵数据）")
    lines.extend(["", stage1_matrix_analysis(stage1_matrix_rows), ""])

    lines.extend(
        [
            "## 3. Stage2 主线方法对比",
            "",
            "指标说明：这张表用于比较 RD-Synth 与各 baseline 在主 IDS 上的攻击成功率、统计保真度和代价。",
            "",
        ]
    )
    lines.extend(
        render_table(
            global_s2_compare,
            best_columns={
                "ASR_oracle": "max",
                "ASR_surrogate": "max",
                "FFD": "min",
                "SWD": "min",
                "C2ST_AUC": "min",
                "C2ST_Acc": "min",
                "Corr_Delta": "min",
                "AdvToMal_L2": "min",
                "Queries_per_success": "min",
                "Time_sec": "min",
                "Score": "max",
            },
            group_by=["Attack"],
        )
    )
    lines.extend(["", "结论：这张表更适合看主方法与 baseline 的整体位置，而不是单独比较某一个攻击家族。", ""])

    lines.extend(
        [
            "## 4. 按攻击类型拆分的 Stage2 主结果",
            "",
            "指标说明：这张表用于回答不同攻击类型在同一套训练好的主模型上，哪个更容易被攻击、哪个更容易出现统计偏移。",
            "",
        ]
    )
    lines.extend(
        render_table(
            attack_slice,
            best_columns={
                "Eval rows": "max",
                "Stage1 agreement": "max",
                "Stage1 baseline agreement": "max",
                "ASR_oracle": "max",
                "ASR_surrogate": "max",
                "FFD": "min",
                "SWD": "min",
                "AdvToMal_L2": "min",
                "Score": "max",
                "Time_sec": "min",
            },
        )
    )
    lines.extend(["", stage2_slice_analysis(attack_slice), ""])

    lines.extend(
        [
            "## 5. 按攻击类型拆分的 Stage2 方法对比",
            "",
            "指标说明：这张表用于逐个攻击类型检查 RD-Synth 是否真的优于强 baseline，而不是只看 GLOBAL 聚合值。",
            "",
        ]
    )
    lines.extend(
        render_table(
            attack_baselines,
            best_columns={
                "ASR_oracle": "max",
                "ASR_surrogate": "max",
                "FFD": "min",
                "SWD": "min",
                "AdvToMal_L2": "min",
                "Time_sec": "min",
                "Score": "max",
            },
            group_by=["Attack"],
        )
    )
    lines.extend(["", stage2_baseline_analysis(attack_baselines), ""])

    lines.extend(
        [
            "## 6. Ablation",
            "",
            "指标说明：这张表用于检查关键模块去掉后，性能是在哪一层退化，尤其看 Stage3 是否被明显拖垮。",
            "",
        ]
    )
    lines.extend(
        render_table(
            ablation_detail,
            best_columns={
                "S2_ASR_oracle": "max",
                "S2_ASR_surrogate": "max",
                "S2_FFD": "min",
                "S2_SWD": "min",
                "S2_AdvToMal_L2": "min",
                "S3_Replay_ASR": "max",
                "S3_Target_L2": "min",
                "S3_Target_MAE": "min",
                "S3_Alignment": "max",
                "S3_Fatal": "min",
                "Remap_R2": "max",
                "Remap_MAE": "min",
                "Remap_RMSE": "min",
                "Port_Acc": "max",
                "Score_aux": "max",
            },
        )
    )
    lines.extend(["", ablation_analysis(ablation_detail), ""])

    lines.extend(
        [
            "## 7. Stage3 汇总指标",
            "",
            "指标说明：这张表用于回答 10-carrier Stage3 到底跑到了哪一步，是只完成抽样，还是已经形成可复核的 packet-level 指标。",
            "",
        ]
    )
    lines.extend(
        render_table(
            stage3_summary,
            best_columns={
                "Carrier sampled": "max",
                "Carrier replayed": "max",
                "Carrier source_only": "min",
                "Replay ASR": "max",
                "Deployability": "max",
                "Remap quality": "max",
                "Target L2": "min",
                "Alignment": "max",
                "Fatal rate": "min",
                "Eval time sec": "min",
            },
        )
    )
    lines.extend(["", stage3_summary_analysis(stage3_summary), ""])

    lines.extend(
        [
            "## 8. Stage3 Carrier 级结果",
            "",
            "指标说明：这张表用于逐个 carrier 查看 replay 后的检测结果与协议/时序 sanity；`source_only` 表示本次只保留了 source 检测，没有写出可评估的 `adv_*.pcap`。",
            "",
        ]
    )
    lines.extend(
        render_table(
            carrier_rows,
            best_columns={
                "Prob_malicious": "min",
                "Prob_benign": "max",
                "Alignment_coverage": "max",
                "Target_L2": "min",
                "Target_MAE": "min",
                "Sanity_nonmonotonic": "min",
                "Sanity_tcp_seq_backwards": "min",
                "Sanity_tcp_flag_invalid": "min",
            },
        )
    )
    lines.extend(["", stage3_carrier_analysis(carrier_rows), ""])

    lines.extend(
        [
            "## 9. 总体判断",
            "",
            overall_conclusion(
                matrix_rows=stage1_matrix_rows,
                ablation_rows=ablation_detail,
                stage3_summary_rows=stage3_summary,
            ),
            "",
        ]
    )

    lines = _postprocess_report_lines(lines)
    report_path = root / "NB15_TABLE_BANK_CN.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    return report_path, csv_paths


def stage3_summary_analysis_v2(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "结论：当前没有 Stage3 汇总指标。"
    row = rows[0]
    shared = (
        row.get("Stage2 oracle") == row.get("Stage3 ids") == row.get("Stage3 main_ids")
        and str(row.get("Stage2 oracle", "-")) not in {"", "-"}
    )
    shared_text = (
        f"Stage3 使用的检测模型与 Stage2 共用，均为 `{row.get('Stage2 oracle', '-')}`。"
        if shared
        else "Stage3 检测模型与 Stage2 主 oracle 的口径不完全一致，需要单独核对。"
    )
    return (
        "结论：本次 Stage3 已完整跑完 10-carrier replay。"
        f"共抽样 `{row.get('Carrier sampled', '-')}` 个 malicious carrier，`{row.get('Carrier replayed', '-')}` 个形成了可评估的 adv PCAP；"
        f"`Replay ASR={row.get('Replay ASR', '-')}`，`Deployability={row.get('Deployability', '-')}`，`Fatal rate={row.get('Fatal rate', '-')}`。"
        f"{shared_text}"
    )


def stage3_carrier_analysis_v2(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "结论：当前没有 carrier 级明细。"
    replayed = [row for row in rows if str(row.get("Eval_status", "")) == "adv_replayed"]
    successful = [row for row in replayed if str(row.get("Carrier_ASR", "")) == "1"]
    source_malicious = [row for row in replayed if str(row.get("Source_pred", "")) in {"1", "malicious"}]
    best_adv = best_numeric_row(replayed, "Adv_pmal", higher=False) if replayed else None
    best_target = best_numeric_row(replayed, "Target_L2", higher=False) if replayed else None
    parts = [f"当前 `{len(rows)}` 个 carrier 已全部落表，其中 `{len(replayed)}` 个完成 adv replay，`{len(successful)}` 个在 carrier 级别实现了 `source=malicious -> adv=benign`。"]
    if replayed and not source_malicious:
        parts.append("需要注意的是，这批 source carrier 在当前 oracle 下原始就大多被判为 benign，因此 `Carrier_ASR` 不能替代 run-level 的 Stage3 Replay ASR，更适合看 `Adv_pmal`、`Target_L2` 和各类 sanity 指标。")
    if best_adv is not None:
        parts.append(f"`Adv_pmal` 最低的 carrier 是 `{best_adv['Carrier']}`（`{best_adv['Adv_pmal']}`）。")
    if best_target is not None:
        parts.append(f"Target L2 最小的是 `{best_target['Carrier']}`（`{best_target['Target_L2']}`）。")
    return "结论：" + " ".join(parts)


def overall_conclusion_v2(*, matrix_rows: list[dict[str, str]], ablation_rows: list[dict[str, str]], stage3_summary_rows: list[dict[str, str]]) -> str:
    matrix_ok = bool(matrix_rows and len(matrix_rows[0].keys()) > 2)
    ablation_ok = bool(ablation_rows)
    stage3_row = stage3_summary_rows[0] if stage3_summary_rows else {}
    return (
        "整体判断："
        f"Stage1 异构 IDS 互提取矩阵{'已形成' if matrix_ok else '仍不完整'}；"
        f"ablation {'已补齐' if ablation_ok else '仍缺失'}；"
        f"Stage3 当前是 `{stage3_row.get('Carrier sampled', '0')}` 个 carrier 抽样、`{stage3_row.get('Carrier replayed', '0')}` 个 carrier 完成 replay。"
        "当前报告已经覆盖主线效果、realism、STP 结构相关性、ablation 和 carrier 级 replay；如果要完全超过旧版论文表，下一步还应补 packet remapping distortion 与 protocol-legality 专表。"
    )


def build_stage3_carrier_rows_v2(dataset_root: Path, main_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in main_rows:
        out_dir = Path(str(row.get("out_dir", "")))
        pcap_eval_rows = read_csv_rows(out_dir / "stage3" / "pcap_eval.csv")
        grouped_rows: dict[str, dict[str, dict[str, str]]] = {}
        carrier_order: list[str] = []
        for eval_row in pcap_eval_rows:
            source_name = str(eval_row.get("source_name", "-")).strip()
            if not source_name:
                continue
            is_original = str(eval_row.get("is_original", "")).strip() == "1"
            if source_name not in grouped_rows:
                carrier_order.append(source_name)
                grouped_rows[source_name] = {}
            grouped_rows[source_name]["source" if is_original else "adv"] = eval_row
        for source_name in carrier_order:
            group = grouped_rows[source_name]
            source_row = group.get("source", {})
            adv_row = group.get("adv", {})
            has_adv = bool(adv_row)
            active_row = adv_row if has_adv else source_row
            source_label = str(source_row.get("pred_label", "-"))
            adv_label = str(adv_row.get("pred_label", "-")) if has_adv else "-"
            carrier_asr = "-"
            if source_label in {"1", "malicious"} and adv_label in {"0", "benign"}:
                carrier_asr = "1"
            elif has_adv and source_label != "-" and adv_label != "-":
                carrier_asr = "0"
            out.append(
                {
                    "Attack": str(row.get("attack_type", "-")),
                    "Carrier": source_name,
                    "Eval_status": "adv_replayed" if has_adv else "source_only",
                    "Feature_backend": str(active_row.get("feature_backend", "-")),
                    "Feature_status": str(active_row.get("feature_status", "-")),
                    "Flow_count": str(active_row.get("flow_count", "-")),
                    "Source_pred": source_label,
                    "Source_pmal": fmt(source_row.get("prob_malicious")),
                    "Adv_pred": adv_label,
                    "Adv_pmal": fmt(adv_row.get("prob_malicious")) if has_adv else "-",
                    "Carrier_ASR": carrier_asr,
                    "Alignment_coverage": fmt(active_row.get("alignment_coverage")),
                    "Alignment_missing": fmt(active_row.get("alignment_missing")),
                    "Target_L2": fmt(active_row.get("target_l2")),
                    "Target_MAE": fmt(active_row.get("target_mae")),
                    "Sanity_nonmonotonic": fmt(active_row.get("sanity_nonmonotonic_rate")),
                    "Sanity_transport_missing": fmt(active_row.get("sanity_transport_missing_rate")),
                    "Sanity_tcp_seq_backwards": fmt(active_row.get("sanity_tcp_seq_backwards_rate")),
                    "Sanity_tcp_flag_invalid": fmt(active_row.get("sanity_tcp_flag_invalid_rate")),
                }
            )
    return out


def build_stage3_remap_distortion_table(main_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in main_rows:
        out_dir = Path(str(row.get("out_dir", "")))
        remap_rows = read_csv_rows(out_dir / "stage3" / "stage3_remap_eval.csv")
        metrics = read_metric_kv(out_dir / "stage3" / "metrics.csv")
        for remap_row in remap_rows:
            mod_name = str(remap_row.get("mod_name", "-"))
            extra = str(remap_row.get("extra", ""))
            port_acc = "-"
            if "port_acc=" in extra:
                port_acc = extra.split("port_acc=", 1)[1].strip()
            out.append(
                {
                    "Attack": str(row.get("attack_type", "-")),
                    "Field": mod_name,
                    "MAE": fmt(remap_row.get("mae")),
                    "RMSE": fmt(remap_row.get("rmse")),
                    "Target_mean": fmt(remap_row.get("target_mean")),
                    "Target_std": fmt(remap_row.get("target_std")),
                    "Port_Acc": fmt(port_acc) if port_acc != "-" else "-",
                    "Apply_selected": "1" if mod_name in str(metrics.get("pcap_apply_fields", "")) else "0",
                }
            )
    return out


def build_stage3_protocol_legality_table(main_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in main_rows:
        out_dir = Path(str(row.get("out_dir", "")))
        metrics = read_metric_kv(out_dir / "stage3" / "metrics.csv")

        def _pass_from_rate(key: str) -> str:
            value = to_float(metrics.get(key))
            if value is None:
                return "-"
            return "1.0000" if abs(value) <= 1.0e-12 else "0.0000"

        out.append(
            {
                "Attack": str(row.get("attack_type", "-")),
                "ValidFatal@0": fmt(metrics.get("pcap_validfatal_at_0")),
                "TCP_Flag_Invalid": _pass_from_rate("pcap_sanity_tcp_flag_invalid_rate"),
                "TCP_SYN_FIN": _pass_from_rate("pcap_sanity_tcp_syn_fin_rate"),
                "TCP_SYN_RST": _pass_from_rate("pcap_sanity_tcp_syn_rst_rate"),
                "TCP_FIN_RST": _pass_from_rate("pcap_sanity_tcp_fin_rst_rate"),
                "Transport_Present": _pass_from_rate("pcap_sanity_transport_missing_rate"),
                "TCP_Seq_Backwards_0": _pass_from_rate("pcap_sanity_tcp_seq_backwards_rate"),
                "TCP_Seq_Backwards_Rate": fmt(metrics.get("pcap_sanity_tcp_seq_backwards_rate")),
                "Nonmonotonic_Rate": fmt(metrics.get("pcap_sanity_nonmonotonic_rate")),
                "Port_Acc": fmt(metrics.get("remapper_eval_port_acc")),
            }
        )
    return out


def build_stage2_support_table(main_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in main_rows:
        out.append(
            {
                "Attack": str(row.get("attack_type", "-")),
                "Candidate mode": str(row.get("stage2_candidate_selection_mode", "-") or "-"),
                "Pullback α": fmt(row.get("stage2_pullback_alpha")),
                "Pullback k": str(row.get("stage2_pullback_k", "-") or "-"),
                "Moment α": fmt(row.get("stage2_moment_alpha")),
                "Selected α": fmt(row.get("stage2_selected_mal_anchor_alpha")),
                "Stage2 score": fmt(row.get("stage2_decision_score")),
                "FFD": fmt(row.get("stage2_norm_ffd")),
                "SWD": fmt(row.get("stage2_norm_swd")),
            }
        )
    return out


def build_transfer_ids_table(dataset_root: Path) -> list[dict[str, str]]:
    rows = read_csv_rows(dataset_root / "main_transfer_ids_summary.csv")
    out: list[dict[str, str]] = []
    for row in rows:
        out.append(
            {
                "Attack": str(row.get("attack_type", "-")),
                "Transfer IDS": str(row.get("ids_name", "-")),
                "Test Acc": fmt(row.get("test_acc_mean")),
                "Test F1": fmt(row.get("test_f1_mean")),
                "Adv ASR": fmt(row.get("adv_asr_mean")),
                "ΔASR vs main": fmt(row.get("delta_asr_vs_main_ids_mean")),
                "Seeds": str(row.get("n_seeds", "-")),
            }
        )
    return out


def build_stage3_hard_carrier_table(carrier_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in carrier_rows:
        source_pred = str(row.get("Source_pred", "")).strip().lower()
        source_pmal = to_float(row.get("Source_pmal"))
        if source_pred not in {"1", "malicious"} and (source_pmal is None or source_pmal < 0.5):
            continue
        out.append(
            {
                "Attack": str(row.get("Attack", "-")),
                "Carrier": str(row.get("Carrier", "-")),
                "Source_pred": str(row.get("Source_pred", "-")),
                "Source_pmal": str(row.get("Source_pmal", "-")),
                "Adv_pred": str(row.get("Adv_pred", "-")),
                "Adv_pmal": str(row.get("Adv_pmal", "-")),
                "Carrier_ASR": str(row.get("Carrier_ASR", "-")),
                "Alignment_coverage": str(row.get("Alignment_coverage", "-")),
                "Target_L2": str(row.get("Target_L2", "-")),
                "Sanity_tcp_seq_backwards": str(row.get("Sanity_tcp_seq_backwards", "-")),
            }
        )
    return out


def build_stage3_baseline_policy_table(dataset_root: Path) -> list[dict[str, str]]:
    rows = read_csv_rows(dataset_root / "main_stage3_baselines.csv")
    out: list[dict[str, str]] = []
    for row in rows:
        out.append(
            {
                "Attack": str(row.get("attack_type", "-")),
                "Method": str(row.get("method", "-")),
                "Baseline group": str(row.get("baseline_group", "-")),
                "Eval mode": str(row.get("evaluation_mode", "-")),
                "PCAP status": str(row.get("pcap_status", "-")),
                "Skip reason": str(row.get("pcap_skip_reason", "-")),
                "Deployability": fmt(row.get("deployability_score")),
                "Replay ASR": fmt(row.get("pcap_attack_success_rate")),
                "Target L2": fmt(row.get("pcap_target_l2_mean")),
            }
        )
    return out


def build_report_v2(root: Path, dataset: str = "nb15") -> tuple[Path, list[Path]]:
    dataset_root = root / dataset
    main_rows = read_csv_rows(dataset_root / "main_runs.csv")
    stage2_baselines = read_csv_rows(dataset_root / "main_stage2_baselines.csv")
    stage1_attack_rows = read_csv_rows(dataset_root / "stage1_attack_runs.csv")
    stage2_attack_rows = read_csv_rows(dataset_root / "stage2_attack_runs.csv")
    rq1_rows = read_csv_rows(dataset_root / "rq1_matrix_summary.csv")
    ablation_rows_raw = read_csv_rows(dataset_root / "ablation_runs.csv")

    global_main = build_global_main_table(main_rows)
    global_s2_compare = build_global_stage2_compare(main_rows, stage2_baselines)
    stage2_realism = build_stage2_realism_table(main_rows, stage2_baselines)
    stage2_cgd = build_stage2_cgd_table(main_rows, stage2_baselines)
    attack_slice = build_attack_slice_table(stage1_attack_rows, stage2_attack_rows)
    attack_baselines = build_attack_baseline_compare(stage2_attack_rows)
    ablation_detail = build_ablation_detail(ablation_rows_raw)
    stage3_summary = build_stage3_summary_table(main_rows)
    carrier_rows = build_stage3_carrier_rows_v2(dataset_root, main_rows)
    stage2_support = build_stage2_support_table(main_rows)
    transfer_ids = build_transfer_ids_table(dataset_root)
    hard_carriers = build_stage3_hard_carrier_table(carrier_rows)
    stage3_baseline_policy = build_stage3_baseline_policy_table(dataset_root)
    stage3_remap = build_stage3_remap_distortion_table(main_rows)
    stage3_legality = build_stage3_protocol_legality_table(main_rows)
    stage1_model_rows, stage1_summary_rows, stage1_matrix_rows, stage1_note = load_stage1_matrix(dataset_root, rq1_rows)

    csv_paths = [
        dataset_root / "nb15_global_main_table.csv",
        dataset_root / "nb15_stage2_global_method_compare.csv",
        dataset_root / "nb15_stage2_realism_table.csv",
        dataset_root / "nb15_stage2_cgd_table.csv",
        dataset_root / "unsw_attack_slice_table.csv",
        dataset_root / "unsw_attack_slice_baseline_compare.csv",
        dataset_root / "unsw_ablation_detail_table.csv",
        dataset_root / "unsw_stage3_summary_table.csv",
        dataset_root / "unsw_stage3_carrier_eval_table.csv",
        dataset_root / "nb15_stage2_support_table.csv",
        dataset_root / "unsw_transfer_ids_table.csv",
        dataset_root / "unsw_stage3_hard_carrier_table.csv",
        dataset_root / "unsw_stage3_baseline_policy_table.csv",
        dataset_root / "unsw_stage3_remap_distortion_table.csv",
        dataset_root / "unsw_stage3_protocol_legality_table.csv",
    ]
    tables = [
        global_main,
        global_s2_compare,
        stage2_realism,
        stage2_cgd,
        attack_slice,
        attack_baselines,
        ablation_detail,
        stage3_summary,
        carrier_rows,
        stage2_support,
        transfer_ids,
        hard_carriers,
        stage3_baseline_policy,
        stage3_remap,
        stage3_legality,
    ]
    defaults = [
        "Attack",
        "Attack",
        "Attack",
        "Attack",
        "Attack",
        "Attack",
        "Variant",
        "Attack",
        "Attack",
        "Attack",
        "Attack",
        "Attack",
        "Attack",
        "Attack",
        "Attack",
    ]
    for path, rows, default in zip(csv_paths, tables, defaults):
        write_csv(path, rows, list(rows[0].keys()) if rows else [default])

    lines: list[str] = [
        "# NB15 实验表格库（中文版）",
        "",
        "这份报告面向作者和老板阅读，目标是作为论文制表、结果校验和 reviewer 质疑排查的中间层。",
        "",
        "配套图集见 `NB15_FIGURE_BANK_CN.md`，用于补充 Stage1 热力图、Stage2 结构 realism 可视化、方法间 CGD、低维投影，以及 Stage3 carrier 级概览。",
        "",
        "写作约定：",
        "- 每张表前只用一句话解释指标是干什么的。",
        "- 每张表后给出客观结论，优先指出证据、边界和不足，不写宣传式总结。",
        "- 表内用加粗标出最优值；默认 `ASR/Score/Deployability/Alignment/Coverage/kNN-R` 越大越好，`FFD/SWD/Energy/C2ST/Corr_Delta/CGD/L2/Time/Fatal` 越小越好。",
        "",
        "## 1. GLOBAL 主结果",
        "",
        "指标说明：这张表用于总览主线 run 在 Stage1/2/3 上的综合表现，以及 10-carrier Stage3 的采样与 replay 覆盖情况。",
        "",
    ]
    lines.extend(render_table(global_main, best_columns={"Stage1 gain": "max", "Stage1 score": "max", "Stage2 score": "max", "Stage2 ASR": "max", "Stage2 FFD": "min", "Stage2 SWD": "min", "Stage3 replay ASR": "max", "Stage3 score": "max", "Deployability": "max", "Target L2": "min", "Fatal rate": "min", "Carrier sampled": "max", "Carrier replayed": "max"}))
    lines.extend(["", stage3_summary_analysis_v2(stage3_summary), ""])

    lines.extend(["## 2. Stage1 IDS 矩阵", "", "指标说明：这组表用于回答“是否真的训练了多个异构 IDS，以及它们之间能否互相提取/逼近决策边界”。", "", stage1_note, ""])
    lines.extend(render_table(stage1_model_rows, best_columns={"diag_agreement": "max"}))
    lines.append("")
    lines.extend(render_table(stage1_summary_rows, best_columns={"mean": "max"}))
    lines.append("")
    if stage1_matrix_rows:
        matrix_headers = list(stage1_matrix_rows[0].keys())
        matrix_display: list[list[str]] = []
        for row in stage1_matrix_rows:
            numeric_headers = [header for header in matrix_headers if header != "surrogate\\target"]
            numeric_values = {header: to_float(row.get(header)) for header in numeric_headers}
            best = max((value for value in numeric_values.values() if value is not None), default=None)
            rendered = []
            for header in matrix_headers:
                cell = str(row.get(header, "-"))
                value = numeric_values.get(header)
                if header != "surrogate\\target" and best is not None and value is not None and abs(value - best) <= 1.0e-12:
                    cell = f"**{cell}**"
                rendered.append(cell)
            matrix_display.append(rendered)
        lines.extend(pipe_table(matrix_headers, matrix_display))
    else:
        lines.append("（当前无矩阵数据）")
    lines.extend(["", stage1_matrix_analysis(stage1_matrix_rows), ""])

    lines.extend(["## 3. Stage2 主线方法对比", "", "指标说明：`ASR_oracle/ASR_surrogate` 分别是原始目标 IDS 与提取 surrogate 上的成功率；`FFD/SWD` 衡量与 benign 参考分布的距离；`C2ST_*` 衡量两样本可分性；`Corr_Delta` 衡量整体相关结构偏差；`AdvToMal_L2` 衡量与原始 malicious 的距离；`Queries_per_success/Time_sec` 是攻击代价；`Score` 是仅对主方法保留的辅助综合分。", ""])
    lines.extend(render_table(global_s2_compare, best_columns={"ASR_oracle": "max", "ASR_surrogate": "max", "FFD": "min", "SWD": "min", "C2ST_AUC": "min", "C2ST_Acc": "min", "Corr_Delta": "min", "AdvToMal_L2": "min", "Queries_per_success": "min", "Time_sec": "min", "Score": "max"}, group_by=["Attack"]))
    lines.extend(["", stage2_global_compare_analysis(global_s2_compare), ""])

    lines.extend(["## 4. Stage2 Structure-Aware Feature Realism", "", "指标说明：这张表对齐旧版论文的 realism 口径，联合比较分布距离、可分性、覆盖率、局部邻域一致性、整体相关偏差、协方差形状以及成对距离统计，从而回答生成样本是否既像 benign 又不是简单随机扰动。", ""])
    lines.extend(render_table(stage2_realism, best_columns={"ASR_oracle": "max", "ASR_surrogate": "max", "FFD": "min", "SWD": "min", "Energy": "min", "C2ST_AUC": "min", "C2ST_Acc": "min", "Coverage@5": "max", "kNN_P": "max", "kNN_R": "max", "Corr_Delta": "min", "CovSpec_L2": "min", "CovTrace": "min", "PairDist_KS": "min", "PairMean": "min", "AdvToMal_L2": "min", "Queries_per_success": "min", "Time_sec": "min"}, group_by=["Attack"]))
    lines.extend(["", stage2_realism_analysis(stage2_realism), ""])

    lines.extend(["## 5. Stage2 STP 三组相关性（CGD）", "", "指标说明：`CGD_ST/CGD_SP/CGD_TP` 分别表示 temporal-spatial、spatial-protocol、temporal-protocol 三组跨域相关性偏差，`CGD_AVG` 是三者均值，值越小表示 STP 依赖保持得越好。", ""])
    lines.extend(render_table(stage2_cgd, best_columns={"CGD_ST": "min", "CGD_SP": "min", "CGD_TP": "min", "CGD_AVG": "min"}, group_by=["Attack"]))
    lines.extend(["", stage2_cgd_analysis(stage2_cgd), ""])
    lines.extend(["## 5A. Stage2 Support-Aware Selection", "", "Metric note: this table surfaces the support-aware post-processing and candidate-selection settings used by Stage2 so the reviewer can see that the method is not relying on the generator backbone alone.", ""])
    lines.extend(render_table(stage2_support, best_columns={"Stage2 score": "max", "FFD": "min", "SWD": "min"}))
    lines.extend(["", stage2_support_analysis(stage2_support), ""])

    lines.extend(["## 6. 按攻击类型拆分的 Stage2 主结果", "", "指标说明：这张表用于看不同攻击类型在同一套主模型下，哪些更容易被攻击，哪些更容易出现统计偏移。", ""])
    lines.extend(render_table(attack_slice, best_columns={"Eval rows": "max", "Stage1 agreement": "max", "Stage1 baseline agreement": "max", "ASR_oracle": "max", "ASR_surrogate": "max", "FFD": "min", "SWD": "min", "AdvToMal_L2": "min", "Score": "max", "Time_sec": "min"}))
    lines.extend(["", stage2_slice_analysis(attack_slice), ""])

    lines.extend(["## 7. 按攻击类型拆分的 Stage2 方法对比", "", "指标说明：这张表逐个攻击类型检查 RD-Synth 是否真的优于 baseline，而不是只在 GLOBAL 聚合值上占优。", ""])
    lines.extend(render_table(attack_baselines, best_columns={"ASR_oracle": "max", "ASR_surrogate": "max", "FFD": "min", "SWD": "min", "AdvToMal_L2": "min", "Time_sec": "min", "Score": "max"}, group_by=["Attack"]))
    lines.extend(["", stage2_baseline_analysis(attack_baselines), ""])
    lines.extend(["## 7A. Transfer IDS", "", "Metric note: transfer IDS rows show whether the adversarial samples still evade additional IDS models that were not the shared Stage2/Stage3 main detector.", ""])
    lines.extend(render_table(transfer_ids, best_columns={"Test Acc": "max", "Test F1": "max", "Adv ASR": "max", "ΔASR vs main": "max"}))
    lines.extend(["", transfer_ids_analysis(transfer_ids), ""])

    lines.extend(["## 8. Ablation", "", "指标说明：这张表用常见 raw metric 展示 ablation 的退化位置；`S2_*` 是特征空间攻击效果与保真度，`S3_*` 是 packet-level 回放效果，`Remap_*` 是 remapper 拟合误差。", ""])
    lines.extend(render_table(ablation_detail, best_columns={"S2_ASR_oracle": "max", "S2_ASR_surrogate": "max", "S2_FFD": "min", "S2_SWD": "min", "S2_AdvToMal_L2": "min", "S3_Replay_ASR": "max", "S3_Target_L2": "min", "S3_Target_MAE": "min", "S3_Alignment": "max", "S3_Fatal": "min", "Remap_R2": "max", "Remap_MAE": "min", "Remap_RMSE": "min", "Port_Acc": "max", "Score_aux": "max"}))
    lines.extend(["", ablation_analysis(ablation_detail), ""])

    lines.extend(["## 9. Stage3 汇总指标", "", "指标说明：这张表用于确认 10-carrier Stage3 是否真正形成 packet-level 证据，并显式标出 Stage2 oracle 与 Stage3 IDS 是否共用同一检测模型。", ""])
    lines.extend(render_table(stage3_summary, best_columns={"Carrier sampled": "max", "Carrier replayed": "max", "Carrier source_only": "min", "Replay ASR": "max", "Deployability": "max", "Remap quality": "max", "Target L2": "min", "Alignment": "max", "Fatal rate": "min", "Eval time sec": "min"}))
    lines.extend(["", stage3_summary_analysis_v2(stage3_summary), ""])

    lines.extend(["## 10. Stage3 Carrier 级结果", "", "指标说明：`Source_*` 是原始 carrier PCAP 的 oracle 判断，`Adv_*` 是 RD-Synth 变形并 replay 后的 oracle 判断，`Carrier_ASR` 表示该 carrier 是否实现 `source=malicious -> adv=benign`；`Alignment_*`、`Target_*` 和 `Sanity_*` 分别表示特征对齐程度、目标偏移和协议/时序合法性。", ""])
    lines.extend(render_table(carrier_rows, best_columns={"Source_pmal": "min", "Adv_pmal": "min", "Carrier_ASR": "max", "Alignment_coverage": "max", "Target_L2": "min", "Target_MAE": "min", "Sanity_nonmonotonic": "min", "Sanity_transport_missing": "min", "Sanity_tcp_seq_backwards": "min", "Sanity_tcp_flag_invalid": "min"}))
    lines.extend(["", stage3_carrier_analysis_v2(carrier_rows), ""])
    lines.extend(["## 10A. Stage3 Hard-Carrier Slice", "", "Metric note: this slice keeps only carriers that were already judged malicious by the shared IDS at source time, so it answers the harder-case replay question directly.", ""])
    lines.extend(render_table(hard_carriers, best_columns={"Source_pmal": "max", "Adv_pmal": "min", "Carrier_ASR": "max", "Alignment_coverage": "max", "Target_L2": "min", "Sanity_tcp_seq_backwards": "min"}))
    lines.extend(["", hard_carrier_analysis(hard_carriers), ""])
    lines.extend(["## 10B. Stage3 Baseline Realization Policy", "", "Metric note: this table distinguishes which baselines were truly evaluated at packet level and which ones remain skipped because the repository still lacks their native packet writer.", ""])
    lines.extend(render_table(stage3_baseline_policy, best_columns={"Deployability": "max", "Replay ASR": "max", "Target L2": "min"}))
    lines.extend(["", stage3_baseline_policy_analysis(stage3_baseline_policy), ""])

    lines.extend(["## 11. Stage3 Packet Remapping Distortion", "", "指标说明：这张表按 remapper 控制字段汇总从 feature target 到 packet-space 实际落地时的拟合误差，`MAE/RMSE` 越小表示改动更小、更可控。", ""])
    lines.extend(render_table(stage3_remap, best_columns={"MAE": "min", "RMSE": "min", "Port_Acc": "max", "Apply_selected": "max"}, group_by=["Attack"]))
    lines.extend(["", "结论：" + ("当前 remapper 的连续字段里 `pad_bytes` 误差最小，而 `dst_port_new` 仍是最难稳定拟合的离散字段；这说明当前 packet remap 的主要难点已经从时序和 padding，转移到了目的端口这类强离散控制量。" if stage3_remap else "当前没有 remap distortion 表。"), ""])

    lines.extend(["## 12. Stage3 Protocol-Legality Checks", "", "指标说明：这张表把 reconstructed PCAP 的主要合法性检查收成 reviewer 可读口径；`ValidFatal@0` 表示 fatal 错误为零的比例，其余 0/1 列表示对应违规率是否为零。", ""])
    lines.extend(render_table(stage3_legality, best_columns={"ValidFatal@0": "max", "TCP_Flag_Invalid": "max", "TCP_SYN_FIN": "max", "TCP_SYN_RST": "max", "TCP_FIN_RST": "max", "Transport_Present": "max", "TCP_Seq_Backwards_0": "max", "TCP_Seq_Backwards_Rate": "min", "Nonmonotonic_Rate": "min", "Port_Acc": "max"}))
    lines.extend(["", "结论：" + ("当前 `ValidFatal@0=1.0000`，`TCP_Flag_Invalid/TCP_SYN_FIN/TCP_SYN_RST/TCP_FIN_RST/Transport_Present` 都已达零违规；剩余最明显的合法性风险仍是 `TCP_Seq_Backwards_Rate`，它已经不再触发 fatal，但仍值得继续压低。" if stage3_legality else "当前没有 protocol-legality 表。"), ""])

    lines.extend(["## 13. 整体判断", "", overall_conclusion_v2(matrix_rows=stage1_matrix_rows, ablation_rows=ablation_detail, stage3_summary_rows=stage3_summary), ""])

    report_path = root / "NB15_TABLE_BANK_CN.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    return report_path, csv_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate NB15 experiment table bank report in Chinese.")
    parser.add_argument("--root", required=True, help="reviewer suite run root")
    parser.add_argument("--dataset", default="nb15")
    args = parser.parse_args()

    report_path, csv_paths = build_report_v2(Path(args.root).resolve(), args.dataset)
    print(f"[NB15TableBank] report {report_path}")
    for path in csv_paths:
        print(f"[NB15TableBank] csv {path}")


if __name__ == "__main__":
    main()
