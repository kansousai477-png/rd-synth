from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

from rdsynth.baselines.paper_attacks import (
    generate_paper_attack_baseline,
    get_paper_baseline_spec,
    stage3_policy_for_baseline,
)
from rdsynth.utils.metrics_stage2 import compute_stage2_metrics, nearest_reference_distance, paired_sample_l2
from rdsynth.utils.paper_metrics import add_paper_attack_metrics
from rdsynth.utils.query_oracle import QueryOracle


def _baseline_artifact_meta(
    name: str, *, include_stage3_policy: bool = False
) -> tuple[str, bool] | tuple[str, bool, str]:
    baseline_name = str(name).lower()
    spec = get_paper_baseline_spec(baseline_name)
    if spec is not None:
        policy = stage3_policy_for_baseline(baseline_name)
        return (
            (spec.family, bool(spec.traffic_space), policy)
            if include_stage3_policy
            else (
                spec.family,
                bool(spec.traffic_space),
            )
        )
    family_map = {
        "identity": ("control_identity", False),
        "global_random": ("control_random", False),
        "knn_benign": ("control_neighbor", False),
        "benign_neighbor_random": ("control_neighbor_random", False),
        "fgsm": ("gradient_attack", False),
        "pgd": ("gradient_attack", False),
    }
    family, traffic_space = family_map.get(baseline_name, ("baseline_other", False))
    if include_stage3_policy:
        return family, traffic_space, "feature_only_random_remap"
    return family, traffic_space


def fgsm_attack(
    surrogate: torch.nn.Module,
    x: np.ndarray,
    target_label: int,
    eps: float,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    surrogate.eval()
    ce = torch.nn.CrossEntropyLoss()
    adv = []
    for start in range(0, x.shape[0], batch_size):
        xb = torch.tensor(x[start : start + batch_size], dtype=torch.float32, device=device, requires_grad=True)
        target = torch.full((xb.size(0),), target_label, dtype=torch.long, device=device)
        logits = surrogate(xb)
        loss = ce(logits, target)
        surrogate.zero_grad()
        if xb.grad is not None:
            xb.grad.zero_()
        loss.backward()
        xb_adv = xb - eps * xb.grad.sign()
        adv.append(xb_adv.detach().cpu().numpy())
    return np.vstack(adv)


def pgd_attack(
    surrogate: torch.nn.Module,
    x: np.ndarray,
    target_label: int,
    eps: float,
    alpha: float,
    steps: int,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> np.ndarray:
    surrogate.eval()
    ce = torch.nn.CrossEntropyLoss()
    adv = []
    rng = np.random.default_rng(seed)
    for start in range(0, x.shape[0], batch_size):
        x_batch = x[start : start + batch_size]
        delta = rng.uniform(-eps, eps, size=x_batch.shape).astype(np.float32)
        xb = torch.tensor(x_batch, dtype=torch.float32, device=device)
        x_adv = xb + torch.tensor(delta, dtype=torch.float32, device=device)
        for _ in range(steps):
            x_adv = x_adv.clone().detach().requires_grad_(True)
            target = torch.full((x_adv.size(0),), target_label, dtype=torch.long, device=device)
            logits = surrogate(x_adv)
            loss = ce(logits, target)
            surrogate.zero_grad()
            if x_adv.grad is not None:
                x_adv.grad.zero_()
            loss.backward()
            x_adv = x_adv - alpha * x_adv.grad.sign()
            delta = torch.clamp(x_adv - xb, min=-eps, max=eps)
            x_adv = xb + delta
        adv.append(x_adv.detach().cpu().numpy())
    return np.vstack(adv)


def random_baseline(x_pool: np.ndarray, n: int, rng_local: np.random.Generator) -> np.ndarray:
    idx = rng_local.choice(len(x_pool), n, replace=True)
    return x_pool[idx]


def neighbor_random_baseline(
    x_query_norm: np.ndarray,
    x_pool_pre: np.ndarray,
    x_pool_norm: np.ndarray,
    n_neighbors: int,
    rng_local: np.random.Generator,
) -> np.ndarray:
    nn = NearestNeighbors(
        n_neighbors=max(1, min(n_neighbors, x_pool_norm.shape[0])),
        metric="euclidean",
    ).fit(x_pool_norm)
    _, idx = nn.kneighbors(x_query_norm, return_distance=True)
    if idx.shape[1] == 1:
        chosen = idx[:, 0]
    else:
        sampled_col = rng_local.integers(0, idx.shape[1], size=idx.shape[0])
        chosen = idx[np.arange(idx.shape[0]), sampled_col]
    return x_pool_pre[chosen]


def stabilize_preprocessed(
    x: np.ndarray,
    x_ben_pre: np.ndarray,
    enabled: bool,
    quantile: float,
) -> np.ndarray:
    arr = np.nan_to_num(np.asarray(x, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if not enabled or x_ben_pre.shape[0] == 0:
        return arr
    q = min(max(quantile, 0.0), 0.2)
    if q <= 0.0:
        lo = np.min(x_ben_pre, axis=0)
        hi = np.max(x_ben_pre, axis=0)
    else:
        lo = np.quantile(x_ben_pre, q, axis=0)
        hi = np.quantile(x_ben_pre, 1.0 - q, axis=0)
    return np.clip(arr, lo, hi)


def run_stage2_baselines(
    *,
    cfg: dict[str, Any],
    bundle_feature_names: list[str],
    out_dir: Path,
    device: torch.device,
    seed: int,
    surrogate: torch.nn.Module,
    oracle: Any,
    y_train: np.ndarray,
    x_train: np.ndarray,
    x_ben_pre: np.ndarray,
    x_mal_pre: np.ndarray,
    x_ben_norm: np.ndarray,
    x_mal_norm: np.ndarray,
    denorm_mean: np.ndarray,
    denorm_std: np.ndarray,
    norm_bounds_min: np.ndarray | None,
    norm_bounds_max: np.ndarray | None,
    norm_nonneg: np.ndarray,
    metrics_payload: dict[str, Any],
    sample_denorm: bool,
    mal_benign_rate: float | None,
    mal_benign_rate_oracle: float | None,
    attack_score_fn: Callable[[np.ndarray], np.ndarray],
    surrogate_predict_probs: Callable[[np.ndarray, int], tuple[np.ndarray, np.ndarray]],
    oracle_predict_probs: Callable[[np.ndarray, int], tuple[np.ndarray, np.ndarray | None]],
    postprocess_adv: Callable[[np.ndarray, np.ndarray, bool, bool], tuple[np.ndarray, np.ndarray]],
) -> None:
    baseline_cfg = cfg["stage2"].get("baselines", {})
    baseline_enable = bool(baseline_cfg.get("enable", True))
    if not baseline_enable:
        return

    methods = baseline_cfg.get("methods", ["global_random", "benign_neighbor_random", "fgsm", "pgd"])
    apply_constraints_baseline = bool(baseline_cfg.get("apply_constraints", True))
    fgsm_eps = float(baseline_cfg.get("fgsm_eps", 0.05))
    pgd_eps = float(baseline_cfg.get("pgd_eps", 0.1))
    pgd_alpha = float(baseline_cfg.get("pgd_alpha", 0.02))
    pgd_steps = int(baseline_cfg.get("pgd_steps", 10))
    baseline_budget_scale = float(max(0.1, baseline_cfg.get("budget_scale", 1.0)))
    paper_budget_scale = float(max(0.1, baseline_cfg.get("paper_budget_scale", baseline_budget_scale)))
    hard_label_queries = bool(baseline_cfg.get("hard_label_queries", False))
    query_budget_raw = baseline_cfg.get("query_budget")
    query_budget = None if query_budget_raw in (None, "", "null") else int(query_budget_raw)
    paper_query_budget_raw = baseline_cfg.get("paper_query_budget", query_budget)
    paper_query_budget = None if paper_query_budget_raw in (None, "", "null") else int(paper_query_budget_raw)
    hard_label_threshold = float(baseline_cfg.get("hard_label_threshold", 0.5))
    exhausted_fill = float(baseline_cfg.get("exhausted_fill", 1.0))
    pgd_steps_eff = max(1, int(round(float(pgd_steps) * baseline_budget_scale)))
    knn_k = int(baseline_cfg.get("knn_k", 1))
    neighbor_random_k = int(baseline_cfg.get("neighbor_random_k", max(3, knn_k)))
    baseline_eval_metrics = bool(baseline_cfg.get("eval_metrics", True))
    baseline_stabilize_inputs = bool(baseline_cfg.get("stabilize_inputs", True))
    baseline_stabilize_quantile = float(baseline_cfg.get("stabilize_quantile", 0.01))
    save_baseline_samples = bool((cfg.get("stage3") or {}).get("pcap_compare_baselines", False))
    baseline_rows = []
    baseline_samples: dict[str, np.ndarray] = {}
    knn_model = None

    x_ben_pre_baseline = stabilize_preprocessed(
        x_ben_pre, x_ben_pre, baseline_stabilize_inputs, baseline_stabilize_quantile
    )
    x_mal_pre_baseline = stabilize_preprocessed(
        x_mal_pre, x_ben_pre, baseline_stabilize_inputs, baseline_stabilize_quantile
    )
    x_train_pre_baseline = stabilize_preprocessed(
        x_train, x_ben_pre, baseline_stabilize_inputs, baseline_stabilize_quantile
    )
    x_ben_norm_baseline = (x_ben_pre_baseline - denorm_mean) / (denorm_std + 1.0e-8)
    x_mal_norm_baseline = (x_mal_pre_baseline - denorm_mean) / (denorm_std + 1.0e-8)
    metrics_payload["baseline_budget_scale"] = baseline_budget_scale
    metrics_payload["baseline_paper_budget_scale"] = paper_budget_scale
    metrics_payload["baseline_pgd_steps_effective"] = pgd_steps_eff
    metrics_payload["baseline_hard_label_queries"] = hard_label_queries
    if query_budget is not None:
        metrics_payload["baseline_query_budget"] = int(query_budget)
    if paper_query_budget is not None:
        metrics_payload["baseline_paper_query_budget"] = int(paper_query_budget)

    def _baseline_attack_score_fn(x_pre: np.ndarray) -> np.ndarray:
        x_safe = stabilize_preprocessed(x_pre, x_ben_pre, baseline_stabilize_inputs, baseline_stabilize_quantile)
        return attack_score_fn(x_safe)

    if any(str(m).lower() in {"knn_benign", "knn", "benign_neighbor_random", "neighbor_random"} for m in methods):
        if x_ben_norm_baseline.shape[0] > 0:
            knn_model = NearestNeighbors(
                n_neighbors=max(1, min(max(knn_k, neighbor_random_k), x_ben_norm_baseline.shape[0])),
                metric="euclidean",
            ).fit(x_ben_norm_baseline)

    def _baseline_metrics(
        tag: str,
        x_adv_pre_local: np.ndarray,
        runtime_sec: float,
        *,
        attack_runtime_sec: float,
        query_oracle: QueryOracle | None,
    ) -> None:
        ffd = float("nan")
        swd = float("nan")
        energy = float("nan")
        c2st_auc = float("nan")
        c2st_acc = float("nan")
        x_adv_pre_local = stabilize_preprocessed(
            x_adv_pre_local, x_ben_pre, baseline_stabilize_inputs, baseline_stabilize_quantile
        )
        if baseline_eval_metrics:
            metrics_local = compute_stage2_metrics(
                x_ben_norm_baseline,
                (x_adv_pre_local - denorm_mean) / (denorm_std + 1.0e-8),
                feature_names=bundle_feature_names,
                max_real=cfg["stage2"].get("metrics_max_real", 2000),
                max_gen=cfg["stage2"].get("metrics_max_gen", 2000),
                seed=seed,
                bounds_min=norm_bounds_min,
                bounds_max=norm_bounds_max,
                nonneg_mask=norm_nonneg,
            )
            metrics_dict = metrics_local.as_dict()
            ffd = metrics_dict.get("FFD", ffd)
            swd = metrics_dict.get("SWD", swd)
            energy = metrics_dict.get("Energy", energy)
            c2st_auc = metrics_dict.get("C2ST-AUC", c2st_auc)
            c2st_acc = metrics_dict.get("C2ST-Acc", c2st_acc)
        x_adv_norm_local = (x_adv_pre_local - denorm_mean) / (denorm_std + 1.0e-8)
        adv_ben_l2_local = nearest_reference_distance(x_adv_norm_local, x_ben_norm_baseline)
        adv_mal_l2_local = paired_sample_l2(x_adv_pre_local, x_mal_pre_baseline)
        s_preds, s_probs = surrogate_predict_probs(x_adv_pre_local, cfg["stage2"].get("sample_batch_size", 512))
        asr_sur = float(np.mean(s_preds == 0))
        adv_pmal_sur = float(np.mean(s_probs[:, 1]))
        asr_orc = float("nan")
        adv_pmal_orc = float("nan")
        if oracle is not None:
            o_preds, o_probs = oracle_predict_probs(x_adv_pre_local, cfg["stage2"].get("sample_batch_size", 512))
            if o_preds.size:
                asr_orc = float(np.mean(o_preds == 0))
            if o_probs is not None and o_probs.size:
                adv_pmal_orc = float(np.mean(o_probs[:, 1]))
        metrics_payload.update(
            {
                f"baseline_{tag}_asr_surrogate": asr_sur,
                f"baseline_{tag}_adv_pmal_surrogate": adv_pmal_sur,
                f"baseline_{tag}_asr_oracle": asr_orc,
                f"baseline_{tag}_adv_pmal_oracle": adv_pmal_orc,
                f"baseline_{tag}_norm_FFD": ffd,
                f"baseline_{tag}_norm_SWD": swd,
                f"baseline_{tag}_norm_Energy": energy,
                f"baseline_{tag}_norm_C2ST-AUC": c2st_auc,
                f"baseline_{tag}_norm_C2ST-Acc": c2st_acc,
                f"baseline_{tag}_norm_AdvToBen_L2": adv_ben_l2_local,
                f"baseline_{tag}_norm_AdvToMal_L2": adv_mal_l2_local,
                f"baseline_{tag}_time_cost_sec": runtime_sec,
                f"baseline_{tag}_end_to_end_time_sec": runtime_sec,
                f"baseline_{tag}_attack_time_cost_sec": attack_runtime_sec,
                f"baseline_{tag}_sample_count": int(x_adv_pre_local.shape[0]),
                f"baseline_{tag}_samples_per_sec": float(x_adv_pre_local.shape[0] / runtime_sec)
                if runtime_sec > 0.0
                else float("nan"),
                f"baseline_{tag}_end_to_end_samples_per_sec": float(x_adv_pre_local.shape[0] / runtime_sec)
                if runtime_sec > 0.0
                else float("nan"),
            }
        )
        if query_oracle is not None:
            qstats = query_oracle.stats()
            metrics_payload[f"baseline_{tag}_query_count"] = int(qstats.query_count)
            metrics_payload[f"baseline_{tag}_query_calls"] = int(qstats.query_calls)
            metrics_payload[f"baseline_{tag}_query_time_sec"] = float(qstats.query_time_sec)
            metrics_payload[f"baseline_{tag}_query_over_budget_count"] = int(qstats.query_over_budget_count)
            metrics_payload[f"baseline_{tag}_query_budget_exhausted"] = bool(qstats.budget_exhausted)
            if qstats.query_budget is not None:
                metrics_payload[f"baseline_{tag}_query_budget"] = int(qstats.query_budget)
            metrics_payload[f"baseline_{tag}_hard_label_queries"] = bool(qstats.hard_label)
            metrics_payload[f"baseline_{tag}_non_query_time_sec"] = float(max(0.0, runtime_sec - qstats.query_time_sec))
        add_paper_attack_metrics(
            metrics_payload,
            prefix=f"baseline_{tag}_surrogate_",
            asr=asr_sur,
            orig_benign_rate=mal_benign_rate,
            adv_prob_malicious=adv_pmal_sur,
            ffd=ffd,
            swd=swd,
            c2st_auc=c2st_auc,
            c2st_acc=c2st_acc,
            adv_to_ben_l2=adv_ben_l2_local,
            adv_to_mal_l2=adv_mal_l2_local,
            runtime_sec=runtime_sec,
        )
        add_paper_attack_metrics(
            metrics_payload,
            prefix=f"baseline_{tag}_oracle_",
            asr=asr_orc,
            orig_benign_rate=mal_benign_rate_oracle,
            adv_prob_malicious=adv_pmal_orc,
            ffd=ffd,
            swd=swd,
            c2st_auc=c2st_auc,
            c2st_acc=c2st_acc,
            adv_to_ben_l2=adv_ben_l2_local,
            adv_to_mal_l2=adv_mal_l2_local,
            runtime_sec=runtime_sec,
        )
        success_sur = asr_sur * max(1, x_adv_pre_local.shape[0])
        success_orc = asr_orc * max(1, x_adv_pre_local.shape[0]) if np.isfinite(asr_orc) else float("nan")
        qcnt = float(metrics_payload.get(f"baseline_{tag}_query_count", 0.0))
        metrics_payload[f"baseline_{tag}_queries_per_success_surrogate"] = (
            qcnt / success_sur if success_sur > 1.0e-12 else float("nan")
        )
        metrics_payload[f"baseline_{tag}_queries_per_success_oracle"] = (
            qcnt / success_orc
            if isinstance(success_orc, float) and np.isfinite(success_orc) and success_orc > 1.0e-12
            else float("nan")
        )
        if save_baseline_samples:
            baseline_samples[tag] = np.asarray(x_adv_pre_local, dtype=np.float32)
        baseline_rows.append((tag, asr_sur, asr_orc, ffd, swd))

    rng = np.random.default_rng(seed)
    for method in methods:
        name = str(method).lower()
        baseline_start = time.perf_counter()
        query_oracle = QueryOracle(
            _baseline_attack_score_fn,
            max_queries=paper_query_budget if get_paper_baseline_spec(name) is not None else query_budget,
            hard_label=hard_label_queries,
            hard_label_threshold=hard_label_threshold,
            exhausted_fill=exhausted_fill,
        )
        if name in {"random", "global_random"}:
            x_base = random_baseline(x_ben_pre_baseline, len(x_mal_pre_baseline), rng)
            name = "global_random"
        elif name in {"benign_neighbor_random", "neighbor_random"}:
            x_base = neighbor_random_baseline(
                x_mal_norm_baseline,
                x_ben_pre_baseline,
                x_ben_norm_baseline,
                neighbor_random_k,
                rng,
            )
            name = "benign_neighbor_random"
        elif name in {"identity", "none", "noop"}:
            x_base = x_mal_pre_baseline
        elif name in {"knn_benign", "knn"}:
            if knn_model is None:
                continue
            _, idx = knn_model.kneighbors(x_mal_norm_baseline, return_distance=True)
            x_base = x_ben_pre_baseline[idx[:, 0]]
        elif name == "fgsm":
            x_base = fgsm_attack(
                surrogate,
                x_mal_pre_baseline,
                target_label=0,
                eps=fgsm_eps,
                batch_size=cfg["stage2"].get("sample_batch_size", 512),
                device=device,
            )
        elif name == "pgd":
            x_base = pgd_attack(
                surrogate,
                x_mal_pre_baseline,
                target_label=0,
                eps=pgd_eps,
                alpha=pgd_alpha,
                steps=pgd_steps_eff,
                batch_size=cfg["stage2"].get("sample_batch_size", 512),
                device=device,
                seed=seed,
            )
        elif get_paper_baseline_spec(name) is not None:
            x_base = generate_paper_attack_baseline(
                name=name,
                x_mal_pre=x_mal_pre_baseline,
                x_ben_pre=x_ben_pre_baseline,
                feature_names=bundle_feature_names,
                score_fn=query_oracle,
                surrogate_model=surrogate,
                x_train_pre=x_train_pre_baseline,
                y_train=y_train,
                device=device,
                seed=seed,
                budget_scale=paper_budget_scale,
            )
        else:
            continue
        attack_runtime_sec = time.perf_counter() - baseline_start

        if sample_denorm:
            x_base, _ = postprocess_adv(
                x_base,
                x_mal_pre_baseline,
                apply_constraints_baseline,
                apply_constraints_baseline,
            )
        else:
            x_base, _ = postprocess_adv(
                (x_base - denorm_mean) / (denorm_std + 1.0e-8),
                x_mal_pre_baseline,
                apply_constraints_baseline,
                apply_constraints_baseline,
            )

        _baseline_metrics(
            name,
            x_base,
            time.perf_counter() - baseline_start,
            attack_runtime_sec=attack_runtime_sec,
            query_oracle=query_oracle,
        )

    if baseline_rows:
        print("\n[Stage2] baselines (asr_surrogate/asr_oracle/FFD/SWD):")
        for row in baseline_rows:
            print(f"  {row[0]:<8} {row[1]:.6f} {row[2]:.6f} {row[3]:.6f} {row[4]:.6f}")
    if baseline_samples and save_baseline_samples:
        for baseline_name, baseline_adv in baseline_samples.items():
            baseline_family, baseline_traffic_space, baseline_stage3_policy = _baseline_artifact_meta(
                baseline_name,
                include_stage3_policy=True,
            )
            np.savez_compressed(
                out_dir / f"baseline_{baseline_name}_samples.npz",
                artifact_version=np.asarray(1),
                adv=baseline_adv,
                adv_pre=baseline_adv,
                benign_pre=x_ben_pre_baseline,
                mal_pre=x_mal_pre_baseline,
                benign=x_ben_norm_baseline,
                mal=x_mal_norm_baseline,
                ben_stats_mean=denorm_mean,
                ben_stats_std=denorm_std,
                adv_space=np.asarray("preprocessed"),
                feature_names=np.asarray(bundle_feature_names),
                baseline_name=np.asarray(str(baseline_name)),
                baseline_family=np.asarray(str(baseline_family)),
                traffic_space=np.asarray(int(baseline_traffic_space)),
                stage3_policy=np.asarray(str(baseline_stage3_policy)),
                baseline_description=np.asarray(""),
            )
