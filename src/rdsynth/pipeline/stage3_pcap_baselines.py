from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from rdsynth.baselines.paper_attacks import get_paper_baseline_spec
from rdsynth.pipeline.stage3_ops import Stage3Settings, aligned_feature_diff, pcap_output_dir, write_modified_pcaps
from rdsynth.pipeline.stage3_pcap import PcapFeatureExtractionError
from rdsynth.stages.stage3_remap import (
    build_random_remap_modifications,
    build_rule_based_modifications,
    clip_modifications,
    predict_modifications,
)
from rdsynth.utils.artifacts import save_records_csv
from rdsynth.utils.paper_metrics import add_paper_pcap_metrics


def run_stage3_baseline_pcap_eval(
    *,
    cfg: Mapping[str, Any],
    settings: Stage3Settings,
    metrics_payload: dict[str, Any],
    main_adv_pre: np.ndarray | None = None,
    pcap_evasion_valid: bool | None,
    preprocessor: Any,
    remap_mode: str,
    remap_use_direct: bool,
    remap_bundle: Any,
    x_ben_raw: np.ndarray,
    feature_names: list[str],
    scapy_available: bool,
    protocol_auto_fix: bool,
    pcap_path: Path | None,
    pcap_features: Any,
    out_dir: Path,
    seed: int,
    device: Any,
    x_train: np.ndarray,
    effective_blend_fn,
    blend_fn,
) -> None:
    if not settings.pcap_compare_baselines or not pcap_evasion_valid or pcap_path is None:
        return
    if not scapy_available:
        return

    try:
        from scapy.all import rdpcap, wrpcap
    except Exception as exc:
        print(f"[Stage3/BL][Warn] scapy import failed, skipping baseline PCAP eval: {exc}")
        return

    def _classify_with_cache(pcap_file: Path) -> tuple[np.ndarray, dict[str, Any], np.ndarray]:
        classify_pcap = getattr(pcap_features, "classify_pcap", None)
        classified = classify_pcap(pcap_file) if callable(classify_pcap) else None
        if isinstance(classified, tuple) and len(classified) == 5:
            feat, _, meta, probs, feat_pre = classified
        else:
            feat, _, meta = pcap_features.extract(str(pcap_file))
            probs, feat_pre = pcap_features.classify_features(feat)
        return probs, meta if isinstance(meta, dict) else {}, feat_pre

    stage2_dir = Path(cfg["project"]["out_dir"]) / "stage2"
    baseline_npz_files = sorted(stage2_dir.glob("baseline_*_samples.npz"))
    baseline_eval_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    baseline_root_dir = pcap_output_dir(settings, out_dir)
    pkts = rdpcap(str(pcap_path))

    def _evaluate_direct_rule_only() -> dict[str, Any] | None:
        if main_adv_pre is None or not len(np.asarray(main_adv_pre)):
            return None
        direct_main_raw = preprocessor.inverse_transform(np.asarray(main_adv_pre, dtype=np.float32))
        direct_main_mods = build_rule_based_modifications(
            x_adv_raw=direct_main_raw,
            x_ben_raw=x_ben_raw,
            feature_names=feature_names,
        )
        direct_main_mods = clip_modifications(direct_main_mods)
        direct_main_dir = baseline_root_dir / "direct_rule_only"
        written, baseline_packets_total, baseline_apply_time_sec, adv_pcaps = write_modified_pcaps(
            pkts,
            direct_main_mods,
            direct_main_dir,
            seed=seed,
            count=settings.pcap_apply_n,
            settings=settings,
            protocol_auto_fix=protocol_auto_fix,
            wrpcap_fn=wrpcap,
        )
        rows_local: list[dict[str, Any]] = []
        pmal_vals: list[float] = []
        pred_vals: list[int] = []
        target_l2_vals_local: list[float] = []
        target_mae_vals_local: list[float] = []
        failed_eval_count = 0
        for p in adv_pcaps:
            try:
                probs, meta, feat_pre = _classify_with_cache(p)
            except PcapFeatureExtractionError as exc:
                failed_eval_count += 1
                skipped_rows.append(
                    {
                        "baseline": "direct_rule_only",
                        "baseline_group": "shared_backend",
                        "evaluation_mode": "shared_backend",
                        "status": "failed",
                        "reason": str(exc),
                    }
                )
                continue
            pmal_vals.append(float(probs[1]))
            pred_vals.append(int(probs[1] > probs[0]))
            target_idx = None
            try:
                stem = Path(p).stem
                if stem.startswith("adv_"):
                    target_idx = int(stem.split("_", 1)[1])
            except Exception as exc:
                print(f"[Stage3/BL][Warn] baseline PCAP index parse failed for {p}: {exc}")
                target_idx = None
            if target_idx is not None and main_adv_pre is not None and target_idx < len(main_adv_pre):
                feat_diff = feat_pre - main_adv_pre[target_idx]
                align_meta = meta.get("alignment") if isinstance(meta, dict) else None
                feat_diff = aligned_feature_diff(feat_diff, align_meta, feature_names)
                target_l2_vals_local.append(float(np.linalg.norm(feat_diff)))
                target_mae_vals_local.append(float(np.mean(np.abs(feat_diff))))
            rows_local.append(
                {
                    "baseline": "direct_rule_only",
                    "pcap": str(p),
                    "prob_benign": float(probs[0]),
                    "prob_malicious": float(probs[1]),
                    "pred_label": int(probs[1] > probs[0]),
                }
            )
        payload: dict[str, Any] = {
            "baseline_direct_rule_only_pcap_written_count": int(written),
            "baseline_direct_rule_only_pcap_apply_time_sec": float(baseline_apply_time_sec),
            "baseline_direct_rule_only_pcap_packet_count": int(baseline_packets_total),
            "baseline_direct_rule_only_pcap_pcaps_per_sec": float(written / baseline_apply_time_sec)
            if baseline_apply_time_sec > 0.0
            else float("nan"),
            "baseline_direct_rule_only_pcap_packet_throughput_pps": float(
                baseline_packets_total / baseline_apply_time_sec
            )
            if baseline_apply_time_sec > 0.0
            else float("nan"),
            "baseline_direct_rule_only_pcap_out_dir": str(direct_main_dir),
            "baseline_direct_rule_only_pcap_eval_failed_count": int(failed_eval_count),
        }
        if pmal_vals:
            payload["baseline_direct_rule_only_pcap_adv_prob_malicious_mean"] = float(np.mean(pmal_vals))
            payload["baseline_direct_rule_only_pcap_adv_pred_malicious_rate"] = float(np.mean(pred_vals))
        if target_l2_vals_local:
            payload["baseline_direct_rule_only_pcap_target_l2_mean"] = float(np.mean(target_l2_vals_local))
        if target_mae_vals_local:
            payload["baseline_direct_rule_only_pcap_target_mae_mean"] = float(np.mean(target_mae_vals_local))
        add_paper_pcap_metrics(
            payload,
            prefix="baseline_direct_rule_only_",
            adv_pred_malicious_rate=payload.get("baseline_direct_rule_only_pcap_adv_pred_malicious_rate"),
            orig_pred_malicious_rate=metrics_payload.get("pcap_orig_pred_malicious"),
            adv_prob_malicious=payload.get("baseline_direct_rule_only_pcap_adv_prob_malicious_mean"),
            target_l2=payload.get("baseline_direct_rule_only_pcap_target_l2_mean"),
            target_mae=payload.get("baseline_direct_rule_only_pcap_target_mae_mean"),
            alignment_coverage=None,
            runtime_sec=payload.get("baseline_direct_rule_only_pcap_apply_time_sec"),
            pcaps_per_sec=payload.get("baseline_direct_rule_only_pcap_pcaps_per_sec"),
            packets_per_sec=payload.get("baseline_direct_rule_only_pcap_packet_throughput_pps"),
        )
        return {"name": "direct_rule_only", "payload": payload, "rows": rows_local}

    def _evaluate_baseline_file(baseline_path: Path) -> dict[str, Any] | None:
        try:
            baseline_npz = np.load(baseline_path)
        except Exception as exc:
            print(f"[Stage3/BL][Warn] failed to load baseline npz {baseline_path}: {exc}")
            return None

        raw_name = baseline_npz.get("baseline_name")
        baseline_name = str(
            raw_name.tolist() if isinstance(raw_name, np.ndarray) and raw_name.shape == () else raw_name or ""
        ).strip()
        if not baseline_name:
            baseline_name = baseline_path.stem.replace("baseline_", "").replace("_samples", "")
        stage3_policy_raw = baseline_npz.get("stage3_policy")
        stage3_policy = str(
            stage3_policy_raw.tolist()
            if isinstance(stage3_policy_raw, np.ndarray) and stage3_policy_raw.shape == ()
            else stage3_policy_raw or ""
        ).strip()
        spec = get_paper_baseline_spec(baseline_name)
        if not stage3_policy and spec is not None:
            stage3_policy = str(spec.stage3_policy)
        if not stage3_policy:
            stage3_policy = "feature_only_random_remap"
        traffic_claim = bool(spec.traffic_space) if spec is not None else False

        baseline_adv = baseline_npz.get("adv_pre")
        if baseline_adv is None:
            baseline_adv = baseline_npz.get("adv")
        if baseline_adv is None:
            return None
        baseline_adv = np.nan_to_num(np.asarray(baseline_adv, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

        saved_feature_names = baseline_npz.get("feature_names")
        if saved_feature_names is not None:
            restored_names = [str(name) for name in np.asarray(saved_feature_names).tolist()]
            if restored_names != [str(name) for name in feature_names]:
                return None
        if baseline_adv.shape[1] != x_train.shape[1]:
            return None

        baseline_raw = preprocessor.inverse_transform(baseline_adv)
        evaluation_mode = "shared_backend"
        baseline_group = "shared_backend"
        native_realization_pending = False
        if stage3_policy == "native_packet_unimplemented":
            evaluation_mode = "shared_backend_proxy"
            baseline_group = "traffic_claimed_shared_backend_proxy"
            native_realization_pending = True
        if stage3_policy == "feature_only_random_remap":
            baseline_mods = build_random_remap_modifications(
                x_adv_raw=baseline_raw,
                x_ben_raw=x_ben_raw,
                feature_names=feature_names,
                seed=seed,
            )
            evaluation_mode = "feature_only_random_remap_control"
            baseline_group = "feature_only_control"
        elif remap_mode == "random":
            baseline_mods = build_random_remap_modifications(
                x_adv_raw=baseline_raw,
                x_ben_raw=x_ben_raw,
                feature_names=feature_names,
                seed=seed,
            )
        elif remap_use_direct:
            baseline_mods = build_rule_based_modifications(
                x_adv_raw=baseline_raw,
                x_ben_raw=x_ben_raw,
                feature_names=feature_names,
            )
        else:
            if remap_bundle is None:
                return None
            baseline_direct_mods = build_rule_based_modifications(
                x_adv_raw=baseline_raw,
                x_ben_raw=x_ben_raw,
                feature_names=feature_names,
            )
            baseline_learned_mods = predict_modifications(remap_bundle, baseline_adv, device=device)
            baseline_blend_alpha, _ = effective_blend_fn(
                baseline_learned_mods,
                baseline_direct_mods,
                remap_bundle.mod_names,
            )
            baseline_mods = blend_fn(
                baseline_learned_mods,
                baseline_direct_mods,
                baseline_blend_alpha,
                remap_bundle.mod_names,
            )
        baseline_mods = clip_modifications(baseline_mods)

        baseline_pcap_dir = baseline_root_dir / baseline_name
        written, baseline_packets_total, baseline_apply_time_sec, adv_pcaps = write_modified_pcaps(
            pkts,
            baseline_mods,
            baseline_pcap_dir,
            seed=seed,
            count=settings.pcap_apply_n,
            settings=settings,
            protocol_auto_fix=protocol_auto_fix,
            wrpcap_fn=wrpcap,
        )

        rows_local: list[dict[str, Any]] = []
        pmal_vals: list[float] = []
        pred_vals: list[int] = []
        target_l2_vals_local: list[float] = []
        target_mae_vals_local: list[float] = []
        failed_eval_count = 0
        for p in adv_pcaps:
            try:
                probs, meta, feat_pre = _classify_with_cache(p)
            except PcapFeatureExtractionError as exc:
                failed_eval_count += 1
                skipped_rows.append(
                    {
                        "baseline": baseline_name,
                        "baseline_group": baseline_group,
                        "evaluation_mode": evaluation_mode,
                        "status": "failed",
                        "reason": str(exc),
                    }
                )
                continue
            pmal_vals.append(float(probs[1]))
            pred_vals.append(int(probs[1] > probs[0]))
            target_idx = None
            try:
                stem = Path(p).stem
                if stem.startswith("adv_"):
                    target_idx = int(stem.split("_", 1)[1])
            except Exception as exc:
                print(f"[Stage3/BL][Warn] baseline PCAP index parse failed for {p}: {exc}")
                target_idx = None
            if target_idx is not None and target_idx < baseline_adv.shape[0]:
                feat_mean = feat_pre.mean(axis=0)
                diff = feat_mean - baseline_adv[target_idx]
                align_meta = meta.get("alignment") if isinstance(meta, dict) else None
                diff = aligned_feature_diff(diff, align_meta, feature_names)
                target_l2_vals_local.append(float(np.linalg.norm(diff)))
                target_mae_vals_local.append(float(np.mean(np.abs(diff))))
            rows_local.append(
                {
                    "baseline": baseline_name,
                    "baseline_group": baseline_group,
                    "evaluation_mode": evaluation_mode,
                    "pcap": str(p),
                    "prob_benign": float(probs[0]),
                    "prob_malicious": float(probs[1]),
                    "pred_label": int(probs[1] > probs[0]),
                }
            )

        payload: dict[str, Any] = {
            f"baseline_{baseline_name}_pcap_written_count": int(written),
            f"baseline_{baseline_name}_pcap_apply_time_sec": float(baseline_apply_time_sec),
            f"baseline_{baseline_name}_pcap_packet_count": int(baseline_packets_total),
            f"baseline_{baseline_name}_pcap_pcaps_per_sec": float(written / baseline_apply_time_sec)
            if baseline_apply_time_sec > 0.0
            else float("nan"),
            f"baseline_{baseline_name}_pcap_packet_throughput_pps": float(
                baseline_packets_total / baseline_apply_time_sec
            )
            if baseline_apply_time_sec > 0.0
            else float("nan"),
            f"baseline_{baseline_name}_pcap_out_dir": str(baseline_pcap_dir),
            f"baseline_{baseline_name}_pcap_eval_policy": evaluation_mode,
            f"baseline_{baseline_name}_pcap_traffic_claim": bool(traffic_claim),
            f"baseline_{baseline_name}_pcap_native_realization_pending": bool(native_realization_pending),
            f"baseline_{baseline_name}_pcap_eval_failed_count": int(failed_eval_count),
        }
        if pmal_vals:
            payload[f"baseline_{baseline_name}_pcap_adv_prob_malicious_mean"] = float(np.mean(pmal_vals))
            payload[f"baseline_{baseline_name}_pcap_adv_pred_malicious_rate"] = float(np.mean(pred_vals))
        if target_l2_vals_local:
            payload[f"baseline_{baseline_name}_pcap_target_l2_mean"] = float(np.mean(target_l2_vals_local))
        if target_mae_vals_local:
            payload[f"baseline_{baseline_name}_pcap_target_mae_mean"] = float(np.mean(target_mae_vals_local))
        add_paper_pcap_metrics(
            payload,
            prefix=f"baseline_{baseline_name}_",
            adv_pred_malicious_rate=payload.get(f"baseline_{baseline_name}_pcap_adv_pred_malicious_rate"),
            orig_pred_malicious_rate=metrics_payload.get("pcap_orig_pred_malicious"),
            adv_prob_malicious=payload.get(f"baseline_{baseline_name}_pcap_adv_prob_malicious_mean"),
            target_l2=payload.get(f"baseline_{baseline_name}_pcap_target_l2_mean"),
            target_mae=payload.get(f"baseline_{baseline_name}_pcap_target_mae_mean"),
            alignment_coverage=None,
            runtime_sec=payload.get(f"baseline_{baseline_name}_pcap_apply_time_sec"),
            pcaps_per_sec=payload.get(f"baseline_{baseline_name}_pcap_pcaps_per_sec"),
            packets_per_sec=payload.get(f"baseline_{baseline_name}_pcap_packet_throughput_pps"),
        )
        return {"name": baseline_name, "payload": payload, "rows": rows_local, "skipped_rows": []}

    tasks: list[Path | str] = []
    if main_adv_pre is not None and len(np.asarray(main_adv_pre)):
        tasks.append("direct_rule_only")
    tasks.extend(baseline_npz_files)

    worker_count = min(max(1, int(getattr(settings, "pcap_baseline_jobs", 1) or 1)), len(tasks)) if tasks else 1
    print(f"[Stage3/BL] evaluating {len(tasks)} baseline(s) for PCAP comparison (workers={worker_count})", flush=True)
    results: list[dict[str, Any]] = []
    completed = 0
    if worker_count > 1:
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            futures = {}
            for task in tasks:
                if task == "direct_rule_only":
                    futures[pool.submit(_evaluate_direct_rule_only)] = "direct_rule_only"
                else:
                    futures[pool.submit(_evaluate_baseline_file, Path(task))] = str(task)
            for future in as_completed(futures):
                result = future.result()
                completed += 1
                name = result.get("name", "?") if result else futures[future]
                print(f"[Stage3/BL] baseline {completed}/{len(tasks)} done: {name}", flush=True)
                if result is not None:
                    results.append(result)
    else:
        for idx, task in enumerate(tasks, start=1):
            task_label = str(task) if task == "direct_rule_only" else Path(task).name
            print(f"[Stage3/BL] baseline {idx}/{len(tasks)}: {task_label}", flush=True)
            result = _evaluate_direct_rule_only() if task == "direct_rule_only" else _evaluate_baseline_file(Path(task))
            if result is not None:
                results.append(result)

    for result in sorted(results, key=lambda item: str(item.get("name", ""))):
        metrics_payload.update(result.get("payload", {}))
        baseline_eval_rows.extend(result.get("rows", []))
        skipped_rows.extend(result.get("skipped_rows", []))

    if baseline_eval_rows:
        baseline_eval_csv = out_dir / "baseline_pcap_eval.csv"
        save_records_csv(
            baseline_eval_csv,
            baseline_eval_rows,
            fieldnames=[
                "baseline",
                "baseline_group",
                "evaluation_mode",
                "pcap",
                "prob_benign",
                "prob_malicious",
                "pred_label",
            ],
        )
        metrics_payload["baseline_pcap_eval_path"] = str(baseline_eval_csv)
    if skipped_rows:
        skipped_csv = out_dir / "baseline_pcap_skipped.csv"
        save_records_csv(
            skipped_csv,
            skipped_rows,
            fieldnames=["baseline", "baseline_group", "evaluation_mode", "status", "reason"],
        )
        metrics_payload["baseline_pcap_skipped_path"] = str(skipped_csv)
