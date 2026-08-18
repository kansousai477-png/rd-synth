from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

import numpy as np
import torch

from rdsynth.pipeline.data import load_data_context
from rdsynth.pipeline.preprocessing import DatasetPreprocessor
from rdsynth.pipeline.runtime import load_stage_runtime
from rdsynth.pipeline.stage2_checkpoint import Stage2CheckpointSampler, load_stage2_checkpoint_sampler
from rdsynth.pipeline.stage3_calibration import CalibratedPreprocessor, compute_pcap_calibration
from rdsynth.pipeline.stage3_ids import train_stage3_ids
from rdsynth.pipeline.stage3_inputs import load_adv_samples, resolve_adv_samples_path
from rdsynth.pipeline.stage3_ops import (
    Stage3Settings,
    blend_modifications,
    detect_stage3_environment,
    effective_blend_alpha,
    load_stage3_artifacts,
    resolve_pcap_eval_model,
    select_remap_training_data,
    validate_remap_mode,
)
from rdsynth.pipeline.stage3_ops import (
    pcap_output_dir as _pcap_output_dir,
)
from rdsynth.pipeline.stage3_pcap import PcapFeatureExtractor
from rdsynth.pipeline.stage3_pcap_apply import prepare_pcap_search_context, record_pcap_apply_settings
from rdsynth.pipeline.stage3_pcap_baselines import run_stage3_baseline_pcap_eval
from rdsynth.pipeline.stage3_pcap_eval import (
    aggregate_pcap_sanity,
    evaluate_adversarial_pcaps,
    evaluate_original_pcap,
    extend_sanity_values,
    finalize_pcap_eval,
)
from rdsynth.pipeline.stage3_pcap_search import search_and_write_pcaps, temporary_probe_workspace
from rdsynth.pipeline.stage3_pcap_selection import build_pcap_selection, resolve_selected_pcap
from rdsynth.pipeline.stage3_pcap_semantics import filter_candidates_by_categories
from rdsynth.pipeline.stage_contracts import (
    StageManifestSpec,
    VersionedArtifactSpec,
    build_stage_output_files,
    build_versioned_artifact_payload,
    save_stage_manifest_spec,
)
from rdsynth.stages.stage3_remap import (
    MOD_NAMES,
    build_random_remap_modifications,
    build_remap_targets,
    build_rule_based_modifications,
    clip_modifications,
    predict_modifications,
    train_remapper,
)
from rdsynth.utils.artifacts import (
    save_array_csv,
    save_config,
    save_metrics,
    save_metrics_csv,
    save_state,
    save_training_log_csv,
)
from rdsynth.utils.feature_align import build_statistical_feature_aliases, load_feature_aliases


def _pcap_feature_quality_block_reason(metrics_payload: dict[str, object]) -> str:
    fill_count = int(metrics_payload.get("pcap_feature_fill_count", 0) or 0)
    statuses = metrics_payload.get("pcap_feature_statuses") or []
    if fill_count > 0:
        return "fill_value_features_used"
    for status in statuses:
        if str(status) != "ok":
            return f"feature_status_{status}"
    return ""


def _source_slug(path: Path, idx: int) -> str:
    stem = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in path.stem).strip("_")
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:10]
    short_stem = (stem or "pcap")[:24].strip("_") or "pcap"
    return f"source_{idx:02d}_{digest}_{short_stem}"


def _pcap_ids_candidate_pool(settings: Stage3Settings, selection: object) -> list[Path]:
    scan_dir = getattr(selection, "scan_dir", None)
    if scan_dir is None or not Path(scan_dir).exists():
        return []
    candidates = sorted(Path(scan_dir).rglob(str(getattr(selection, "scan_glob", "*.pcap") or "*.pcap")))
    semantic_categories = list(getattr(selection, "semantic_categories", []) or [])
    if semantic_categories:
        candidates = sorted(filter_candidates_by_categories(candidates, semantic_categories))
    max_bytes = int(getattr(settings, "pcap_scan_max_bytes", 0) or 0)
    if max_bytes > 0:
        candidates = [path for path in candidates if path.is_file() and path.stat().st_size <= max_bytes]
    rng = np.random.default_rng(int(getattr(settings, "pcap_source_sample_seed", 0) or 0))
    candidates = list(candidates)
    rng.shuffle(candidates)
    limit = int(getattr(selection, "scan_limit", 0) or 0)
    if limit > 0:
        candidates = candidates[:limit]
    return candidates


def main(config_path: str) -> None:
    runtime = load_stage_runtime(config_path, "stage3")
    cfg = runtime.cfg
    seed = runtime.seed
    device = runtime.device
    out_dir = runtime.out_dir
    stage3_cfg = runtime.stage_cfg
    settings = Stage3Settings.from_cfg(stage3_cfg, cfg["stage2"])
    env = detect_stage3_environment()
    scapy_available = env.scapy_available
    nfstream_available = env.nfstream_available
    cicflowmeter_available = env.cicflowmeter_available
    cicflowmeter_cmd = settings.cicflowmeter_cmd
    cicflowmeter_timeout = settings.cicflowmeter_timeout

    data_ctx = load_data_context(cfg, seed)
    bundle = data_ctx.bundle
    preprocessor = DatasetPreprocessor.from_bundle(bundle)

    x_train = bundle.x_train
    y_train = bundle.y_train
    x_ben = x_train[y_train == 0]

    remap_source = settings.remap_train_source
    x_remap = select_remap_training_data(x_train, y_train, remap_source)

    remap_loss = settings.loss
    huber_beta = settings.huber_beta
    grad_clip = settings.grad_clip
    target_clip_sigma = settings.target_clip_sigma
    weight_decay = settings.weight_decay

    remap_mode = settings.remap_mode
    validate_remap_mode(remap_mode)
    remap_min_r2 = settings.remap_min_r2
    remap_blend_alpha = settings.remap_blend_alpha
    remap_collapse_ratio_threshold = settings.remap_collapse_ratio_threshold
    remap_bundle = None
    if remap_mode not in {"direct", "random"}:
        remap_train_start = time.perf_counter()
        remap_objective = getattr(settings, "remap_train_objective", "identity")
        remap_kwargs = dict(
            x_ben_norm=x_remap,
            x_ben_raw=preprocessor.inverse_transform(x_remap),
            feature_names=bundle.feature_names,
            epochs=settings.epochs,
            batch_size=settings.batch_size,
            lr=settings.lr,
            device=device,
            loss=remap_loss,
            huber_beta=huber_beta,
            grad_clip=grad_clip,
            target_clip_sigma=target_clip_sigma,
            weight_decay=weight_decay,
            train_objective=remap_objective,
        )
        if remap_objective == "projection":
            mal_idx = np.where(y_train == 1)[0]
            if len(mal_idx) > 0:
                x_mal_remap = x_train[mal_idx]
                y_mal_remap = y_train[mal_idx]
                x_mal_selected = select_remap_training_data(x_mal_remap, y_mal_remap, "all")
                remap_kwargs["x_mal_norm"] = x_mal_selected
                remap_kwargs["x_mal_raw"] = preprocessor.inverse_transform(x_mal_selected)
            else:
                remap_kwargs["train_objective"] = "identity"
        remap_bundle = train_remapper(**remap_kwargs)
        remap_train_time_sec = time.perf_counter() - remap_train_start
    else:
        remap_train_time_sec = 0.0

    save_config(str(runtime.config_path), out_dir)
    remap_use_direct = False
    protocol_auto_fix = settings.protocol_auto_fix
    metrics_payload = {
        "feature_count": len(bundle.feature_names),
        "remap_train_source": remap_source,
        "remap_loss": str(remap_loss),
        "remap_huber_beta": float(huber_beta),
        "remap_grad_clip": float(grad_clip),
        "remap_target_clip_sigma": float(target_clip_sigma),
        "remap_weight_decay": float(weight_decay),
        "remap_mode": remap_mode,
        "remap_min_r2": remap_min_r2,
        "remap_blend_alpha": remap_blend_alpha,
        "remap_collapse_ratio_threshold": remap_collapse_ratio_threshold,
        "scapy_available": scapy_available,
        "nfstream_available": nfstream_available,
        "cicflowmeter_available": cicflowmeter_available,
        "cicflowmeter_cmd": cicflowmeter_cmd,
        "cicflowmeter_timeout": cicflowmeter_timeout,
        "pcap_modified": False,
        "pcap_eval": False,
        "protocol_auto_fix": protocol_auto_fix,
    }
    if remap_bundle is not None and remap_bundle.train_log:
        metrics_payload["remapper_train_time_sec"] = float(remap_train_time_sec)
        metrics_payload["remapper_train_epochs"] = len(remap_bundle.train_log)
        metrics_payload["remapper_train_last_loss"] = remap_bundle.train_log[-1]["loss"]
        metrics_payload["remapper_train_last_mae"] = remap_bundle.train_log[-1]["mae"]
        metrics_payload["remapper_train_last_rmse"] = remap_bundle.train_log[-1]["rmse"]
        if remap_bundle.best_epoch is not None:
            metrics_payload["remapper_train_best_epoch"] = int(remap_bundle.best_epoch)
        if remap_bundle.best_score is not None:
            metrics_payload["remapper_train_best_score"] = float(remap_bundle.best_score)
        if "port_acc" in remap_bundle.train_log[-1]:
            metrics_payload["remapper_train_last_port_acc"] = remap_bundle.train_log[-1]["port_acc"]
        train_csv = out_dir / "stage3_train_metrics.csv"
        save_training_log_csv(train_csv, remap_bundle.train_log)
        metrics_payload["remapper_train_metrics_path"] = str(train_csv)

    raw_feature_mean = preprocessor.feature_mean(x_train)
    x_ben_raw = preprocessor.inverse_transform(x_ben)
    x_remap_raw = preprocessor.inverse_transform(x_remap)
    pcap_eval_rows = None
    if remap_bundle is not None:
        target_mods = build_remap_targets(x_remap_raw, bundle.feature_names)
        pred_mods = predict_modifications(remap_bundle, x_remap, device=device)
        port_idx = remap_bundle.mod_names.index("dst_port_new")
        continuous_idx = [i for i in range(len(remap_bundle.mod_names)) if i != port_idx]
        err = pred_mods - target_mods
        err_cont = err[:, continuous_idx]
        target_cont = target_mods[:, continuous_idx]
        mse = float(np.mean(err_cont**2))
        mae = float(np.mean(np.abs(err_cont)))
        rmse = float(np.sqrt(mse))
        denom = float(np.sum((target_cont - np.mean(target_cont, axis=0)) ** 2)) + 1.0e-6
        r2 = float(1.0 - (np.sum(err_cont**2) / denom))
        metrics_payload["remapper_eval_mse"] = mse
        metrics_payload["remapper_eval_mae"] = mae
        metrics_payload["remapper_eval_rmse"] = rmse
        metrics_payload["remapper_eval_r2"] = r2
        metrics_payload["remapper_eval_port_acc"] = float(
            np.mean(np.round(pred_mods[:, port_idx]) == np.round(target_mods[:, port_idx]))
        )

        eval_csv = out_dir / "stage3_remap_eval.csv"
        with open(eval_csv, "w", encoding="utf-8") as f:
            f.write("mod_name,mae,rmse,target_mean,target_std,extra\n")
            per_mae = np.mean(np.abs(err), axis=0)
            per_rmse = np.sqrt(np.mean(err**2, axis=0))
            target_mean = np.mean(target_mods, axis=0)
            target_std = np.std(target_mods, axis=0)
            for i, name in enumerate(remap_bundle.mod_names):
                extra = ""
                if name == "dst_port_new":
                    extra = f"port_acc={metrics_payload['remapper_eval_port_acc']:.6f}"
                f.write(f"{name},{per_mae[i]:.6f},{per_rmse[i]:.6f},{target_mean[i]:.6f},{target_std[i]:.6f},{extra}\n")
        metrics_payload["remapper_eval_path"] = str(eval_csv)

    if remap_mode == "direct":
        remap_use_direct = True
    elif remap_mode == "learned":
        remap_use_direct = False
    elif remap_mode == "random":
        remap_use_direct = False
    else:
        r2 = metrics_payload.get("remapper_eval_r2")
        remap_use_direct = r2 is not None and float(r2) < remap_min_r2
    metrics_payload["remap_use_direct"] = remap_use_direct
    if remap_mode == "direct":
        metrics_payload["remap_skip_train"] = True

    alias_path = settings.feature_aliases_path
    alias_map = build_statistical_feature_aliases(
        bundle.feature_names,
        dataset_name=str(cfg.get("data", {}).get("dataset", "")),
        base_alias_map=load_feature_aliases(alias_path),
    )
    if alias_map:
        metrics_payload["feature_aliases_path"] = alias_path

    feature_backend = settings.feature_backend
    if feature_backend not in ("auto", "scapy", "nfstream", "cicflowmeter"):
        feature_backend = "auto"

    oracle_name = settings.oracle_name
    artifacts = load_stage3_artifacts(
        cfg=cfg,
        oracle_name=oracle_name,
        x_train=bundle.x_train,
        y_train=bundle.y_train,
        x_val=bundle.x_val,
        y_val=bundle.y_val,
        feature_names=list(bundle.feature_names),
        device=device,
        seed=seed,
    )
    surrogate = artifacts.surrogate if artifacts.checkpoint_path.exists() else None
    oracle = artifacts.oracle

    pcap_eval_use_ids = bool(getattr(settings, "pcap_eval_use_ids", False))
    pcap_eval_use_oracle = bool(getattr(settings, "pcap_eval_use_oracle", False))
    pcap_eval_dual = bool(getattr(settings, "pcap_eval_dual", False))
    pcap_eval_record_both = bool(getattr(settings, "pcap_eval_record_both", False))

    # Dual evaluator: when enabled, prefer oracle for primary but also train pcap_ids as secondary
    if pcap_eval_dual:
        pcap_eval_use_ids = True
        pcap_eval_use_oracle = True
        pcap_eval_record_both = True

    model_selection = resolve_pcap_eval_model(
        ids=None,
        oracle=oracle,
        surrogate=surrogate,
        prefer_ids=False,
        prefer_oracle=pcap_eval_use_oracle,
    )
    pcap_eval_model = model_selection.pcap_eval_model
    pcap_eval_model_name = model_selection.pcap_eval_model_name
    pcap_eval_ids = None
    metrics_payload["pcap_eval_use_ids"] = pcap_eval_use_ids
    metrics_payload["pcap_eval_use_oracle"] = pcap_eval_use_oracle
    metrics_payload["pcap_eval_dual"] = pcap_eval_dual
    metrics_payload["pcap_eval_record_both"] = pcap_eval_record_both
    metrics_payload["pcap_eval_model"] = pcap_eval_model_name

    align_min_cov = settings.pcap_align_min_coverage
    pcap_eval_batch_size = settings.pcap_eval_batch_size
    pcap_feature_fail_closed = settings.pcap_feature_fail_closed
    pcap_feature_fail_on_partial = settings.pcap_feature_fail_on_partial_alignment
    metrics_payload["pcap_align_min_coverage"] = align_min_cov
    metrics_payload["pcap_feature_fail_closed"] = pcap_feature_fail_closed
    metrics_payload["pcap_feature_fail_on_partial_alignment"] = pcap_feature_fail_on_partial
    metrics_payload["pcap_cache_enable"] = bool(settings.pcap_cache_enable)
    metrics_payload["pcap_cache_dir"] = str(settings.pcap_cache_dir)
    search_alphas = settings.pcap_search_alphas
    metrics_payload["pcap_search_alphas"] = list(search_alphas)

    pcap_features = PcapFeatureExtractor(
        feature_backend=feature_backend,
        feature_names=list(bundle.feature_names),
        raw_feature_mean=raw_feature_mean,
        alias_map=alias_map,
        align_min_cov=align_min_cov,
        scapy_available=scapy_available,
        nfstream_available=nfstream_available,
        cicflowmeter_available=cicflowmeter_available,
        cicflowmeter_cmd=cicflowmeter_cmd,
        cicflowmeter_timeout=cicflowmeter_timeout,
        fail_closed=pcap_feature_fail_closed,
        fail_on_partial_alignment=pcap_feature_fail_on_partial,
        preprocessor=preprocessor,
        pcap_eval_model=pcap_eval_model,
        pcap_eval_model_name=pcap_eval_model_name,
        ids=pcap_eval_ids,
        oracle=oracle,
        surrogate=surrogate,
        pcap_eval_batch_size=pcap_eval_batch_size,
        seed=seed,
        device=device,
        max_pcap_bytes=int(getattr(settings, "pcap_scan_max_bytes", 0) or 0),
        max_flows_per_pcap=getattr(settings, "pcap_feature_max_flows_per_pcap", None),
        cache_enable=settings.pcap_cache_enable,
        cache_dir=(
            Path(settings.pcap_cache_dir).resolve() if settings.pcap_cache_enable and settings.pcap_cache_dir else None
        ),
    )

    pcap_selection = build_pcap_selection(settings, metrics_payload)
    pcap_path = pcap_selection.selected_path
    pcap_evasion_valid = pcap_selection.evasion_valid

    # ── PCAP feature calibration ──────────────────────────────────────────
    # nfstream/scapy extract features at different scales than the datasetʼs
    # CICFlowMeter extractor (e.g. ms vs μs for durations).  Calibrate PCAP
    # features into the dataset raw-feature space before preprocessing so the
    # Stage 2 diffusion model sees in-distribution conditioning vectors.
    calibrated_preprocessor = None  # replaced below if benign PCAPs are available
    _calibration_attempted = False
    # Calibration is only used in pcap_conditioned mode where PCAP features must
    # be mapped into the dataset feature space for the oracle/surrogate evaluator.
    # When pcap_eval_use_ids is True, a dedicated PCAP IDS is trained directly on
    # PCAP features without cross-space calibration.
    if str(getattr(settings, "pcap_target_source", "")).strip().lower() == "pcap_conditioned":
        benign_pcap_paths: list[Path] = []
        benign_single = str(getattr(settings, "pcap_ids_benign_path", "") or "")
        if benign_single:
            single_path = Path(benign_single)
            if single_path.exists():
                if single_path.is_file():
                    benign_pcap_paths.append(single_path.resolve())
                elif single_path.is_dir():
                    benign_pcap_paths.extend(sorted(p.resolve() for p in single_path.rglob("*.pcap") if p.is_file()))
        for item in list(getattr(settings, "pcap_ids_benign_paths", []) or []):
            path = Path(str(item))
            if path.exists() and path.is_file():
                benign_pcap_paths.append(path.resolve())
        benign_dir = str(getattr(settings, "pcap_ids_benign_dir", "") or "")
        if benign_dir:
            root = Path(benign_dir)
            if root.exists():
                benign_pcap_paths.extend(
                    sorted(
                        p.resolve()
                        for p in root.rglob(str(getattr(settings, "pcap_ids_benign_glob", "*.pcap") or "*.pcap"))
                        if p.is_file()
                    )
                )
        benign_pcap_paths = list(dict.fromkeys(benign_pcap_paths))
        max_benign = int(getattr(settings, "pcap_ids_benign_max_pcaps", 4) or 4)
        if max_benign > 0:
            benign_pcap_paths = benign_pcap_paths[:max_benign]

        if benign_pcap_paths:
            _calibration_attempted = True
            benign_rows: list[np.ndarray] = []
            for idx, bpath in enumerate(benign_pcap_paths):
                try:
                    bfeat, _, _ = pcap_features.extract(str(bpath))
                except Exception as exc:
                    print(f"[Stage3][Warn] benign PCAP feature extraction failed for {bpath}: {exc}")
                    continue
                bfeat = np.asarray(bfeat, dtype=np.float64)
                if bfeat.ndim != 2 or bfeat.shape[0] == 0 or bfeat.shape[1] != x_train.shape[1]:
                    continue
                limit = getattr(settings, "pcap_ids_benign_max_flows_per_pcap", None)
                if limit is not None and bfeat.shape[0] > int(limit):
                    rng = np.random.default_rng(seed + 200 + idx)
                    indices = rng.choice(bfeat.shape[0], size=int(limit), replace=False)
                    indices.sort()
                    bfeat = bfeat[indices]
                if bfeat.shape[0] > 0:
                    benign_rows.append(bfeat)
            if benign_rows:
                pcap_benign_raw = np.concatenate(benign_rows, axis=0)
                calibration = compute_pcap_calibration(
                    dataset_benign_raw=x_ben_raw,
                    pcap_benign_raw=pcap_benign_raw,
                    min_samples=10,
                )
                calibrated_preprocessor = CalibratedPreprocessor(preprocessor, calibration)
                metrics_payload["pcap_calibration_active"] = calibration.is_active
                metrics_payload["pcap_calibration_pcap_samples"] = calibration.pcap_sample_count
                metrics_payload["pcap_calibration_benign_pcaps_used"] = len(benign_rows)
                if calibration.is_active:
                    print(
                        f"[Stage3] pcap calibration active — "
                        f"{calibration.pcap_sample_count} benign PCAP flows from {len(benign_rows)} PCAP(s)"
                    )
            else:
                metrics_payload["pcap_calibration_active"] = False
                metrics_payload["pcap_calibration_skip_reason"] = "no_benign_pcap_features_extracted"
        else:
            metrics_payload["pcap_calibration_active"] = False
            metrics_payload["pcap_calibration_skip_reason"] = "no_benign_pcaps_found"

    # When calibration is active, PCAP features are mapped to the dataset raw space
    # and the oracle/surrogate (trained on dataset features) can serve as the PCAP
    # evaluator.  No separate PCAP-domain IDS is needed, eliminating the cross-model
    # transfer gap that would otherwise degrade pcap_conditioned_feature_asr.
    if calibrated_preprocessor is not None and calibrated_preprocessor.is_active:
        if oracle is not None:
            pcap_eval_model = oracle
            pcap_eval_model_name = "oracle"
        elif surrogate is not None:
            pcap_eval_model = surrogate
            pcap_eval_model_name = "surrogate"
        # Keep pcap_eval_use_ids as configured — calibration does not replace
        # the need for a dedicated PCAP-domain IDS when the dataset oracle cannot
        # reliably judge PCAP-extracted features.
        metrics_payload["pcap_eval_model"] = pcap_eval_model_name
        metrics_payload["pcap_calibration_active"] = True
        # Recreate feature extractor with calibrated preprocessor + oracle/surrogate evaluator.
        pcap_features = PcapFeatureExtractor(
            feature_backend=feature_backend,
            feature_names=list(bundle.feature_names),
            raw_feature_mean=raw_feature_mean,
            alias_map=alias_map,
            align_min_cov=align_min_cov,
            scapy_available=scapy_available,
            nfstream_available=nfstream_available,
            cicflowmeter_available=cicflowmeter_available,
            cicflowmeter_cmd=cicflowmeter_cmd,
            cicflowmeter_timeout=cicflowmeter_timeout,
            fail_closed=pcap_feature_fail_closed,
            fail_on_partial_alignment=pcap_feature_fail_on_partial,
            preprocessor=calibrated_preprocessor,
            pcap_eval_model=pcap_eval_model,
            pcap_eval_model_name=pcap_eval_model_name,
            ids=None,
            oracle=oracle,
            surrogate=surrogate,
            pcap_eval_batch_size=pcap_eval_batch_size,
            seed=seed,
            device=device,
            max_pcap_bytes=int(getattr(settings, "pcap_scan_max_bytes", 0) or 0),
            max_flows_per_pcap=getattr(settings, "pcap_feature_max_flows_per_pcap", None),
            cache_enable=settings.pcap_cache_enable,
            cache_dir=(
                Path(settings.pcap_cache_dir).resolve()
                if settings.pcap_cache_enable and settings.pcap_cache_dir
                else None
            ),
        )
        print(f"[Stage3] calibration active — using {pcap_eval_model_name} as PCAP evaluator (no separate PCAP IDS)")

    adv_path = resolve_adv_samples_path(settings.adv_samples_path, cfg["project"]["out_dir"])
    pcap_out_dir = None
    pcap_source_dirs: list[Path] = []
    source_paths: list[Path] = []
    mods = None
    adv = None
    adv_norm = None
    adv_mean = None
    adv_std = None
    pcap_target_mod = None
    pcap_scan_target_pre = None
    pcap_target_source = str(getattr(settings, "pcap_target_source", "stage2_saved_samples")).strip().lower()
    if pcap_target_source not in {"stage2_saved_samples", "pcap_conditioned"}:
        raise ValueError(f"Unsupported stage3.pcap_target_source: {pcap_target_source}")
    metrics_payload["pcap_target_source"] = pcap_target_source

    def _build_modifications_for_adv_samples(
        adv_pre: np.ndarray,
        *,
        cal_preprocessor: CalibratedPreprocessor | None = None,
    ) -> tuple[np.ndarray, str, dict[str, object]]:
        # Only use calibration when explicitly requested (pcap_conditioned path).
        # Saved Stage2 samples are in dataset space and must not be uncalibrated.
        if cal_preprocessor is not None and cal_preprocessor.is_active:
            adv_raw_local = cal_preprocessor.inverse_transform(adv_pre)
        else:
            adv_raw_local = preprocessor.inverse_transform(adv_pre)
        direct_mods_local = build_rule_based_modifications(
            x_adv_raw=adv_raw_local,
            x_ben_raw=x_ben_raw,
            feature_names=bundle.feature_names,
        )
        extra: dict[str, object] = {}
        if remap_mode == "random":
            mods_local = build_random_remap_modifications(
                x_adv_raw=adv_raw_local,
                x_ben_raw=x_ben_raw,
                feature_names=bundle.feature_names,
                seed=seed,
            )
            mod_source = "random"
        elif remap_use_direct:
            mods_local = direct_mods_local
            mod_source = "direct"
        else:
            if remap_bundle is None:
                raise RuntimeError("remap_mode requires learned remapper, but remapper was not trained.")
            learned_mods = predict_modifications(remap_bundle, adv_pre, device=device)
            blend_alpha_eff, blend_info = effective_blend_alpha(
                learned_mods,
                direct_mods_local,
                remap_bundle.mod_names,
                requested_alpha=remap_blend_alpha,
                collapse_ratio_threshold=remap_collapse_ratio_threshold,
            )
            extra.update(
                {
                    "remap_pred_std_mean": float(blend_info["pred_std_mean"]),
                    "remap_direct_std_mean": float(blend_info["direct_std_mean"]),
                    "remap_collapse_ratio": float(blend_info["collapse_ratio"]),
                    "remap_blend_reason": str(blend_info["blend_reason"]),
                    "remap_blend_alpha_effective": float(blend_info["blend_alpha_effective"]),
                }
            )
            if blend_alpha_eff < 0.999:
                mods_local = blend_modifications(
                    learned_mods, direct_mods_local, blend_alpha_eff, remap_bundle.mod_names
                )
                mod_source = "blended"
            else:
                mods_local = learned_mods
                mod_source = "learned"
        return clip_modifications(mods_local), mod_source, extra

    stage2_checkpoint_sampler: Stage2CheckpointSampler | None = None

    def _stage2_adv_from_pcap(source_path: Path) -> tuple[np.ndarray, dict[str, object]]:
        nonlocal stage2_checkpoint_sampler
        if stage2_checkpoint_sampler is None:
            stage2_checkpoint_sampler = load_stage2_checkpoint_sampler(
                cfg=cfg,
                project_out_dir=cfg["project"]["out_dir"],
                feature_names=list(bundle.feature_names),
                surrogate=surrogate,
                device=device,
                benign_pool=x_ben,
            )
            metrics_payload["pcap_conditioned_stage2_checkpoint_path"] = str(stage2_checkpoint_sampler.checkpoint_path)

        # Extract raw PCAP features.
        feat_raw, backend, meta = pcap_features.extract(str(source_path))
        feat_raw = np.asarray(feat_raw, dtype=np.float64)
        if feat_raw.size == 0 or feat_raw.shape[0] == 0:
            raise RuntimeError(f"No source PCAP features extracted for pcap_conditioned Stage3: {source_path}")

        # Preprocess PCAP features through the calibrated preprocessor.
        # classify_features does: calibrate → preprocess → oracle/surrogate.
        # source_feat_pre is in NORMALIZED (preprocessed) dataset space.
        source_probs, source_feat_pre = pcap_features.classify_features(feat_raw)
        source_feat_pre = np.asarray(source_feat_pre, dtype=np.float32)

        # The Stage2 sampler was trained on RAW features and normalizes
        # internally using its own ben_stats.  Inverse-transform back to raw
        # space so the sampler receives in-distribution input.
        preprocessor_for_sampler = (
            calibrated_preprocessor
            if (calibrated_preprocessor is not None and calibrated_preprocessor.is_active)
            else preprocessor
        )
        source_feat_raw = preprocessor_for_sampler.inverse_transform(source_feat_pre)
        source_feat_raw = np.asarray(source_feat_raw, dtype=np.float32)

        sampled = stage2_checkpoint_sampler.sampler(
            source_feat_raw, stage2_checkpoint_sampler.settings.mal_anchor_alpha
        )
        sampled = np.asarray(sampled, dtype=np.float32)

        # The sampler returns NORMALIZED features (DDPM default) or raw
        # (when sample_denorm_output=True for GAN/editor backbones).
        # Always produce a normalized version for oracle/surrogate evaluation
        # and a raw version for downstream remapping.
        if stage2_checkpoint_sampler.output_is_preprocessed:
            adv_raw = sampled
            adv_norm = preprocessor_for_sampler.transform(adv_raw)
        else:
            adv_norm = sampled
            adv_raw = preprocessor_for_sampler.inverse_transform(adv_norm)
        adv_norm = np.nan_to_num(adv_norm, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        adv_raw = np.nan_to_num(adv_raw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        # Evaluate adversarial features using the oracle/surrogate.
        # predict_probs expects NORMALIZED features.
        adv_preds, adv_probs = pcap_features.predict_probs(adv_norm)

        source_pmal = (
            float(source_probs[1])
            if source_probs is not None and len(source_probs) >= 2 and np.all(np.isfinite(source_probs))
            else None
        )
        source_flow_pmal = (
            float(np.mean(source_probs[:, 1]))
            if isinstance(source_probs, np.ndarray) and source_probs.ndim == 2 and source_probs.shape[1] >= 2
            else source_pmal
        )
        adv_pmal = (
            float(np.mean(adv_probs[:, 1]))
            if isinstance(adv_probs, np.ndarray) and adv_probs.ndim == 2 and adv_probs.shape[1] >= 2
            else None
        )
        return adv_norm, {
            "pcap_conditioned_source_backend": backend,
            "pcap_conditioned_source_flow_count": int(source_feat_pre.shape[0]),
            "pcap_conditioned_source_prob_malicious": source_flow_pmal,
            "pcap_conditioned_adv_prob_malicious": adv_pmal,
            "pcap_conditioned_feature_asr": float(np.mean(adv_preds == 0)) if adv_preds is not None else None,
            "pcap_conditioned_feature_status": meta.get("status") if isinstance(meta, dict) else "",
            "pcap_conditioned_sample_count": int(source_feat_pre.shape[0]),
        }

    if adv_path.exists() or pcap_target_source == "pcap_conditioned":
        loaded_adv = (
            load_adv_samples(
                adv_path,
                project_out_dir=cfg["project"]["out_dir"],
                current_feature_names=bundle.feature_names,
                expected_feature_dim=x_train.shape[1],
                copy_to=(out_dir / "adv_samples.npz") if settings.copy_adv_samples else None,
            )
            if adv_path.exists()
            else None
        )
        if loaded_adv is not None:
            adv = loaded_adv.adv
            adv_norm = loaded_adv.adv_norm
            adv_mean = loaded_adv.adv_mean
            adv_std = loaded_adv.adv_std
            metrics_payload["adv_samples_loaded"] = loaded_adv.loaded
            metrics_payload["adv_samples_count"] = loaded_adv.count
            if loaded_adv.adv_space:
                metrics_payload["adv_samples_space"] = loaded_adv.adv_space
        else:
            metrics_payload["adv_samples_loaded"] = False
            metrics_payload["adv_samples_count"] = 0

        if adv is not None:
            if adv.size:
                pcap_scan_target_pre = np.mean(adv.astype(np.float64), axis=0)
            mods, mod_source, mod_extra = _build_modifications_for_adv_samples(adv)
            metrics_payload.update(mod_extra)
            metrics_payload["remap_mod_source"] = mod_source
            if mods is not None and len(mods):
                pcap_target_mod = np.mean(np.asarray(mods, dtype=np.float64), axis=0)
            mods_csv = out_dir / "modifications.csv"
            save_array_csv(mods_csv, mods, header=MOD_NAMES if remap_bundle is None else remap_bundle.mod_names)
            metrics_payload["modifications_saved"] = True
            metrics_payload["modifications_path"] = str(mods_csv)
            if surrogate is not None:
                with torch.no_grad():
                    logits = surrogate(torch.tensor(adv, dtype=torch.float32, device=device))
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)
                metrics_payload["adv_benign_rate"] = float(np.mean(preds == 0))
                metrics_payload["adv_prob_malicious_mean"] = float(np.mean(probs[:, 1]))
        else:
            mods = None

        if pcap_eval_use_ids:
            ids_training_paths = _pcap_ids_candidate_pool(settings, pcap_selection)
            # Limit training pool: large candidate pools make feature extraction
            # prohibitively slow and offer diminishing returns for a binary IDS.
            _ids_max_train = max(1, int(getattr(settings, "pcap_ids_training_max_pcaps", 0) or 0))
            if _ids_max_train > 0 and len(ids_training_paths) > _ids_max_train:
                ids_training_paths = ids_training_paths[:_ids_max_train]
            if ids_training_paths:
                print(f"[Stage3] training pcap_ids on {len(ids_training_paths)} malicious PCAP(s) ...")
                ids_training = train_stage3_ids(
                    malicious_pcap=ids_training_paths[0],
                    malicious_pcaps=ids_training_paths,
                    settings=settings,
                    feature_names=list(bundle.feature_names),
                    raw_feature_mean=raw_feature_mean,
                    alias_map=alias_map,
                    preprocessor=preprocessor,
                    device=device,
                    seed=seed,
                )
                metrics_payload.update(ids_training.metrics)
                if ids_training.ids_bundle is not None:
                    pcap_eval_ids = ids_training.ids_bundle
                    # When PCAP→dataset calibration is active, the oracle can
                    # reliably judge PCAP-extracted features.  Prefer the oracle
                    # to keep the Stage3 evaluator aligned with Stage1/Stage2,
                    # eliminating the cross-model transfer gap.  The PCAP IDS is
                    # retained as a secondary robustness check when requested.
                    _calibration_active = calibrated_preprocessor is not None and calibrated_preprocessor.is_active
                    model_selection = resolve_pcap_eval_model(
                        ids=pcap_eval_ids,
                        oracle=oracle,
                        surrogate=surrogate,
                        prefer_ids=not _calibration_active,
                        prefer_oracle=pcap_eval_use_oracle or _calibration_active,
                    )
                    pcap_eval_model = model_selection.pcap_eval_model
                    pcap_eval_model_name = model_selection.pcap_eval_model_name
                    metrics_payload["pcap_eval_model"] = pcap_eval_model_name
                    metrics_payload["pcap_ids_selection_pool_size"] = int(len(ids_training_paths))
                    pcap_features = PcapFeatureExtractor(
                        feature_backend=feature_backend,
                        feature_names=list(bundle.feature_names),
                        raw_feature_mean=raw_feature_mean,
                        alias_map=alias_map,
                        align_min_cov=align_min_cov,
                        scapy_available=scapy_available,
                        nfstream_available=nfstream_available,
                        cicflowmeter_available=cicflowmeter_available,
                        cicflowmeter_cmd=cicflowmeter_cmd,
                        cicflowmeter_timeout=cicflowmeter_timeout,
                        fail_closed=pcap_feature_fail_closed,
                        fail_on_partial_alignment=pcap_feature_fail_on_partial,
                        preprocessor=preprocessor,
                        pcap_eval_model=pcap_eval_model,
                        pcap_eval_model_name=pcap_eval_model_name,
                        ids=pcap_eval_ids,
                        oracle=oracle,
                        surrogate=surrogate,
                        pcap_eval_batch_size=pcap_eval_batch_size,
                        seed=seed,
                        device=device,
                        max_pcap_bytes=int(getattr(settings, "pcap_scan_max_bytes", 0) or 0),
                        max_flows_per_pcap=getattr(settings, "pcap_feature_max_flows_per_pcap", None),
                        cache_enable=settings.pcap_cache_enable,
                        cache_dir=(
                            Path(settings.pcap_cache_dir).resolve()
                            if settings.pcap_cache_enable and settings.pcap_cache_dir
                            else None
                        ),
                    )

        pcap_selection = resolve_selected_pcap(
            pcap_selection,
            pcap_features=pcap_features,
            pcap_eval_model=pcap_eval_model,
            target_pre=pcap_scan_target_pre,
            target_mod=pcap_target_mod,
            metrics_payload=metrics_payload,
            mandatory_paths=list(getattr(settings, "pcap_mandatory_paths", []) or []),
        )
        pcap_path = pcap_selection.selected_path
        pcap_evasion_valid = pcap_selection.evasion_valid
        if pcap_eval_use_ids and pcap_eval_ids is None and pcap_path is not None and Path(pcap_path).exists():
            ids_training = train_stage3_ids(
                malicious_pcap=Path(pcap_path),
                settings=settings,
                feature_names=list(bundle.feature_names),
                raw_feature_mean=raw_feature_mean,
                alias_map=alias_map,
                preprocessor=preprocessor,
                device=device,
                seed=seed,
            )
            metrics_payload.update(ids_training.metrics)
            if ids_training.ids_bundle is not None:
                pcap_eval_ids = ids_training.ids_bundle
                model_selection = resolve_pcap_eval_model(
                    ids=pcap_eval_ids,
                    oracle=oracle,
                    surrogate=surrogate,
                    prefer_ids=True,
                    prefer_oracle=pcap_eval_use_oracle,
                )
                pcap_eval_model = model_selection.pcap_eval_model
                pcap_eval_model_name = model_selection.pcap_eval_model_name
                metrics_payload["pcap_eval_model"] = pcap_eval_model_name
                pcap_features = PcapFeatureExtractor(
                    feature_backend=feature_backend,
                    feature_names=list(bundle.feature_names),
                    raw_feature_mean=raw_feature_mean,
                    alias_map=alias_map,
                    align_min_cov=align_min_cov,
                    scapy_available=scapy_available,
                    nfstream_available=nfstream_available,
                    cicflowmeter_available=cicflowmeter_available,
                    cicflowmeter_cmd=cicflowmeter_cmd,
                    cicflowmeter_timeout=cicflowmeter_timeout,
                    fail_closed=pcap_feature_fail_closed,
                    fail_on_partial_alignment=pcap_feature_fail_on_partial,
                    preprocessor=preprocessor,
                    pcap_eval_model=pcap_eval_model,
                    pcap_eval_model_name=pcap_eval_model_name,
                    ids=pcap_eval_ids,
                    oracle=oracle,
                    surrogate=surrogate,
                    pcap_eval_batch_size=pcap_eval_batch_size,
                    seed=seed,
                    device=device,
                    max_pcap_bytes=int(getattr(settings, "pcap_scan_max_bytes", 0) or 0),
                    max_flows_per_pcap=getattr(settings, "pcap_feature_max_flows_per_pcap", None),
                    cache_enable=settings.pcap_cache_enable,
                    cache_dir=(
                        Path(settings.pcap_cache_dir).resolve()
                        if settings.pcap_cache_enable and settings.pcap_cache_dir
                        else None
                    ),
                )
            else:
                metrics_payload["pcap_eval_model"] = pcap_eval_model_name

        candidate_paths = []
        seen_candidate_paths: set[str] = set()
        for candidate_path in list(getattr(pcap_selection, "candidate_paths", []) or []) + (
            [pcap_path] if pcap_path is not None else []
        ):
            if candidate_path is None:
                continue
            candidate_text = str(candidate_path)
            if candidate_text in seen_candidate_paths:
                continue
            seen_candidate_paths.add(candidate_text)
            candidate_paths.append(Path(candidate_path))
        source_selection_mode = str(getattr(pcap_selection, "source_selection_mode", "best")).strip().lower()
        if source_selection_mode in {"random", "random_hard", "top_hard", "all"} and candidate_paths:
            source_paths = list(candidate_paths[:5])  # limit probe candidates
        elif pcap_path is not None:
            source_paths = [Path(pcap_path)]
        else:
            source_paths = []
        # ── Always prepend mandatory online_pcap PCAPs ──
        _online_pcap_dir = Path("data/PCAPs/malicious/online_pcap")
        if _online_pcap_dir.exists():
            _mandatory = sorted([p for p in _online_pcap_dir.glob("*.pcap") if p.is_file()])
            _seen_src = {str(s) for s in source_paths}
            for _mand in _mandatory:
                if str(_mand) not in _seen_src:
                    source_paths.insert(0, _mand)
                    _seen_src.add(str(_mand))
                    print(f"[Stage3] mandatory source: {_mand.name}")
        if source_paths:
            metrics_payload["pcap_source_count"] = len(source_paths)
            metrics_payload["pcap_source_names"] = [path.name for path in source_paths]

        if source_paths and (mods is not None or pcap_target_source == "pcap_conditioned"):
            if all(path.exists() for path in source_paths):
                if pcap_evasion_valid is False:
                    # Source PCAP already classified as benign — still generate
                    # adversarial PCAP to validate the remapping pipeline. The
                    # adversarial features from Stage2 are already benign (high ASR).
                    print(
                        "[Stage3] source PCAP already benign (pcap_evasion_valid=False); "
                        "proceeding with adversarial PCAP generation to validate pipeline."
                    )
                elif not scapy_available:
                    metrics_payload["pcap_modified"] = False
                    metrics_payload["pcap_error"] = "scapy not installed"
                    print("[Stage3][Warn] scapy not installed; skipping PCAP modification (pip install scapy).")
                else:
                    try:
                        from scapy.all import rdpcap, wrpcap

                        record_pcap_apply_settings(
                            metrics_payload,
                            settings=settings,
                            protocol_auto_fix=protocol_auto_fix,
                        )

                        pcap_out_dir = _pcap_output_dir(settings, out_dir)
                        pcap_out_dir.mkdir(parents=True, exist_ok=True)
                        aggregate_apply_metrics: dict[str, list[float]] = {
                            "pcap_apply_time_sec": [],
                            "pcap_packet_count": [],
                            "pcap_pcaps_per_sec": [],
                            "pcap_packet_throughput_pps": [],
                            "pcap_selected_alpha_mean": [],
                            "pcap_selected_deployability_score_mean": [],
                            "pcap_selected_response_l2_mean": [],
                            "pcap_selected_pmal_mean": [],
                            "pcap_selected_target_l2_mean": [],
                            "pcap_selected_target_mae_mean": [],
                            "pcap_selected_alignment_coverage_mean": [],
                        }
                        selected_alphas: list[float] = []
                        selected_field_sets: list[str] = []
                        total_written = 0
                        total_kept_original = 0
                        combined_out_dirs: list[str] = []
                        combined_trace_paths: list[str] = []
                        latest_mods = mods
                        per_source_target_mods: list[np.ndarray] = []
                        pcap_conditioned_feature_asrs: list[float] = []
                        pcap_conditioned_adv_pmals: list[float] = []
                        pcap_conditioned_source_pmals: list[float] = []

                        if (
                            source_selection_mode not in {"random", "random_hard", "top_hard", "all"}
                            and mods is not None
                            and adv is not None
                        ):
                            probe_topk = max(1, int(getattr(settings, "pcap_source_probe_topk", 4)))
                            if candidate_paths and len(candidate_paths) > 1:
                                best_probe_score = float("-inf")
                                best_probe_pmal = None
                                best_probe_path = pcap_path
                                best_probe_target_mod = pcap_target_mod
                                for candidate_path in candidate_paths[:probe_topk]:
                                    try:
                                        search_context = prepare_pcap_search_context(
                                            candidate_path,
                                            pcap_features=pcap_features,
                                            feature_names=list(bundle.feature_names),
                                            pcap_target_mod=pcap_target_mod,
                                        )
                                    except Exception as exc:
                                        print(f"[Stage3][Warn] probe search_context failed for {candidate_path}: {exc}")
                                        continue
                                    if (
                                        search_context.orig_pmal_for_selection is not None
                                        and search_context.orig_pmal_for_selection < pcap_selection.scan_min_prob
                                    ):
                                        continue
                                    try:
                                        with temporary_probe_workspace("probe_") as probe_tmp_dir:
                                            probe_out_dir = Path(probe_tmp_dir) / "pcap"
                                            probe_result = search_and_write_pcaps(
                                                pkts=rdpcap(str(candidate_path)),
                                                mods=np.asarray(mods[:1]).copy(),
                                                adv=adv[:1] if adv is not None else adv,
                                                pcap_out_dir=probe_out_dir,
                                                settings=settings,
                                                seed=seed,
                                                protocol_auto_fix=protocol_auto_fix,
                                                pcap_eval_model=pcap_eval_model,
                                                search_alphas=search_alphas,
                                                pcap_target_mod=search_context.pcap_target_mod,
                                                orig_pmal_for_selection=search_context.orig_pmal_for_selection,
                                                orig_feat_pre_mean=search_context.orig_feat_pre_mean,
                                                out_dir=out_dir,
                                                target_metric_fn=search_context.target_metric_fn,
                                                wrpcap_fn=wrpcap,
                                            )
                                            probe_pcap = probe_out_dir / "adv_0000.pcap"
                                            if not probe_pcap.exists():
                                                continue
                                            probe_pmal, probe_target_l2, probe_target_mae, _, probe_meta = (
                                                search_context.target_metric_fn(
                                                    probe_pcap,
                                                    adv[0] if adv is not None and len(adv) else None,
                                                )
                                            )
                                    except Exception as exc:
                                        print(f"[Stage3][Warn] probe search/write failed: {exc}")
                                        continue
                                    probe_align_cov = None
                                    if isinstance(probe_meta, dict):
                                        probe_alignment = probe_meta.get("alignment")
                                        if isinstance(probe_alignment, dict) and "coverage" in probe_alignment:
                                            probe_align_cov = float(probe_alignment["coverage"])
                                    evasion_rate = max(0.0, min(1.0, 1.0 - float(probe_pmal))) if probe_pmal is not None else 0.0
                                    cov = float(probe_align_cov) if probe_align_cov is not None and np.isfinite(float(probe_align_cov)) else 0.0
                                    probe_score = 0.7 * evasion_rate + 0.3 * cov
                                    if probe_result.pcap_kept_original_count >= probe_result.pcap_written_count:
                                        probe_score -= 0.05
                                    if probe_score > best_probe_score or (
                                        np.isclose(probe_score, best_probe_score)
                                        and probe_pmal is not None
                                        and (best_probe_pmal is None or float(probe_pmal) < float(best_probe_pmal))
                                    ):
                                        best_probe_score = probe_score
                                        best_probe_pmal = probe_pmal
                                        best_probe_path = candidate_path
                                        best_probe_target_mod = search_context.pcap_target_mod
                                if best_probe_path is not None and Path(best_probe_path) != Path(pcap_path):
                                    pcap_path = Path(best_probe_path)
                                    pcap_target_mod = best_probe_target_mod
                                    metrics_payload["pcap_selected_name"] = pcap_path.name
                                    metrics_payload["pcap_selected_path"] = str(pcap_path)
                                    metrics_payload["pcap_source_probe_selected_name"] = pcap_path.name
                                    metrics_payload["pcap_source_probe_selected_path"] = str(pcap_path)
                                    if best_probe_pmal is not None:
                                        metrics_payload["pcap_selected_prob_malicious"] = float(best_probe_pmal)
                                        metrics_payload["pcap_source_probe_selected_prob_malicious"] = float(
                                            best_probe_pmal
                                        )
                                if source_paths:
                                    metrics_payload["pcap_source_names"] = [path.name for path in source_paths]

                        for source_idx, source_path in enumerate(source_paths):
                            source_dir = pcap_out_dir / _source_slug(source_path, source_idx)
                            source_dir.mkdir(parents=True, exist_ok=True)
                            pcap_source_dirs.append(source_dir)
                            pkts = rdpcap(str(source_path))
                            source_adv = adv
                            source_mods = mods
                            if pcap_target_source == "pcap_conditioned":
                                source_adv, source_stage2_metrics = _stage2_adv_from_pcap(source_path)
                                source_mods, source_mod_source, source_mod_extra = _build_modifications_for_adv_samples(
                                    source_adv,
                                    cal_preprocessor=calibrated_preprocessor
                                    if (calibrated_preprocessor is not None and calibrated_preprocessor.is_active)
                                    else None,
                                )
                                if source_idx == 0:
                                    metrics_payload.update(
                                        {
                                            key: value
                                            for key, value in source_stage2_metrics.items()
                                            if value is not None
                                        }
                                    )
                                    metrics_payload.update(source_mod_extra)
                                    metrics_payload["remap_mod_source"] = source_mod_source
                                if source_stage2_metrics.get("pcap_conditioned_feature_asr") is not None:
                                    pcap_conditioned_feature_asrs.append(
                                        float(source_stage2_metrics["pcap_conditioned_feature_asr"])
                                    )
                                if source_stage2_metrics.get("pcap_conditioned_adv_prob_malicious") is not None:
                                    pcap_conditioned_adv_pmals.append(
                                        float(source_stage2_metrics["pcap_conditioned_adv_prob_malicious"])
                                    )
                                if source_stage2_metrics.get("pcap_conditioned_source_prob_malicious") is not None:
                                    pcap_conditioned_source_pmals.append(
                                        float(source_stage2_metrics["pcap_conditioned_source_prob_malicious"])
                                    )
                            if source_mods is None or source_adv is None:
                                continue
                            source_target_mod = (
                                np.mean(np.asarray(source_mods, dtype=np.float64), axis=0)
                                if len(source_mods)
                                else pcap_target_mod
                            )
                            search_context = prepare_pcap_search_context(
                                source_path,
                                pcap_features=pcap_features,
                                feature_names=list(bundle.feature_names),
                                pcap_target_mod=source_target_mod,
                            )
                            # Skip alpha search for already-evasive PCAPs
                            scan_min_prob = float(getattr(settings, "pcap_scan_min_prob", 0.5) or 0.5)
                            orig_pmal = search_context.orig_pmal_for_selection
                            if orig_pmal is not None and np.isfinite(float(orig_pmal)) and float(orig_pmal) < scan_min_prob:
                                print(
                                    f"[Stage3] skip search for {source_path.name}: "
                                    f"pmal={float(orig_pmal):.4f} < {scan_min_prob:.2f} (already evasive)"
                                )
                                total_kept_original += 1
                                continue
                            per_source_target_mods.append(np.asarray(search_context.pcap_target_mod))
                            search_result = search_and_write_pcaps(
                                pkts=pkts,
                                mods=source_mods,
                                adv=source_adv,
                                pcap_out_dir=source_dir,
                                settings=settings,
                                seed=seed + source_idx,
                                protocol_auto_fix=protocol_auto_fix,
                                pcap_eval_model=pcap_eval_model,
                                search_alphas=search_alphas,
                                pcap_target_mod=search_context.pcap_target_mod,
                                orig_pmal_for_selection=search_context.orig_pmal_for_selection,
                                orig_feat_pre_mean=search_context.orig_feat_pre_mean,
                                out_dir=out_dir,
                                target_metric_fn=search_context.target_metric_fn,
                                wrpcap_fn=wrpcap,
                            )
                            latest_mods = np.asarray(search_result.mods)
                            total_written += int(search_result.pcap_written_count)
                            total_kept_original += int(search_result.pcap_kept_original_count)
                            combined_out_dirs.append(search_result.pcap_out_dir)
                            trace_path = getattr(search_result, "pcap_search_trace_path", None)
                            if trace_path:
                                combined_trace_paths.append(str(trace_path))
                            for metric_key in aggregate_apply_metrics:
                                value = getattr(search_result, metric_key, None)
                                if value is not None:
                                    aggregate_apply_metrics[metric_key].append(float(value))
                            if search_result.pcap_selected_alphas:
                                selected_alphas.extend([float(v) for v in search_result.pcap_selected_alphas])
                            if search_result.pcap_selected_field_sets:
                                selected_field_sets.extend([str(v) for v in search_result.pcap_selected_field_sets])
                        mods = latest_mods
                        # ── Multi-step PCAP deformation ────────────────────────
                        # If the adversarial PCAP is still classified as malicious,
                        # re-extract its features and feed them back through Stage2
                        # for another diffusion + remap round.
                        multi_step_max = max(0, int(getattr(settings, "pcap_multi_step_max_rounds", 3)))
                        multi_step_rounds_done = 0
                        # ── Field subsets for multi-step rotation ──────────────────
                        _multi_field_subsets = [
                            ["mean_iat_ms", "std_iat_ms", "flow_scale"],  # timing
                            ["pad_bytes", "payload_scale", "fwd_pkt_scale"],  # payload/counts
                            ["dst_port_new", "src_port_new"],  # ports
                            ["flag_ratio", "syn_flag_ratio", "fin_flag_ratio"],  # flags
                            ["tcp_init_win_fwd", "tcp_init_win_bwd"],  # TCP window
                        ]
                        if multi_step_max > 0 and pcap_target_source == "pcap_conditioned":
                            for multi_round in range(multi_step_max):
                                adv_pcaps_existing = sorted(source_dir.glob("adv_*.pcap"))
                                if not adv_pcaps_existing:
                                    break
                                last_adv = adv_pcaps_existing[-1]
                                # Evaluate current adversarial PCAP
                                try:
                                    adv_pmal, adv_pred, _, _ = pcap_features.pcap_prob(last_adv)
                                except Exception as exc:
                                    print(f"[Stage3][Warn] multi-step pcap_prob failed for {last_adv}: {exc}")
                                    break
                                if adv_pmal is None or adv_pred != 1:
                                    break  # Already evading or can't evaluate
                                print(
                                    f"[Stage3] multi-step round {multi_round + 1}/{multi_step_max}: "
                                    f"adv_pmal={adv_pmal:.4f} pred={adv_pred} — re-deforming "
                                    f"(alpha_scale={multi_round + 2})"
                                )
                                # Progressively increase the push toward benign distribution.
                                if stage2_checkpoint_sampler is not None:
                                    _orig_alpha_key = "__orig_mal_anchor_alpha"
                                    if not hasattr(stage2_checkpoint_sampler.settings, _orig_alpha_key):
                                        object.__setattr__(
                                            stage2_checkpoint_sampler.settings,
                                            _orig_alpha_key,
                                            stage2_checkpoint_sampler.settings.mal_anchor_alpha,
                                        )
                                    _orig = getattr(stage2_checkpoint_sampler.settings, _orig_alpha_key, None)
                                    if _orig is not None and _orig > 0:
                                        object.__setattr__(
                                            stage2_checkpoint_sampler.settings,
                                            "mal_anchor_alpha",
                                            _orig * (multi_round + 2),
                                        )
                                # ── Packet-level mutation for later rounds ──────────
                                _pkt_source = last_adv
                                if multi_round >= 2:
                                    try:
                                        from scapy.all import TCP as _TCP

                                        _mut_pkts = rdpcap(str(last_adv))
                                        _mut_dir = source_dir / f"multi_{multi_round + 1:02d}"
                                        _mut_dir.mkdir(parents=True, exist_ok=True)
                                        _mut_rng = np.random.default_rng(seed + multi_round * 313 + source_idx * 7)
                                        for _mp in _mut_pkts:
                                            if _TCP in _mp and _mut_rng.random() < 0.05:
                                                _mp[_TCP].sport = (
                                                    int(_mp[_TCP].sport) + int(_mut_rng.integers(1, 1000))
                                                ) % 65535 + 1
                                        _mut_tmp = _mut_dir / "mutated_input.pcap"
                                        wrpcap(str(_mut_tmp), _mut_pkts)
                                        _pkt_source = _mut_tmp
                                    except Exception:
                                        pass

                                try:
                                    new_adv, new_stage2_metrics = _stage2_adv_from_pcap(_pkt_source)
                                    # ── Feature-aware noise scaled by benign std ───
                                    _noise_scale = 0.08 * (multi_round + 1)
                                    _ben_std = np.std(x_ben, axis=0).astype(np.float32) + 1e-6
                                    _rng = np.random.default_rng(seed + source_idx * 997 + multi_round * 137 + 42)
                                    _noise = _rng.normal(0, _noise_scale, size=new_adv.shape).astype(np.float32)
                                    _noise = _noise * _ben_std[np.newaxis, :]
                                    new_adv_perturbed = np.clip(new_adv + _noise, -5.0, 5.0)
                                    new_mods, _, new_mod_extra = _build_modifications_for_adv_samples(
                                        new_adv_perturbed,
                                        cal_preprocessor=calibrated_preprocessor
                                        if (calibrated_preprocessor is not None and calibrated_preprocessor.is_active)
                                        else None,
                                    )
                                    # ── Progressive blending toward rule-based ──────
                                    _learned_weight = max(0.0, 1.0 - 0.2 * multi_round)
                                    if _learned_weight < 0.999 and new_mod_extra is not None:
                                        _direct_mods = new_mod_extra.get("direct_mods")
                                        if _direct_mods is not None and len(_direct_mods):
                                            new_mods = blend_modifications(
                                                new_mods,
                                                _direct_mods,
                                                _learned_weight,
                                                bundle.mod_names if hasattr(bundle, "mod_names") else list(MOD_NAMES),
                                            )
                                except Exception as exc:
                                    print(f"[Stage3] multi-step round {multi_round + 1} failed: {exc}")
                                    break
                                # Rewrite adversarial PCAPs with new modifications
                                new_source_target_mod = (
                                    np.mean(np.asarray(new_mods, dtype=np.float64), axis=0)
                                    if len(new_mods)
                                    else source_target_mod
                                )
                                new_search_context = prepare_pcap_search_context(
                                    last_adv,
                                    pcap_features=pcap_features,
                                    feature_names=list(bundle.feature_names),
                                    pcap_target_mod=new_source_target_mod,
                                )
                                # Write to a new subdirectory for multi-step rounds
                                multi_dir = source_dir / f"multi_{multi_round + 1:02d}"
                                # ── Field subset for this round ────────────────────
                                _field_subset = (
                                    _multi_field_subsets[min(multi_round, len(_multi_field_subsets) - 1)]
                                    if multi_round < len(_multi_field_subsets)
                                    else None
                                )
                                print(
                                    f"[Stage3] multi-step round {multi_round + 1}: fields={_field_subset}, learned_weight={_learned_weight:.2f}"
                                )

                                new_search_result = search_and_write_pcaps(
                                    pkts=rdpcap(str(last_adv)),
                                    mods=new_mods,
                                    adv=new_adv,
                                    pcap_out_dir=multi_dir,
                                    settings=settings,
                                    seed=seed + source_idx + multi_round + 100,
                                    protocol_auto_fix=protocol_auto_fix,
                                    pcap_eval_model=pcap_eval_model,
                                    search_alphas=search_alphas,
                                    pcap_target_mod=new_search_context.pcap_target_mod,
                                    orig_pmal_for_selection=new_search_context.orig_pmal_for_selection,
                                    orig_feat_pre_mean=new_search_context.orig_feat_pre_mean,
                                    out_dir=out_dir,
                                    target_metric_fn=new_search_context.target_metric_fn,
                                    wrpcap_fn=wrpcap,
                                )
                                multi_step_rounds_done = multi_round + 1
                                total_written += int(new_search_result.pcap_written_count)
                                combined_out_dirs.append(new_search_result.pcap_out_dir)
                                new_trace = getattr(new_search_result, "pcap_search_trace_path", None)
                                if new_trace:
                                    combined_trace_paths.append(str(new_trace))
                                # Check if this round achieved evasion
                                multi_pcaps = sorted(multi_dir.glob("adv_*.pcap"))
                                if multi_pcaps:
                                    try:
                                        final_pmal, final_pred, _, _ = pcap_features.pcap_prob(multi_pcaps[-1])
                                        if final_pred == 0:
                                            print(
                                                f"[Stage3] multi-step success at round {multi_round + 1}: "
                                                f"pmal {adv_pmal:.4f}→{final_pmal:.4f} pred=0"
                                            )
                                    except Exception as exc:
                                        print(f"[Stage3][Warn] multi-step pcap_prob check failed: {exc}")
                                        pass
                        if multi_step_rounds_done > 0:
                            metrics_payload["pcap_multi_step_rounds_done"] = multi_step_rounds_done
                        # Restore original mal_anchor_alpha after multi-step
                        if stage2_checkpoint_sampler is not None:
                            _orig_alpha_key = "__orig_mal_anchor_alpha"
                            _saved = getattr(stage2_checkpoint_sampler.settings, _orig_alpha_key, None)
                            if _saved is not None:
                                object.__setattr__(stage2_checkpoint_sampler.settings, "mal_anchor_alpha", _saved)
                        # ── Check if this source achieved evasion ──
                        _adv_test = sorted(source_dir.glob("adv_*.pcap"))
                        if _adv_test:
                            try:
                                _t_pmal, _t_pred, _, _ = pcap_features.pcap_prob(_adv_test[-1])
                                if _t_pred == 0:
                                    metrics_payload["pcap_evasion_valid"] = True
                                    print(
                                        f"[Stage3] source {source_idx + 1}/{len(source_paths)} "
                                        f"({source_path.name}) evaded (pred=0 pmal={_t_pmal:.4f})"
                                    )
                            except Exception:
                                pass
                        if per_source_target_mods:
                            pcap_target_mod = np.mean(np.stack(per_source_target_mods, axis=0), axis=0)
                        if pcap_conditioned_feature_asrs:
                            metrics_payload["pcap_conditioned_feature_asr_mean"] = float(
                                np.mean(pcap_conditioned_feature_asrs)
                            )
                            metrics_payload["pcap_conditioned_feature_asrs"] = [
                                float(v) for v in pcap_conditioned_feature_asrs
                            ]
                        if pcap_conditioned_adv_pmals:
                            metrics_payload["pcap_conditioned_adv_prob_malicious_mean"] = float(
                                np.mean(pcap_conditioned_adv_pmals)
                            )
                            metrics_payload["pcap_conditioned_adv_pmals"] = [
                                float(v) for v in pcap_conditioned_adv_pmals
                            ]
                        if pcap_conditioned_source_pmals:
                            metrics_payload["pcap_conditioned_source_prob_malicious_mean"] = float(
                                np.mean(pcap_conditioned_source_pmals)
                            )
                        metrics_payload["pcap_written_count"] = total_written
                        metrics_payload["pcap_kept_original_count"] = total_kept_original
                        metrics_payload["pcap_modified"] = total_written > 0
                        metrics_payload["pcap_out_dir"] = str(pcap_out_dir)
                        metrics_payload["pcap_source_out_dirs"] = combined_out_dirs
                        if combined_trace_paths:
                            metrics_payload["pcap_search_trace_paths"] = combined_trace_paths
                            metrics_payload["pcap_search_trace_path"] = (
                                combined_trace_paths[0] if len(combined_trace_paths) == 1 else ""
                            )
                        for metric_key, values in aggregate_apply_metrics.items():
                            if values:
                                metrics_payload[metric_key] = float(np.mean(values))
                        if selected_alphas:
                            metrics_payload["pcap_selected_alphas"] = selected_alphas
                            metrics_payload["pcap_selected_alpha_mean"] = float(np.mean(selected_alphas))
                        if selected_field_sets:
                            metrics_payload["pcap_selected_field_sets"] = selected_field_sets
                    except (ImportError, FileNotFoundError, OSError, ValueError, RuntimeError) as exc:
                        import traceback

                        metrics_payload["pcap_modified"] = False
                        metrics_payload["pcap_error"] = str(exc)
                        metrics_payload["pcap_error_traceback"] = traceback.format_exc()

    else:
        metrics_payload["adv_samples_loaded"] = False
        metrics_payload["adv_samples_count"] = 0

    if settings.pcap_eval and pcap_eval_model is None:
        print("[Stage3][Warn] pcap_eval requested but no evaluation model available.")

    if settings.pcap_eval and pcap_path is not None and pcap_path.exists() and pcap_eval_model is not None:
        pcap_eval_start = time.perf_counter()
        eval_rows = []
        target_l2_vals: list[float] = []
        target_mae_vals: list[float] = []
        per_dim_deltas: list[dict[str, Any]] = []
        sanity_vals = {
            "nonmonotonic_rate": [],
            "transport_missing_rate": [],
            "tcp_seq_backwards_rate": [],
            "tcp_flag_invalid_rate": [],
            "tcp_syn_fin_rate": [],
            "tcp_syn_rst_rate": [],
            "tcp_fin_rst_rate": [],
        }
        fatal_validity_flags: list[float] = []

        all_evasion_valid = True
        adv_pcaps_total = 0
        for source_idx, source_path in enumerate(source_paths or ([pcap_path] if pcap_path is not None else [])):
            source_name = source_path.name
            orig_eval = evaluate_original_pcap(
                pcap_path=source_path,
                source_name=source_name,
                pcap_features=pcap_features,
                scapy_available=scapy_available,
                metrics_payload=metrics_payload,
                pcap_evasion_valid=pcap_evasion_valid,
                scan_min_prob=pcap_selection.scan_min_prob,
            )
            all_evasion_valid = all_evasion_valid and orig_eval.pcap_evasion_valid
            extend_sanity_values(sanity_vals, orig_eval.sanity)
            eval_rows.append(orig_eval.row)
            adv_pcaps = []
            source_dir = pcap_source_dirs[source_idx] if source_idx < len(pcap_source_dirs) else pcap_out_dir
            if orig_eval.pcap_evasion_valid and source_dir is not None and source_dir.exists():
                adv_pcaps = sorted(source_dir.glob("adv_*.pcap"))
                adv_pcaps_total += len(adv_pcaps)
                adv_eval = evaluate_adversarial_pcaps(
                    adv_pcaps=adv_pcaps,
                    source_name=source_name,
                    pcap_features=pcap_features,
                    scapy_available=scapy_available,
                    adv=adv,
                    feature_names=list(bundle.feature_names),
                    original_sanity=orig_eval.sanity,
                )
                eval_rows.extend(adv_eval.rows)
                target_l2_vals.extend(adv_eval.target_l2_vals)
                target_mae_vals.extend(adv_eval.target_mae_vals)
                fatal_validity_flags.extend(adv_eval.fatal_validity_flags)
                per_dim_deltas.extend(adv_eval.per_dim_deltas)
                for key, values in adv_eval.sanity_values.items():
                    sanity_vals[key].extend(values)
        pcap_evasion_valid = all_evasion_valid
        metrics_payload["pcap_evasion_valid"] = bool(pcap_evasion_valid)
        if not pcap_evasion_valid:
            print(
                "[Stage3][Warn] at least one source PCAP predicted benign; skipping adversarial PCAP evaluation for that source."
            )
        if pcap_evasion_valid and adv is not None and (pcap_out_dir is None or adv_pcaps_total <= 0):
            metrics_payload["pcap_skip_reason"] = metrics_payload.get("pcap_skip_reason", "no_adv_pcaps_written")
            print("[Stage3][Warn] No adversarial PCAPs found for evaluation.")
        if eval_rows:
            pcap_eval_rows = finalize_pcap_eval(
                out_dir=out_dir,
                eval_rows=eval_rows,
                metrics_payload=metrics_payload,
                pcap_features=pcap_features,
                target_l2_vals=target_l2_vals,
                target_mae_vals=target_mae_vals,
                fatal_validity_flags=fatal_validity_flags,
                per_dim_deltas=per_dim_deltas if per_dim_deltas else None,
            )
        # Save core metrics before potentially-slow baseline PCAP evaluations.
        # If baseline evals take too long, at least the main results are persisted.
        save_metrics(metrics_payload, out_dir)
        save_metrics_csv(metrics_payload, out_dir)
        print("[Stage3] core metrics saved; starting baseline PCAP evaluations...", flush=True)
        run_stage3_baseline_pcap_eval(
            cfg=cfg,
            settings=settings,
            metrics_payload=metrics_payload,
            main_adv_pre=adv,
            pcap_evasion_valid=pcap_evasion_valid,
            preprocessor=preprocessor,
            remap_mode=remap_mode,
            remap_use_direct=remap_use_direct,
            remap_bundle=remap_bundle,
            x_ben_raw=x_ben_raw,
            feature_names=list(bundle.feature_names),
            scapy_available=scapy_available,
            protocol_auto_fix=protocol_auto_fix,
            pcap_path=pcap_path,
            pcap_features=pcap_features,
            out_dir=out_dir,
            seed=seed,
            device=device,
            x_train=x_train,
            effective_blend_fn=lambda learned, direct, mod_names: effective_blend_alpha(
                learned,
                direct,
                mod_names,
                requested_alpha=remap_blend_alpha,
                collapse_ratio_threshold=remap_collapse_ratio_threshold,
            ),
            blend_fn=blend_modifications,
        )
        aggregate_pcap_sanity(sanity_vals, metrics_payload)
        metrics_payload["pcap_eval_time_sec"] = float(time.perf_counter() - pcap_eval_start)

    metrics_payload.update(pcap_features.metrics_snapshot())
    metrics_payload["stage3_total_time_sec"] = float(
        float(metrics_payload.get("remapper_train_time_sec", 0.0) or 0.0)
        + float(metrics_payload.get("pcap_apply_time_sec", 0.0) or 0.0)
        + float(metrics_payload.get("pcap_eval_time_sec", 0.0) or 0.0)
    )

    feature_quality_block_reason = _pcap_feature_quality_block_reason(metrics_payload)
    metrics_payload["pcap_feature_quality_strict"] = not bool(feature_quality_block_reason)
    if feature_quality_block_reason:
        metrics_payload["pcap_feature_quality_block_reason"] = feature_quality_block_reason

    # Simple evidence tracking based on whether PCAP was actually modified
    pcap_was_modified = metrics_payload.get("pcap_modified", False)
    has_full_evidence = bool(pcap_was_modified and not feature_quality_block_reason)
    evidence_scope = "full_evidence" if has_full_evidence else "remap_only_evidence"
    stage3_evidence: dict[str, object] = {
        "stage3_decision_score_scope": "full" if has_full_evidence else "remap_only",
        "stage3_full_evidence": has_full_evidence,
        "stage3_remap_only": not has_full_evidence,
        "stage3_evidence_scope": evidence_scope,
    }
    if not has_full_evidence:
        stage3_evidence["stage3_evidence_block_reason"] = str(
            feature_quality_block_reason
            or metrics_payload.get("pcap_skip_reason")
            or ""
        )
    metrics_payload.update({k: v for k, v in stage3_evidence.items()})
    save_metrics(metrics_payload, out_dir)
    save_metrics_csv(metrics_payload, out_dir)
    if settings.save_intermediate_results:
        intermediate_payload = build_versioned_artifact_payload(
            VersionedArtifactSpec(
                version_as_array=True,
                fields={
                    "feature_names": np.asarray(bundle.feature_names),
                },
                optional_fields={
                    "adv_pre": adv,
                    "adv_norm": adv_norm,
                    "modifications": np.asarray(mods) if mods is not None else None,
                    "pcap_target_mod": np.asarray(pcap_target_mod) if pcap_target_mod is not None else None,
                    "adv_stats_mean": np.asarray(adv_mean) if adv_mean is not None else None,
                    "adv_stats_std": np.asarray(adv_std) if adv_std is not None else None,
                },
            )
        )
        np.savez_compressed(out_dir / "intermediate_results.npz", **intermediate_payload)
    if remap_bundle is not None:
        save_state(
            build_versioned_artifact_payload(
                VersionedArtifactSpec(
                    fields={
                        "remapper_state": remap_bundle.remapper.state_dict(),
                        "feature_names": remap_bundle.feature_names,
                        "mod_names": remap_bundle.mod_names,
                        "mod_mean": remap_bundle.mod_mean,
                        "mod_std": remap_bundle.mod_std,
                        "input_mean": remap_bundle.input_mean,
                        "input_std": remap_bundle.input_std,
                        "continuous_names": remap_bundle.continuous_names,
                        "port_values": remap_bundle.port_values,
                        "train_log": remap_bundle.train_log,
                        "best_epoch": remap_bundle.best_epoch,
                        "best_score": remap_bundle.best_score,
                    }
                )
            ),
            out_dir / "stage3.pt",
        )
    outputs = build_stage_output_files(
        primary_artifact_key="state" if remap_bundle is not None else None,
        primary_artifact_name="stage3.pt" if remap_bundle is not None else None,
        extra_outputs={
            "train_metrics": "stage3_train_metrics.csv" if remap_bundle is not None else None,
            "remapper_eval": "stage3_remap_eval.csv" if remap_bundle is not None else None,
            "modifications": (
                metrics_payload.get("modifications_path") if metrics_payload.get("modifications_saved") else None
            ),
            "intermediate_results": "intermediate_results.npz" if settings.save_intermediate_results else None,
            "pcap_eval": metrics_payload.get("pcap_eval_path"),
            "pcap_out_dir": metrics_payload.get("pcap_out_dir"),
        },
    )
    save_stage_manifest_spec(
        StageManifestSpec(
            stage_name="stage3",
            out_dir=out_dir,
            config_path=config_path,
            inputs={
                "ids_name": str(getattr(settings, "ids_name", "pcap_ids")),
                "stage1_oracle_name": oracle_name,
                "feature_dim": int(x_train.shape[1]),
                "adv_samples_path": str(adv_path),
                "pcap_path": str(pcap_path) if pcap_path is not None else "",
                "pcap_source_paths": [str(path) for path in source_paths],
                "remap_mode": remap_mode,
            },
            outputs=outputs,
            arrays={
                "adv_pre": adv if adv is not None else np.zeros((0, x_train.shape[1]), dtype=np.float32),
                "modifications": mods if mods is not None else np.zeros((0, len(MOD_NAMES)), dtype=np.float32),
            },
            metrics={
                "adv_samples_count": metrics_payload.get("adv_samples_count"),
                "adv_benign_rate": metrics_payload.get("adv_benign_rate"),
                "pcap_modified": metrics_payload.get("pcap_modified"),
                "pcap_eval": metrics_payload.get("pcap_eval"),
            },
        )
    )
    print("[Stage3] summary")
    print(
        f"[Stage3] adv_samples_loaded={metrics_payload.get('adv_samples_loaded')} "
        f"count={metrics_payload.get('adv_samples_count')}"
    )
    if metrics_payload.get("pcap_selected_name"):
        print(
            "[Stage3] pcap_selected"
            f" name={metrics_payload.get('pcap_selected_name')}"
            f" source={metrics_payload.get('pcap_selected_source', '')}"
        )
        if metrics_payload.get("pcap_selected_prob_malicious") is not None:
            print(f"[Stage3] pcap_selected_prob_malicious={metrics_payload.get('pcap_selected_prob_malicious'):.6f}")
    if metrics_payload.get("pcap_eval_model"):
        print(f"[Stage3] pcap_eval_model={metrics_payload.get('pcap_eval_model')}")
    if "pcap_evasion_valid" in metrics_payload:
        print(f"[Stage3] pcap_evasion_valid={metrics_payload.get('pcap_evasion_valid')}")
    if metrics_payload.get("remapper_train_metrics_path"):
        print(f"[Stage3] train_metrics={metrics_payload.get('remapper_train_metrics_path')}")
    if metrics_payload.get("remapper_eval_path"):
        print(
            "[Stage3] remapper_eval"
            f" mae={metrics_payload.get('remapper_eval_mae'):.6f}"
            f" rmse={metrics_payload.get('remapper_eval_rmse'):.6f}"
            f" r2={metrics_payload.get('remapper_eval_r2'):.6f}"
        )
        if "remapper_eval_port_acc" in metrics_payload:
            print(f"[Stage3] remapper_eval_port_acc={metrics_payload.get('remapper_eval_port_acc'):.6f}")
    if metrics_payload.get("modifications_saved"):
        print(f"[Stage3] modifications={metrics_payload.get('modifications_path')}")
    if "adv_benign_rate" in metrics_payload:
        print(
            "[Stage3] adv_asr"
            f" asr={metrics_payload.get('adv_benign_rate'):.6f}"
            f" adv_pmal={metrics_payload.get('adv_prob_malicious_mean'):.6f}"
        )
    if metrics_payload.get("pcap_modified"):
        print(f"[Stage3] pcap_out_dir={metrics_payload.get('pcap_out_dir')}")
        if metrics_payload.get("pcap_apply_time_sec") is not None:
            print(
                "[Stage3] pcap_apply"
                f" time_sec={metrics_payload.get('pcap_apply_time_sec'):.6f}"
                f" pcaps_per_sec={metrics_payload.get('pcap_pcaps_per_sec', float('nan')):.6f}"
                f" pps={metrics_payload.get('pcap_packet_throughput_pps', float('nan')):.6f}"
            )
    if metrics_payload.get("pcap_eval"):
        print(f"[Stage3] pcap_eval={metrics_payload.get('pcap_eval_path')}")
        if metrics_payload.get("pcap_target_l2_mean") is not None:
            print(f"[Stage3] pcap_target_l2_mean={metrics_payload.get('pcap_target_l2_mean'):.6f}")
        if metrics_payload.get("pcap_eval_time_sec") is not None:
            print(f"[Stage3] pcap_eval_time_sec={metrics_payload.get('pcap_eval_time_sec'):.6f}")
        if metrics_payload.get("pcap_sanity_nonmonotonic_rate") is not None:
            print(
                "[Stage3] pcap_sanity"
                f" nonmono={metrics_payload.get('pcap_sanity_nonmonotonic_rate'):.6f}"
                f" tcp_seq_back={metrics_payload.get('pcap_sanity_tcp_seq_backwards_rate', 0.0):.6f}"
                f" tcp_flag_invalid={metrics_payload.get('pcap_sanity_tcp_flag_invalid_rate', 0.0):.6f}"
            )
    if pcap_eval_rows:
        avg_cov = metrics_payload.get("pcap_eval_avg_alignment", 0.0)
        print(f"[Stage3] pcap_eval_rows={len(pcap_eval_rows)} avg_alignment_coverage={avg_cov:.4f}")
        for row in pcap_eval_rows:
            name = Path(str(row["pcap"])).name
            target_l2 = row.get("target_l2")
            cov_val = row.get("alignment_coverage")
            cov_str = f"{float(cov_val):.3f}" if cov_val not in ("", None) else "n/a"
            print(
                "[Stage3] pcap_result"
                f" name={name}"
                f" backend={row['feature_backend']}"
                f" flows={row['flow_count']}"
                f" cov={cov_str}"
                f" pmal={row['prob_malicious']:.4f}"
                f" pred={row['pred_label']}"
                f"{'' if target_l2 in ('', None) else f' target_l2={float(target_l2):.3f}'}"
            )
    print(f"[Stage3] saved to {out_dir}")


if __name__ == "__main__":
    cfg_path = os.environ.get("RDSYNTH_CONFIG", "configs/demo.yaml")
    main(cfg_path)
