from __future__ import annotations

import json
from pathlib import Path

from rdsynth.pipeline.data import load_data_context
from rdsynth.pipeline.preprocessing import DatasetPreprocessor
from rdsynth.pipeline.runtime import load_stage_runtime
from rdsynth.pipeline.stage3_ids import train_stage3_ids
from rdsynth.pipeline.stage3_inputs import load_adv_samples, resolve_adv_samples_path
from rdsynth.pipeline.stage3_ops import (
    Stage3Settings,
    detect_stage3_environment,
    load_stage3_artifacts,
    resolve_pcap_eval_model,
)
from rdsynth.pipeline.stage3_pcap import PcapFeatureExtractor
from rdsynth.pipeline.stage3_pcap_eval import (
    aggregate_pcap_sanity,
    evaluate_adversarial_pcaps,
    evaluate_original_pcap,
    extend_sanity_values,
    finalize_pcap_eval,
)
from rdsynth.utils.artifacts import save_metrics, save_metrics_csv
from rdsynth.utils.feature_align import build_statistical_feature_aliases, load_feature_aliases


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_stage3_paths(
    out_dir: Path, settings: Stage3Settings, metrics_payload: dict[str, object]
) -> tuple[Path | None, Path | None]:
    manifest = _load_json(out_dir / "manifest.json")
    manifest_inputs = manifest.get("inputs") if isinstance(manifest, dict) else {}
    pcap_path_text = str(manifest_inputs.get("pcap_path", "") or settings.pcap_path or "").strip()
    pcap_path = Path(pcap_path_text) if pcap_path_text else None

    pcap_out_dir_text = str(metrics_payload.get("pcap_out_dir", "") or settings.pcap_out_dir or "").strip()
    pcap_out_dir = Path(pcap_out_dir_text) if pcap_out_dir_text else (out_dir / "pcap")
    return pcap_path, pcap_out_dir


def _resolve_pcap_ids_training_pool(prev_metrics: dict[str, object], fallback_pcap: Path) -> list[Path]:
    raw_pool = prev_metrics.get("pcap_ids_malicious_pcaps_configured") or prev_metrics.get(
        "pcap_ids_malicious_pcaps_used"
    )
    paths: list[Path] = []
    if isinstance(raw_pool, list):
        for item in raw_pool:
            path = Path(str(item))
            if path.exists() and path.is_file():
                paths.append(path)
    return paths or [fallback_pcap]


def run_stage3_pcap_eval_only(config_path: str | Path) -> None:
    runtime = load_stage_runtime(config_path, "stage3")
    cfg = runtime.cfg
    seed = runtime.seed
    device = runtime.device
    out_dir = runtime.out_dir
    settings = Stage3Settings.from_cfg(runtime.stage_cfg, cfg["stage2"])
    prev_metrics = _load_json(out_dir / "metrics.json")

    env = detect_stage3_environment()
    data_ctx = load_data_context(cfg, seed)
    bundle = data_ctx.bundle
    preprocessor = DatasetPreprocessor.from_bundle(bundle)
    artifacts = load_stage3_artifacts(
        cfg=cfg,
        oracle_name=settings.oracle_name,
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
    pcap_path, pcap_out_dir = _resolve_stage3_paths(out_dir, settings, prev_metrics)
    if pcap_path is None or not pcap_path.exists():
        raise FileNotFoundError(f"Stage3 PCAP-only evaluation could not resolve original pcap path: {pcap_path}")
    model_selection = resolve_pcap_eval_model(
        ids=None,
        oracle=oracle,
        surrogate=surrogate,
        prefer_ids=False,
        prefer_oracle=bool(getattr(settings, "pcap_eval_use_oracle", False)),
    )
    pcap_eval_model = model_selection.pcap_eval_model
    pcap_eval_model_name = model_selection.pcap_eval_model_name
    alias_map = build_statistical_feature_aliases(
        bundle.feature_names,
        dataset_name=str(cfg.get("data", {}).get("dataset", "")),
        base_alias_map=load_feature_aliases(settings.feature_aliases_path),
    )
    ids_bundle = None
    if bool(getattr(settings, "pcap_eval_use_ids", False)):
        malicious_pcaps = _resolve_pcap_ids_training_pool(prev_metrics, pcap_path)
        ids_training = train_stage3_ids(
            malicious_pcap=pcap_path,
            malicious_pcaps=malicious_pcaps,
            settings=settings,
            feature_names=list(bundle.feature_names),
            raw_feature_mean=preprocessor.feature_mean(bundle.x_train),
            alias_map=alias_map,
            preprocessor=preprocessor,
            device=device,
            seed=seed,
        )
        prev_metrics.update(ids_training.metrics)
        ids_bundle = ids_training.ids_bundle
        model_selection = resolve_pcap_eval_model(
            ids=ids_bundle,
            oracle=oracle,
            surrogate=surrogate,
            prefer_ids=ids_bundle is not None,
            prefer_oracle=bool(getattr(settings, "pcap_eval_use_oracle", False)),
        )
        pcap_eval_model = model_selection.pcap_eval_model
        pcap_eval_model_name = model_selection.pcap_eval_model_name
    if pcap_eval_model is None:
        raise RuntimeError("No PCAP evaluation model available for Stage3 PCAP-only evaluation.")

    pcap_features = PcapFeatureExtractor(
        feature_backend=settings.feature_backend,
        feature_names=list(bundle.feature_names),
        raw_feature_mean=preprocessor.feature_mean(bundle.x_train),
        alias_map=alias_map,
        align_min_cov=settings.pcap_align_min_coverage,
        scapy_available=env.scapy_available,
        nfstream_available=env.nfstream_available,
        cicflowmeter_available=env.cicflowmeter_available,
        cicflowmeter_cmd=settings.cicflowmeter_cmd,
        cicflowmeter_timeout=settings.cicflowmeter_timeout,
        fail_closed=settings.pcap_feature_fail_closed,
        fail_on_partial_alignment=settings.pcap_feature_fail_on_partial_alignment,
        preprocessor=preprocessor,
        pcap_eval_model=pcap_eval_model,
        pcap_eval_model_name=pcap_eval_model_name,
        ids=ids_bundle,
        oracle=oracle,
        surrogate=surrogate,
        pcap_eval_batch_size=settings.pcap_eval_batch_size,
        seed=seed,
        device=device,
        max_pcap_bytes=int(getattr(settings, "pcap_scan_max_bytes", 0) or 0),
        max_flows_per_pcap=getattr(settings, "pcap_feature_max_flows_per_pcap", None),
        cache_enable=settings.pcap_cache_enable,
        cache_dir=(
            Path(settings.pcap_cache_dir).resolve() if settings.pcap_cache_enable and settings.pcap_cache_dir else None
        ),
    )

    metrics_payload = dict(prev_metrics)
    metrics_payload["pcap_eval_model"] = pcap_eval_model_name
    metrics_payload["pcap_eval_only"] = True

    adv_path = resolve_adv_samples_path(settings.adv_samples_path, cfg["project"]["out_dir"])
    loaded_adv = load_adv_samples(
        adv_path,
        project_out_dir=cfg["project"]["out_dir"],
        current_feature_names=bundle.feature_names,
        expected_feature_dim=bundle.x_train.shape[1],
        copy_to=None,
    )
    adv = loaded_adv.adv
    metrics_payload["adv_samples_loaded"] = loaded_adv.loaded
    metrics_payload["adv_samples_count"] = loaded_adv.count

    eval_rows = []
    target_l2_vals: list[float] = []
    target_mae_vals: list[float] = []
    fatal_validity_flags: list[float] = []
    sanity_vals = {
        "nonmonotonic_rate": [],
        "transport_missing_rate": [],
        "tcp_seq_backwards_rate": [],
        "tcp_flag_invalid_rate": [],
        "tcp_syn_fin_rate": [],
        "tcp_syn_rst_rate": [],
        "tcp_fin_rst_rate": [],
    }

    orig_eval = evaluate_original_pcap(
        pcap_path=pcap_path,
        source_name=pcap_path.name,
        pcap_features=pcap_features,
        scapy_available=env.scapy_available,
        metrics_payload=metrics_payload,
        pcap_evasion_valid=prev_metrics.get("pcap_evasion_valid"),
        scan_min_prob=float(prev_metrics.get("pcap_scan_min_prob", settings.pcap_scan_min_prob)),
    )
    extend_sanity_values(sanity_vals, orig_eval.sanity)
    eval_rows.append(orig_eval.row)

    adv_pcaps = sorted(pcap_out_dir.glob("adv_*.pcap")) if pcap_out_dir.exists() else []
    if adv_pcaps:
        adv_eval = evaluate_adversarial_pcaps(
            adv_pcaps=adv_pcaps,
            source_name=pcap_path.name,
            pcap_features=pcap_features,
            scapy_available=env.scapy_available,
            adv=adv,
            feature_names=list(bundle.feature_names),
            original_sanity=orig_eval.sanity,
        )
        eval_rows.extend(adv_eval.rows)
        target_l2_vals.extend(adv_eval.target_l2_vals)
        target_mae_vals.extend(adv_eval.target_mae_vals)
        fatal_validity_flags.extend(adv_eval.fatal_validity_flags)
        for key, values in adv_eval.sanity_values.items():
            sanity_vals[key].extend(values)

    if eval_rows:
        finalize_pcap_eval(
            out_dir=out_dir,
            eval_rows=eval_rows,
            metrics_payload=metrics_payload,
            pcap_features=pcap_features,
            target_l2_vals=target_l2_vals,
            target_mae_vals=target_mae_vals,
            fatal_validity_flags=fatal_validity_flags,
        )
    aggregate_pcap_sanity(sanity_vals, metrics_payload)
    metrics_payload.update(pcap_features.metrics_snapshot())
    save_metrics(metrics_payload, out_dir)
    save_metrics_csv(metrics_payload, out_dir)
    print(f"[Stage3PcapEvalOnly] pcap={pcap_path}")
    print(f"[Stage3PcapEvalOnly] adv_pcaps={len(adv_pcaps)}")
    print(f"[Stage3PcapEvalOnly] metrics={out_dir / 'metrics.json'}")
