from __future__ import annotations

import hashlib
import platform
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from rdsynth.utils.config import require_mapping

_PROJECT_DEFAULTS: dict[str, Any] = {
    "device": "auto",
    "deterministic": True,
    "strict_repro": False,
    "allow_tf32": True,
    "matmul_precision": "high",
    "allow_unsafe_checkpoint_load": False,
    "num_threads": None,
    "num_interop_threads": None,
    "stage_timeout_sec": None,
    "stage1_shared_root": "",
}

_SPEC_OUTPUT_BUCKETS = {
    "paper_main",
    "reviewer_suite",
    "ablations",
    "stress_tests",
    "debug",
    "cache",
    "failed",
    "figures",
    "tables",
    "reports",
}


_DATA_DEFAULTS: dict[str, Any] = {
    "strict_ingest": False,
    "encoding_errors": "replace",
    "pcap_malicious_backend": "auto",
    "pcap_malicious_glob": "*.pcap",
}


_STAGE1_DEFAULTS: dict[str, Any] = {
    "query_strategy": "entropy",
    "query_pool": 4,
    "query_mix_ratio": 0.5,
    "query_real_ratio": 1.0,
    "query_balance": True,
    "query_label_noise": 0.0,
    "extraction_rounds": 1,
    "real_warmup_steps": 200,
    "calibration_bins": 10,
    "save_eval_snapshot": True,
    "data_quality": {"enable": True},
}

_STAGE2_DEFAULTS: dict[str, Any] = {
    "generator_backbone": "ddpm",
    "surrogate_guidance_mode": "embedding",
    "gan_noise_dim": 64,
    "gan_critic_steps": 5,
    "gan_weight_clip": 0.01,
    "eval_metrics": True,
    "save_samples": True,
    "save_intermediate_results": True,
    "require_stage1": True,
    "constraints": {"enable": True},
    "post_clip_norm_range": True,
    "lambda_preserve": 0.1,
    "mal_anchor_alpha": 0.1,
    "selection_eval_every": 5,
    "selection_eval_samples": 256,
    "selection_batch_size": 256,
    "selection_mal_anchor_alpha": 0.1,
    "deployable_constraints": {
        "enable": True,
        "port_policy": "keep",
        "flag_policy": "clip",
        "temporal_policy": "clip_benign",
        "port_allowlist": [],
    },
    "pareto_eval": {
        "enable": True,
        "anchor_grid": [0.0, 0.05, 0.1, 0.2, 0.3, 0.5],
        "max_samples": 1000,
        "selection": {
            "candidate_selection": "global_pareto",
            "iterative_rounds": 1,
            "iterative_points": 3,
            "iterative_radius_decay": 0.5,
            "prefer_oracle": True,
            "min_asr_oracle": 0.95,
            "min_asr_surrogate": 0.90,
            "max_adv_pmal_oracle": None,
            "max_adv_pmal_surrogate": None,
            "max_ffd": 50.0,
            "max_swd": None,
            "max_adv_to_ben_l2": None,
            "max_adv_to_mal_l2": None,
        },
    },
    "loss_schedule": {
        "enable": False,
        "fidelity_scale_start": 1.0,
        "fidelity_scale_end": 1.0,
        "attack_scale_start": 1.0,
        "attack_scale_end": 1.0,
    },
    "baselines": {
        "enable": True,
        "budget_scale": 5.0,
        "paper_budget_scale": 5.0,
        "methods": [
            "identity",
            "knn_benign",
            "random",
            "fgsm",
            "pgd",
            "idsgan_lite",
            "digfupas_lite",
            "gpmt_lite",
            "progen_lite",
            "amoeba_lite",
            "vulnergan_lite",
            "netdiffusion_lite",
        ],
    },
}

_STAGE3_DEFAULTS: dict[str, Any] = {
    "save_intermediate_results": True,
    "remap_train_source": "benign",
    "ids_name": "pcap_ids",
    # Dual evaluator: when true, reports both oracle and pcap_ids PCAP results.
    # When false (legacy), uses only one evaluator per run.
    "pcap_eval_dual": True,
    "pcap_eval_record_both": True,
    "pcap_apply_fields": [
        "mean_iat_ms",
        "std_iat_ms",
        "pad_bytes",
        "dst_port_new",
        "flag_ratio",
        "flow_scale",
        "payload_scale",
        "tcp_init_win_fwd",
        "tcp_init_win_bwd",
        "fwd_pkt_scale",
    ],
    "pcap_search_alphas": [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
    "pcap_search_probe_topk": 12,
    "pcap_search_rounds": 3,
    "pcap_search_bidirectional": True,
    "pcap_tcp_fixup": True,
    "pcap_dst_port_policy": "keep",
    "pcap_dst_port_allowlist": [],
    "feature_backend": "auto",
    "pcap_eval_use_ids": True,
    "pcap_eval_use_oracle": True,
    "pcap_compare_baselines": True,
    "pcap_scan_dir": "data/PCAPs/malicious",
    "pcap_scan_limit": 30,
    "pcap_scan_min_prob": 0.5,
    "pcap_scan_glob": "*.pcap",
    "pcap_scan_compare_existing": True,
    "pcap_scan_pmal_weight": 0.70,
    "pcap_scan_target_fit_weight": 0.20,
    "pcap_scan_target_mod_fit_weight": 0.10,
    "pcap_scan_max_bytes": 0,
    "pcap_feature_max_flows_per_pcap": 2048,
    "pcap_semantic_filter": True,
    "pcap_dataset": "",
    "pcap_attack_label": "",
    "pcap_attack_labels": [],
    "pcap_source_selection_mode": "best",
    "pcap_source_sample_n": 1,
    "pcap_source_sample_seed": 0,
    "pcap_target_source": "stage2_saved_samples",
    "pcap_feature_fail_closed": False,
    "pcap_feature_fail_on_partial_alignment": False,
    "pcap_search_field_subsets": True,
    "protocol_auto_fix": True,
    "remap_mode": "auto",
    "remap_min_r2": 0.0,
    "remap_blend_alpha": 0.70,
    "remap_collapse_ratio_threshold": 0.25,
    "pcap_ids_benign_path": "data/PCAPs/benign/benign.pcap",
    "pcap_ids_benign_paths": [],
    "pcap_ids_benign_dir": "",
    "pcap_ids_benign_glob": "*.pcap",
    "pcap_ids_benign_max_pcaps": 1,
    "pcap_ids_benign_max_flows_per_pcap": 2048,
    "pcap_ids_training_max_pcaps": 5,
    "pcap_ids_malicious_max_flows_per_pcap": 2048,
    "pcap_ids_feature_backend": "",
    "pcap_ids_model_type": "extra_trees",
    "pcap_ids_hidden_dims": [128, 128],
    "pcap_ids_epochs": 5,
    "pcap_ids_batch_size": 256,
    "pcap_ids_lr": 1.0e-3,
    "pcap_ids_max_batches_per_epoch": 50,
    "pcap_multi_step_max_rounds": 3,
    "pcap_mandatory_paths": [],
}


def _deep_setdefault(target: dict[str, Any], defaults: Mapping[str, Any]) -> None:
    for key, value in defaults.items():
        if key not in target:
            target[key] = deepcopy(value)
            continue
        if isinstance(target[key], dict) and isinstance(value, Mapping):
            _deep_setdefault(target[key], value)


def _optional_int(value: object, name: str, *, minimum: int | None = None) -> int | None:
    if value is None or value == "":
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Config key '{name}' must be an integer or null.") from exc
    if minimum is not None and parsed < minimum:
        raise ValueError(f"Config key '{name}' must be >= {minimum}.")
    return parsed


def _required_int(value: object, name: str, *, minimum: int | None = None) -> int:
    parsed = _optional_int(value, name, minimum=minimum)
    if parsed is None:
        raise ValueError(f"Config key '{name}' must not be empty.")
    return parsed


def _required_str(value: object, name: str) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        raise ValueError(f"Config key '{name}' must not be empty.")
    return text


def _config_sha256(config_path: Path) -> str:
    return hashlib.sha256(config_path.read_bytes()).hexdigest()


def _normalize_out_dir(raw_out_dir: str) -> tuple[str, str, bool]:
    path = Path(raw_out_dir)
    parts = path.parts
    if path.is_absolute():
        bucket = ""
        if "outputs" in parts:
            outputs_index = parts.index("outputs")
            if outputs_index + 1 < len(parts):
                candidate = parts[outputs_index + 1]
                if candidate in _SPEC_OUTPUT_BUCKETS:
                    bucket = candidate
        return raw_out_dir, bucket, bucket != ""

    if not parts:
        normalized = Path("outputs") / "debug"
        return normalized.as_posix(), "debug", False

    if parts[0] != "outputs":
        normalized = Path("outputs") / "debug" / path
        return normalized.as_posix(), "debug", False

    if len(parts) == 1:
        normalized = Path("outputs") / "debug"
        return normalized.as_posix(), "debug", False

    bucket = parts[1]
    if bucket in _SPEC_OUTPUT_BUCKETS:
        return path.as_posix(), bucket, True

    normalized = Path("outputs") / "debug" / Path(*parts[1:])
    return normalized.as_posix(), "debug", False


def _normalize_project_config(project_cfg: Mapping[str, Any], config_path: str | Path) -> dict[str, Any]:
    normalized = require_mapping(project_cfg, "Config section 'project'")
    _deep_setdefault(normalized, _PROJECT_DEFAULTS)

    normalized["seed"] = _required_int(normalized.get("seed"), "project.seed", minimum=0)
    normalized["device"] = _required_str(normalized.get("device"), "project.device")
    normalized["out_dir"] = _required_str(normalized.get("out_dir"), "project.out_dir")
    normalized_out_dir, output_bucket, spec_valid = _normalize_out_dir(normalized["out_dir"])
    normalized["out_dir"] = normalized_out_dir
    normalized["deterministic"] = bool(normalized.get("deterministic", True))
    normalized["strict_repro"] = bool(normalized.get("strict_repro", False))
    normalized["allow_tf32"] = bool(normalized.get("allow_tf32", True))
    normalized["allow_unsafe_checkpoint_load"] = bool(normalized.get("allow_unsafe_checkpoint_load", False))
    normalized["matmul_precision"] = _required_str(normalized.get("matmul_precision"), "project.matmul_precision")
    normalized["num_threads"] = _optional_int(normalized.get("num_threads"), "project.num_threads", minimum=1)
    normalized["num_interop_threads"] = _optional_int(
        normalized.get("num_interop_threads"),
        "project.num_interop_threads",
        minimum=1,
    )
    normalized["stage_timeout_sec"] = _optional_int(
        normalized.get("stage_timeout_sec"),
        "project.stage_timeout_sec",
        minimum=1,
    )

    resolved_config_path = Path(config_path).resolve()
    runtime = require_mapping(normalized.get("runtime") or {}, "Config section 'project.runtime'")
    runtime.setdefault("config_path", str(resolved_config_path))
    runtime.setdefault("config_dir", str(resolved_config_path.parent))
    runtime.setdefault("cwd", str(Path.cwd().resolve()))
    runtime.setdefault("python_executable", sys.executable)
    runtime.setdefault("config_sha256", _config_sha256(resolved_config_path))
    runtime.setdefault("python_version", platform.python_version())
    runtime.setdefault("platform", platform.platform())
    runtime.setdefault("output_bucket", output_bucket)
    runtime.setdefault("spec_output_path_valid", bool(spec_valid))
    if normalized_out_dir.startswith("outputs/"):
        failed_suffix = normalized_out_dir[len("outputs/") :]
        # Strip the bucket prefix when it is "debug" so that developer-iteration
        # failures land under outputs/failed/<meaningful_suffix> rather than
        # outputs/failed/debug/<meaningful_suffix>.
        if output_bucket == "debug" and "/" in failed_suffix:
            failed_suffix = failed_suffix.split("/", 1)[1]
        runtime.setdefault("failed_out_dir", str((Path("outputs") / "failed" / failed_suffix).as_posix()))
    else:
        runtime.setdefault("failed_out_dir", "")
    normalized["runtime"] = runtime
    return normalized


def resolve_oracle_name(cfg: Mapping[str, Any], cli_oracle: str = "") -> str:
    if cli_oracle:
        return str(cli_oracle)
    # Prefer stage2.oracle_name (the primary diffusion oracle) over
    # stage3.oracle_name/ids_name (which controls PCAP evaluator selection).
    stage2 = cfg.get("stage2")
    if isinstance(stage2, Mapping):
        name = stage2.get("oracle_name")
        if name:
            return str(name)
        name = stage2.get("ids_name")
        if name:
            return str(name)
    stage3 = cfg.get("stage3")
    if isinstance(stage3, Mapping):
        name = stage3.get("oracle_name")
        if name:
            return str(name)
        name = stage3.get("main_ids_name")
        if name:
            return str(name)
    ids_models = cfg.get("ids_models")
    if isinstance(ids_models, list) and ids_models:
        first = ids_models[0]
        if isinstance(first, Mapping) and first.get("name"):
            return str(first["name"])
    oracle_models = cfg.get("oracle_models")
    if isinstance(oracle_models, list) and oracle_models:
        first = oracle_models[0]
        if isinstance(first, Mapping) and first.get("name"):
            return str(first["name"])
    return ""


def apply_pipeline_defaults(cfg: Mapping[str, Any], cli_oracle: str = "") -> dict[str, Any]:
    normalized = dict(cfg)
    project = dict(normalized.get("project") or {})
    data = dict(normalized.get("data") or {})
    stage1 = dict(normalized.get("stage1") or {})
    stage2 = dict(normalized.get("stage2") or {})
    stage3 = dict(normalized.get("stage3") or {})

    ids_models = normalized.get("ids_models")
    if ids_models is not None and "oracle_models" not in normalized:
        normalized["oracle_models"] = ids_models
    oracle_models = normalized.get("oracle_models")
    if oracle_models is not None and "ids_models" not in normalized:
        normalized["ids_models"] = oracle_models

    ids_names = stage1.get("ids_names")
    if ids_names and "oracle_names" not in stage1:
        stage1["oracle_names"] = list(ids_names)
    oracle_names = stage1.get("oracle_names")
    if oracle_names and "ids_names" not in stage1:
        stage1["ids_names"] = list(oracle_names)

    if stage2.get("ids_name") and "oracle_name" not in stage2:
        stage2["oracle_name"] = stage2["ids_name"]
    if stage2.get("oracle_name") and "ids_name" not in stage2:
        stage2["ids_name"] = stage2["oracle_name"]

    if stage3.get("main_ids_name") and "oracle_name" not in stage3:
        stage3["oracle_name"] = stage3["main_ids_name"]
    # Only copy ids_name → oracle_name as a fallback when no oracle is configured.
    # ids_name is typically a PCAP evaluator label ("pcap_ids"), not the primary
    # diffusion oracle. A blind copy here would overwrite the correct stage2 oracle.
    if stage3.get("ids_name") and "oracle_name" not in stage3 and "oracle_name" not in stage2:
        stage3["oracle_name"] = stage3["ids_name"]
    if stage3.get("oracle_name") and "main_ids_name" not in stage3:
        stage3["main_ids_name"] = stage3["oracle_name"]

    normalized["project"] = project
    normalized["data"] = data
    normalized["stage1"] = stage1
    normalized["stage2"] = stage2
    normalized["stage3"] = stage3

    _deep_setdefault(project, _PROJECT_DEFAULTS)
    _deep_setdefault(data, _DATA_DEFAULTS)
    oracle_name = resolve_oracle_name(normalized, cli_oracle=cli_oracle)
    if oracle_name:
        # Stage2/3 need a single primary oracle name, but Stage1 reviewer-matrix
        # runs may intentionally keep a heterogeneous oracle list for mutual
        # extraction evaluation. Do not collapse Stage1 oracle_names/ids_names.
        if not stage1.get("oracle_names"):
            stage1["oracle_names"] = [oracle_name]
        if not stage1.get("ids_names"):
            stage1["ids_names"] = [oracle_name]
        stage2["oracle_name"] = oracle_name
        stage2["ids_name"] = oracle_name
        stage3["oracle_name"] = oracle_name
        stage3["main_ids_name"] = oracle_name

    _deep_setdefault(stage1, _STAGE1_DEFAULTS)
    _deep_setdefault(stage2, _STAGE2_DEFAULTS)
    _deep_setdefault(stage3, _STAGE3_DEFAULTS)
    return normalized


def prepare_pipeline_config(cfg: Mapping[str, Any], config_path: str | Path, cli_oracle: str = "") -> dict[str, Any]:
    normalized = apply_pipeline_defaults(cfg, cli_oracle=cli_oracle)
    normalized["project"] = _normalize_project_config(normalized.get("project") or {}, config_path)
    data_cfg = require_mapping(normalized.get("data") or {}, "Config section 'data'")
    if normalized["project"].get("strict_repro", False):
        data_cfg["strict_ingest"] = True
        data_cfg["encoding_errors"] = "strict"
    normalized["data"] = data_cfg
    return normalized
