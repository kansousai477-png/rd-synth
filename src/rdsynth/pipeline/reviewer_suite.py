from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from rdsynth.data.csv_datasets import resolve_dataset_profile

ROOT = Path(__file__).resolve().parents[3]

DATASET_SPECS: dict[str, dict[str, Any]] = {
    "nb15": {
        "title": "CIC NB15",
        "base_config": "configs/nb15_attack_sweep.yaml",
        "default_attacks": ["Generic", "Exploits", "Fuzzers", "DoS", "Worms"],
        "global_binary": True,
        "audit_dataset": "cic_nb15",
    },
    "2017": {
        "title": "CIC-IDS2017",
        "base_config": "configs/cic_ids2017_attack_sweep.yaml",
        "default_attacks": ["DDoS", "PortScan", "Bot", "Heartbleed", "Web Attack - XSS", "Infiltration"],
        "global_binary": True,
        "audit_dataset": "cic_ids2017",
    },
    "2018": {
        "title": "CIC-IDS2018",
        "base_config": "configs/cic_ids2018_attack_sweep.yaml",
        "default_attacks": [
            "DDOS attack-HOIC",
            "Bot",
            "FTP-BruteForce",
            "SSH-Bruteforce",
            "SQL Injection",
            "Infilteration",
        ],
        "global_binary": True,
        "audit_dataset": "cic_ids2018",
    },
    "iot23": {
        "title": "CIC-IoT-2023",
        "base_config": "configs/iot23_attack_sweep.yaml",
        "default_attacks": [
            "MITM-ArpSpoofing",
            "CommandInjection",
            "Backdoor_Malware",
            "DDoS-UDP_Flood",
            "VulnerabilityScan",
        ],
        "global_binary": True,
        "audit_dataset": "cic_iot2023",
    },
}

DEFAULT_STAGE2_BASELINES = [
    "identity",
    "global_random",
    "knn_benign",
    "fgsm",
    "pgd",
    "iat_jitter",
    "padding_only",
    "topk_perturb",
    "idsgan_lite",
    "digfupas_lite",
    "gpmt_lite",
    "progen_lite",
    "amoeba_lite",
    "vulnergan_lite",
    "netdiffusion_lite",
]

DEFAULT_ABLATION_VARIANTS = [
    "full",
    "backbone_gan",
    "w_o_stage1",
    "w_o_conditioning",
    "random_remap",
]

DEFAULT_TRANSFER_IDS = ["logistic_small", "random_forest_small", "linear_svm_small"]
DEFAULT_TRANSFER_ORACLES = list(DEFAULT_TRANSFER_IDS)

REVIEWER_PROFILES: dict[str, dict[str, Any]] = {
    "paper": {
        "seeds": [42],
        "stage2_baselines": list(DEFAULT_STAGE2_BASELINES),
        "ablation_variants": list(DEFAULT_ABLATION_VARIANTS),
        "transfer_ids": list(DEFAULT_TRANSFER_IDS),
        "max_attacks_per_dataset": 0,
        "stage2_baselines_enabled": True,
        "stage3_baselines_enabled": True,
        "speed_mode": "paper",
    },
    "standard": {
        "seeds": [42],
        "stage2_baselines": [
            "identity",
            "global_random",
            "knn_benign",
            "pgd",
            "gpmt_lite",
            "netdiffusion_lite",
        ],
        "ablation_variants": ["full", "w_o_stage1", "backbone_gan", "random_remap"],
        "transfer_ids": ["logistic_small", "random_forest_small"],
        "max_attacks_per_dataset": 2,
        "stage2_baselines_enabled": True,
        "stage3_baselines_enabled": True,
        "speed_mode": "standard",
    },
    "quick": {
        "seeds": [42],
        "stage2_baselines": ["identity", "global_random", "pgd", "gpmt_lite"],
        "ablation_variants": ["full", "w_o_stage1", "backbone_gan", "random_remap"],
        "transfer_ids": [],
        "max_attacks_per_dataset": 1,
        "stage2_baselines_enabled": True,
        "stage3_baselines_enabled": False,
        "speed_mode": "quick",
    },
}

ABLATION_PATCHES: dict[str, dict[str, Any]] = {
    "full": {},
    "backbone_gan": {"stage2": {"generator_backbone": "gan"}},
    "w_o_conditioning": {"stage2": {"conditioning_enabled": False}},
    "w_o_surrogate_embedding": {"stage2": {"surrogate_guidance_mode": "raw_only"}},
    "w_o_stage1": {
        "stage1": {
            "extraction_mode": "baseline_only",
            "query_real_ratio": 0.0,
            "query_mix_ratio": 0.0,
            "real_warmup_steps": 0,
        },
        "stage2": {
            "surrogate_guidance_mode": "raw_only",
            "require_stage1": False,
        },
    },
    "w_o_deep_semantics_logits": {"stage2": {"surrogate_guidance_mode": "logits"}},
    "w_o_deep_semantics_hard_label": {"stage2": {"surrogate_guidance_mode": "hard_label"}},
    "backbone_cgan": {"stage2": {"generator_backbone": "cgan"}},
    "backbone_wgan": {"stage2": {"generator_backbone": "wgan"}},
    "random_remap": {"stage3": {"remap_mode": "random"}},
    "w_o_protocol_projection": {
        "stage2": {
            "constraints": {"enable": False},
            "deployable_constraints": {"enable": False},
        },
        "stage3": {"protocol_auto_fix": False},
    },
    "w_o_payload_preservation": {"stage2": {"lambda_preserve": 0.0}},
    "w_o_auto_fix": {"stage3": {"protocol_auto_fix": False}},
}

STRESS_TEST_PATCHES: dict[str, dict[str, Any]] = {
    # --- Hard-label noise stress (SPEC §14.1) ---
    "label_noise_0": {"stage1": {"query_label_noise": 0.0}},
    "label_noise_5": {"stage1": {"query_label_noise": 0.05}},
    "label_noise_10": {"stage1": {"query_label_noise": 0.10}},
    "label_noise_20": {"stage1": {"query_label_noise": 0.20}},
    "label_noise_30": {"stage1": {"query_label_noise": 0.30}},
    # --- Feature extractor mismatch (SPEC §14.2) ---
    "extractor_mismatch_reduced": {
        "data": {"mismatch_mode": "reduced_profile"},
    },
    "extractor_mismatch_timeout": {
        "data": {"mismatch_mode": "timeout_shift"},
    },
    # --- Efficiency profile (SPEC §14.4) ---
    "efficiency_profile": {
        "project": {"profile_efficiency": True},
    },
}


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def _rq1_ids_zoo(primary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    common = {
        "epochs": 3,
        "batch_size": 256,
        "max_batches_per_epoch": 24,
        "class_weight": "balanced",
        "sample_strategy": "balanced",
    }
    extras = [
        {**common, "name": "cnn_small", "type": "cnn", "channels": 16, "lr": 1.0e-3},
        {**common, "name": "rnn_small", "type": "rnn", "hidden_dim": 48, "lr": 1.0e-3},
        {**common, "name": "lstm_small", "type": "lstm", "hidden_dim": 48, "lr": 1.0e-3},
        {**common, "name": "gru_small", "type": "gru", "hidden_dim": 48, "lr": 1.0e-3},
        {
            **common,
            "name": "transformer_small",
            "type": "transformer",
            "d_model": 32,
            "nhead": 4,
            "num_layers": 1,
            "lr": 1.0e-3,
        },
    ]
    seen = {str(item.get("name", "")).strip() for item in primary}
    merged = [dict(item) for item in primary]
    for item in extras:
        if item["name"] not in seen:
            merged.append(item)
    return merged


def deep_update(target: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            deep_update(target[key], value)
        else:
            target[key] = value
    return target


def run_command(command: list[str], *, cwd: Path) -> None:
    env = dict(os.environ)
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    subprocess.run(command, check=True, cwd=str(cwd), env=env)


def safe_console_text(text: str) -> str:
    encoding = sys.stdout.encoding or "utf-8"
    return text.encode(encoding, errors="replace").decode(encoding, errors="replace")


def safe_print(text: str) -> None:
    print(safe_console_text(text))


def resolve_python_executable(repo_root: Path | None = None, explicit: str = "") -> str:
    if explicit.strip():
        return explicit.strip()
    env_python = os.environ.get("RDSYNTH_PYTHON", "").strip()
    if env_python:
        return env_python

    root = repo_root or ROOT
    candidates = [
        root / "venv" / "Scripts" / "python.exe",
        Path.home() / "anaconda3" / "envs" / "rdsynth" / "python.exe",
        Path.home() / "miniconda3" / "envs" / "rdsynth" / "python.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return sys.executable


def slugify(value: str) -> str:
    slug = value.strip()
    for old, new in (("/", "_"), ("\\", "_"), (" ", "_"), (":", "_")):
        slug = slug.replace(old, new)
    return slug


def to_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def fmt_float(value: object, digits: int = 4) -> str:
    number = to_float(value)
    if number is None:
        return "-"
    return f"{number:.{digits}f}"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or ["placeholder"])
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def upsert_row(
    rows: dict[tuple[str, ...], dict[str, str]],
    key: tuple[str, ...],
    payload: dict[str, str],
) -> None:
    rows[key] = payload


def load_indexed_rows(
    path: Path,
    key_fields: list[str],
) -> dict[tuple[str, ...], dict[str, str]]:
    indexed: dict[tuple[str, ...], dict[str, str]] = {}
    for row in load_csv_rows(path):
        indexed[tuple(str(row.get(field, "")) for field in key_fields)] = row
    return indexed


def sorted_indexed_rows(indexed: dict[tuple[str, ...], dict[str, str]]) -> list[dict[str, str]]:
    return [indexed[key] for key in sorted(indexed)]


def normalize_label(value: object) -> str:
    return " ".join(str(value).replace("\ufeff", "").strip().split())


def canonicalize_label(value: str) -> str:
    text = normalize_label(value)
    text = text.replace("\ufffd", "-")
    text = text.replace("閿?", "-")
    while "  " in text:
        text = text.replace("  ", " ")
    for token in (" - ", "- ", " -"):
        text = text.replace(token, "-")
    return text.casefold()


def _resolve_csv_paths(data_cfg: dict[str, Any]) -> list[Path]:
    profile = resolve_dataset_profile(data_cfg)
    csv_path = str(profile.csv_path or "").strip()
    csv_dir = str(profile.csv_dir or "").strip()
    csv_glob = str(profile.csv_glob)
    paths: list[Path] = []
    if csv_path:
        path = Path(csv_path)
        if path.exists():
            paths.append(path)
    if csv_dir:
        directory = Path(csv_dir)
        if directory.exists():
            paths.extend(sorted(path for path in directory.glob(csv_glob) if path.is_file()))
    unique = sorted({path.resolve() for path in paths})
    if not unique:
        raise FileNotFoundError("No CSV files matched the configured dataset source.")
    return unique


def _attack_label_cache_dir() -> Path:
    path = ROOT / ".cache" / "reviewer_attack_labels"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _attack_label_cache_key(base_cfg: dict[str, Any]) -> str:
    data_cfg = dict(base_cfg.get("data") or {})
    profile = resolve_dataset_profile(data_cfg)
    csv_paths = _resolve_csv_paths(data_cfg)
    file_state = []
    for path in csv_paths:
        stat = path.stat()
        file_state.append((str(path), int(stat.st_mtime_ns), int(stat.st_size)))
    payload = {
        "version": 1,
        "dataset": str(data_cfg.get("dataset", "")),
        "label_source": str(profile.label_source),
        "label_col": str(profile.label_col),
        "benign_labels": [normalize_label(label) for label in profile.benign_labels],
        "drop_cols": list(profile.drop_cols),
        "merge_strategy": str(data_cfg.get("merge_strategy", "intersection")),
        "include_labels": [str(label) for label in data_cfg.get("include_labels", []) or []],
        "max_rows": data_cfg.get("max_rows"),
        "max_rows_per_label": data_cfg.get("max_rows_per_label"),
        "csv_glob": str(profile.csv_glob),
        "files": file_state,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_cached_attack_labels(base_cfg: dict[str, Any]) -> list[str] | None:
    cache_path = _attack_label_cache_dir() / f"{_attack_label_cache_key(base_cfg)}.json"
    if not cache_path.exists():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    labels = payload.get("labels")
    if not isinstance(labels, list):
        return None
    return [str(label) for label in labels]


def _save_cached_attack_labels(base_cfg: dict[str, Any], labels: list[str]) -> None:
    cache_path = _attack_label_cache_dir() / f"{_attack_label_cache_key(base_cfg)}.json"
    payload = {"labels": [str(label) for label in labels]}
    temp_path = cache_path.with_suffix(".tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(cache_path)


def discover_attack_labels(base_cfg: dict[str, Any]) -> list[str]:
    cached = _load_cached_attack_labels(base_cfg)
    if cached is not None:
        return cached

    data_cfg = base_cfg["data"]
    dataset_name = str(data_cfg.get("dataset", ""))
    if dataset_name == "cic_iot2023":
        csv_dir = Path(str(data_cfg.get("csv_dir", "data/CIC_IOT_Dataset2023/CSV")))
        attacks = [path.name for path in sorted(csv_dir.iterdir()) if path.is_dir() and path.name != "Benign_Final"]
        _save_cached_attack_labels(base_cfg, attacks)
        return attacks

    profile = resolve_dataset_profile(data_cfg)
    benign_labels = {normalize_label(label) for label in profile.benign_labels}
    label_col = normalize_label(profile.label_col)
    counts: dict[str, int] = {}
    for path in _resolve_csv_paths(data_cfg):
        with path.open("r", newline="", encoding="utf-8", errors="replace") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if not header:
                continue
            fields = [normalize_label(name) for name in header]
            if label_col not in fields:
                continue
            label_idx = fields.index(label_col)
            for row in reader:
                if label_idx >= len(row):
                    continue
                label = normalize_label(row[label_idx])
                if not label or label in benign_labels or label == label_col:
                    continue
                counts[label] = counts.get(label, 0) + 1
    attacks = [name for name, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]
    _save_cached_attack_labels(base_cfg, attacks)
    return attacks


def resolve_attacks(base_cfg: dict[str, Any], requested: list[str]) -> list[str]:
    if not requested:
        return []
    available = discover_attack_labels(base_cfg)
    canonical_map: dict[str, list[str]] = {}
    for name in available:
        canonical_map.setdefault(canonicalize_label(name), []).append(name)
    resolved: list[str] = []
    missing: list[str] = []
    seen: set[str] = set()
    for token in requested:
        raw = token.strip()
        if not raw:
            continue
        matches = canonical_map.get(canonicalize_label(raw), [])
        if len(matches) != 1:
            missing.append(raw)
            continue
        chosen = matches[0]
        if chosen not in seen:
            seen.add(chosen)
            resolved.append(chosen)
    if missing:
        raise SystemExit(f"Unknown or ambiguous attack labels: {', '.join(missing)}")
    return resolved


def resolve_profile_overrides(profile: str) -> dict[str, Any]:
    key = str(profile or "paper").strip().lower()
    if key not in REVIEWER_PROFILES:
        raise SystemExit(f"Unknown reviewer profile: {profile}")
    return deepcopy(REVIEWER_PROFILES[key])


def summarize_workload(
    *,
    selected_attacks: dict[str, list[str]],
    global_binary_datasets: set[str] | None = None,
    seeds: list[int],
    stage2_baselines: list[str],
    ablation_variants: list[str],
    transfer_oracles: list[str],
    stage2_baselines_enabled: bool,
    skip_transfer: bool,
) -> dict[str, Any]:
    if global_binary_datasets is None:
        global_binary_datasets = {
            name for name, spec in DATASET_SPECS.items() if bool(spec.get("global_binary", False))
        }
    total_attacks = sum(len(attacks) for attacks in selected_attacks.values())
    seed_count = len(seeds)
    combinations = 0
    main_runs = 0
    for dataset, attacks in selected_attacks.items():
        attack_count = len(attacks)
        if dataset in global_binary_datasets:
            combinations += attack_count * seed_count
            main_runs += seed_count
        else:
            combinations += attack_count * seed_count
            main_runs += attack_count * seed_count
    actual_ablation_variants = [variant for variant in ablation_variants if variant != "full"]
    ablation_reuses = main_runs if "full" in ablation_variants else 0
    ablation_reruns = main_runs * len(actual_ablation_variants)
    transfer_runs = 0 if skip_transfer or not transfer_oracles else main_runs
    transfer_oracle_fits = transfer_runs * len(transfer_oracles)
    pipeline_invocations = main_runs + ablation_reruns
    stage2_baseline_runs = main_runs * len(stage2_baselines) if stage2_baselines_enabled else 0
    return {
        "dataset_attack_counts": {dataset: len(attacks) for dataset, attacks in selected_attacks.items()},
        "total_attacks": total_attacks,
        "seed_count": seed_count,
        "combinations": combinations,
        "main_runs": main_runs,
        "ablation_variants": len(ablation_variants),
        "ablation_reuses": ablation_reuses,
        "ablation_reruns": ablation_reruns,
        "transfer_runs": transfer_runs,
        "transfer_oracle_fits": transfer_oracle_fits,
        "stage2_baseline_runs": stage2_baseline_runs,
        "pipeline_invocations": pipeline_invocations,
    }


def print_workload_summary(
    *,
    selected_attacks: dict[str, list[str]],
    workload: dict[str, Any],
    profile: str,
    stage2_baselines: list[str],
    ablation_variants: list[str],
    transfer_oracles: list[str],
    stage2_baselines_enabled: bool,
    stage3_baselines_enabled: bool,
    skip_transfer: bool,
) -> None:
    safe_print(f"[ReviewerSuite] workload profile={profile}")
    for dataset, attacks in selected_attacks.items():
        preview = ", ".join(attacks[:4])
        if len(attacks) > 4:
            preview += ", ..."
        safe_print(f"  dataset={dataset} attacks={len(attacks)}" + (f" [{preview}]" if preview else ""))
    safe_print(
        "  combinations="
        f"{workload['combinations']} "
        f"(attacks={workload['total_attacks']} x seeds={workload['seed_count']})"
    )
    safe_print(
        "  main_runs="
        f"{workload['main_runs']} "
        f"stage2_baselines_per_main={len(stage2_baselines) if stage2_baselines_enabled else 0} "
        f"stage3_baselines_enabled={stage3_baselines_enabled}"
    )
    safe_print(
        "  ablations="
        f"{len(ablation_variants)} "
        f"(reruns={workload['ablation_reruns']}, reused_full={workload['ablation_reuses']})"
    )
    safe_print(
        "  transfer_runs="
        f"{workload['transfer_runs']} "
        f"(oracle_fits={workload['transfer_oracle_fits']}, enabled={not skip_transfer and bool(transfer_oracles)})"
    )
    safe_print(
        "  run_pipeline_invocations="
        f"{workload['pipeline_invocations']} "
        f"stage2_baseline_executions={workload['stage2_baseline_runs']}"
    )
    if workload["pipeline_invocations"] >= 100 or workload["combinations"] >= 50:
        safe_print(
            "  tip=large workload; consider --profile standard/quick, --jobs N, "
            "--skip-existing, or limiting attacks before launching."
        )


def selected_attacks(
    dataset: str,
    *,
    suite_cfg: dict[str, Any],
    base_cfg: dict[str, Any],
    override_attacks: list[str],
    max_attacks: int,
) -> list[str]:
    if override_attacks:
        return resolve_attacks(base_cfg, override_attacks)
    suite_dataset_cfg = (suite_cfg.get("datasets") or {}).get(dataset, {})
    requested = [str(token) for token in suite_dataset_cfg.get("attacks", [])]
    if bool(DATASET_SPECS.get(dataset, {}).get("global_binary", False)):
        if requested:
            attacks = resolve_attacks(base_cfg, requested)
        else:
            attacks = discover_attack_labels(base_cfg)
        if not attacks:
            attacks = list(DATASET_SPECS[dataset]["default_attacks"])
        if max_attacks > 0:
            return attacks[:max_attacks]
        return attacks
    if requested:
        attacks = resolve_attacks(base_cfg, requested)
    else:
        attacks = list(DATASET_SPECS[dataset]["default_attacks"])
    if max_attacks > 0:
        return attacks[:max_attacks]
    return attacks


def apply_speed_profile(cfg: dict[str, Any], profile: str) -> dict[str, Any]:
    mode = str(profile or "paper").strip().lower()
    if mode == "paper":
        # Paper mode still benefits from query budget and structural efficiency.
        _apply_paper_efficiency(cfg)
        return cfg

    project_cfg = cfg.setdefault("project", {})
    stage1_cfg = cfg.setdefault("stage1", {})
    stage2_cfg = cfg.setdefault("stage2", {})
    stage3_cfg = cfg.setdefault("stage3", {})

    project_cfg["num_threads"] = min(int(project_cfg.get("num_threads", 4)), 4 if mode == "standard" else 2)
    project_cfg["num_interop_threads"] = min(
        int(project_cfg.get("num_interop_threads", 2)),
        2 if mode == "standard" else 1,
    )

    for oracle_cfg in cfg.get("oracle_models", []) or []:
        if mode == "standard":
            oracle_cfg["epochs"] = min(int(oracle_cfg.get("epochs", 5)), 3)
            oracle_cfg["max_batches_per_epoch"] = min(int(oracle_cfg.get("max_batches_per_epoch", 40)), 24)
        else:
            oracle_cfg["epochs"] = min(int(oracle_cfg.get("epochs", 5)), 1)
            oracle_cfg["max_batches_per_epoch"] = min(int(oracle_cfg.get("max_batches_per_epoch", 40)), 10)
            oracle_cfg["batch_size"] = min(int(oracle_cfg.get("batch_size", 512)), 256)

    stage1_cfg["force_retrain"] = False
    if mode == "standard":
        stage1_cfg["steps"] = min(int(stage1_cfg.get("steps", 500)), 240)
        stage1_cfg["real_warmup_steps"] = min(int(stage1_cfg.get("real_warmup_steps", 100)), 60)
        stage1_cfg["baseline_steps"] = min(int(stage1_cfg.get("baseline_steps", 200)), 120)
        stage1_cfg["eval_max_rows"] = min(int(stage1_cfg.get("eval_max_rows", 3000)), 2000)
        stage1_cfg["matrix_max_rows"] = min(int(stage1_cfg.get("matrix_max_rows", 2000)), 1000)
    else:
        stage1_cfg["steps"] = min(int(stage1_cfg.get("steps", 500)), 60)
        stage1_cfg["real_warmup_steps"] = min(int(stage1_cfg.get("real_warmup_steps", 100)), 20)
        stage1_cfg["baseline_steps"] = min(int(stage1_cfg.get("baseline_steps", 200)), 40)
        stage1_cfg["eval_max_rows"] = min(int(stage1_cfg.get("eval_max_rows", 3000)), 1000)
        stage1_cfg["eval_batch_size"] = min(int(stage1_cfg.get("eval_batch_size", 1024)), 512)
        stage1_cfg["matrix_max_rows"] = min(int(stage1_cfg.get("matrix_max_rows", 2000)), 1000)
        stage1_cfg["compare_baseline"] = False

    # Apply efficiency optimizations at all speed levels.
    _apply_paper_efficiency(cfg)
    stage1_cfg.setdefault("query_pool", max(2, min(4, int(stage1_cfg.get("query_pool", 4)) // 2)))
    stage1_cfg.setdefault("query_budget", stage1_cfg.get("query_budget", int(stage1_cfg["steps"] * 512)))

    if mode == "standard":
        stage2_cfg["epochs"] = min(int(stage2_cfg.get("epochs", 18)), 12)
        stage2_cfg["ae_epochs"] = min(int(stage2_cfg.get("ae_epochs", 12)), 8)
        stage2_cfg["timesteps"] = min(int(stage2_cfg.get("timesteps", 150)), 120)
        stage2_cfg["latent_warmup_epochs"] = min(int(stage2_cfg.get("latent_warmup_epochs", 8)), 6)
        stage2_cfg["cond_dropout_warmup_epochs"] = min(int(stage2_cfg.get("cond_dropout_warmup_epochs", 6)), 4)
        stage2_cfg["eval_samples"] = min(int(stage2_cfg.get("eval_samples", 1000)), 600)
        stage2_cfg["metrics_max_real"] = min(int(stage2_cfg.get("metrics_max_real", 1000)), 600)
        stage2_cfg["metrics_max_gen"] = min(int(stage2_cfg.get("metrics_max_gen", 1000)), 600)
        stage2_cfg["sample_batch_size"] = min(int(stage2_cfg.get("sample_batch_size", 512)), 384)
        stage2_cfg.setdefault("baselines", {})
        stage2_cfg["baselines"]["pgd_steps"] = min(int(stage2_cfg["baselines"].get("pgd_steps", 12)), 10)
        stage2_cfg["baselines"]["eval_metrics"] = bool(stage2_cfg["baselines"].get("eval_metrics", True))
    else:
        stage2_cfg["epochs"] = min(int(stage2_cfg.get("epochs", 18)), 6)
        stage2_cfg["ae_epochs"] = min(int(stage2_cfg.get("ae_epochs", 12)), 4)
        stage2_cfg["timesteps"] = min(int(stage2_cfg.get("timesteps", 150)), 60)
        stage2_cfg["latent_warmup_epochs"] = min(int(stage2_cfg.get("latent_warmup_epochs", 8)), 3)
        stage2_cfg["cond_dropout_warmup_epochs"] = min(int(stage2_cfg.get("cond_dropout_warmup_epochs", 6)), 3)
        stage2_cfg["eval_samples"] = min(int(stage2_cfg.get("eval_samples", 1000)), 400)
        stage2_cfg["metrics_max_real"] = min(int(stage2_cfg.get("metrics_max_real", 1000)), 400)
        stage2_cfg["metrics_max_gen"] = min(int(stage2_cfg.get("metrics_max_gen", 1000)), 400)
        stage2_cfg["sample_batch_size"] = min(int(stage2_cfg.get("sample_batch_size", 512)), 256)
        stage2_cfg.setdefault("baselines", {})
        stage2_cfg["baselines"]["pgd_steps"] = min(int(stage2_cfg["baselines"].get("pgd_steps", 12)), 8)
        stage2_cfg["baselines"]["eval_metrics"] = False

    if mode == "standard":
        stage3_cfg["epochs"] = min(int(stage3_cfg.get("epochs", 20)), 12)
        current_scan_limit = int(stage3_cfg.get("pcap_scan_limit", 30) or 0)
        stage3_cfg["pcap_scan_limit"] = 0 if current_scan_limit <= 0 else min(current_scan_limit, 12)
        stage3_cfg["pcap_eval_batch_size"] = min(int(stage3_cfg.get("pcap_eval_batch_size", 512) or 512), 384)
        stage3_cfg["pcap_apply_n"] = min(int(stage3_cfg.get("pcap_apply_n", 1)), 1)
    else:
        stage3_cfg["epochs"] = min(int(stage3_cfg.get("epochs", 20)), 8)
        stage3_cfg["batch_size"] = min(int(stage3_cfg.get("batch_size", 256)), 128)
        current_scan_limit = int(stage3_cfg.get("pcap_scan_limit", 30) or 0)
        stage3_cfg["pcap_scan_limit"] = 0 if current_scan_limit <= 0 else min(current_scan_limit, 8)
        stage3_cfg["pcap_eval_batch_size"] = min(int(stage3_cfg.get("pcap_eval_batch_size", 512) or 512), 256)
        stage3_cfg["pcap_apply_n"] = min(int(stage3_cfg.get("pcap_apply_n", 1)), 1)
        stage3_cfg["save_intermediate_results"] = False

    return cfg


def _apply_paper_efficiency(cfg: dict[str, Any]) -> None:
    """Apply efficiency optimizations safe for paper-quality results."""
    stage1_cfg = cfg.setdefault("stage1", {})
    stage2_cfg = cfg.setdefault("stage2", {})
    stage3_cfg = cfg.setdefault("stage3", {})

    # Cap oracle queries to prevent unbounded budgets.
    if stage1_cfg.get("query_budget") is None and stage1_cfg.get("steps"):
        stage1_cfg.setdefault("query_budget", int(stage1_cfg["steps"]) * 1024)

    # Halve structured-loss frequency: compute STP/corr/MMD/SWD every 2nd step.
    stage2_cfg.setdefault("structure_every", 2)

    # Enable PCAP feature cache to avoid re-extraction across rounds.
    stage3_cfg.setdefault("pcap_cache_enable", True)


def reviewer_stage3_scan_limit(profile: str, configured_limit: object) -> int:
    """Return the bounded reviewer-suite Stage3 carrier scan budget."""

    try:
        current_limit = int(configured_limit or 0)
    except (TypeError, ValueError):
        current_limit = 0
    if current_limit > 0:
        return current_limit
    scan_defaults = {"paper": 32, "standard": 12, "quick": 8}
    return scan_defaults.get(str(profile or "paper").strip().lower(), 32)


def enforce_reviewer_stage3_policy(
    stage3_cfg: dict[str, Any],
    *,
    profile: str,
    seed: int,
    pcap_source_selection_mode: str = "best",
    pcap_source_sample_n: int = 1,
) -> None:
    """Keep reviewer-suite Stage3 on the bounded, attack-aware carrier path."""

    selection_mode = str(pcap_source_selection_mode or "best").strip().lower()
    allowed_modes = {"best", "top_hard", "random_hard", "random", "all"}
    if selection_mode not in allowed_modes:
        raise ValueError(f"Unsupported reviewer-suite pcap_source_selection_mode={selection_mode!r}")
    source_sample_n = int(pcap_source_sample_n)
    if selection_mode == "best":
        source_sample_n = 1
    elif selection_mode == "all":
        source_sample_n = 0
    else:
        source_sample_n = max(1, source_sample_n)

    stage3_cfg.setdefault("pcap_scan_dir", "data/PCAPs/malicious")
    stage3_cfg.setdefault("pcap_scan_glob", "*.pcap")
    stage3_cfg["pcap_scan_limit"] = reviewer_stage3_scan_limit(profile, stage3_cfg.get("pcap_scan_limit", 32))
    stage3_cfg["pcap_scan_max_bytes"] = int(stage3_cfg.get("pcap_scan_max_bytes", 32 * 1024 * 1024) or 0)
    if stage3_cfg["pcap_scan_max_bytes"] <= 0:
        stage3_cfg["pcap_scan_max_bytes"] = 32 * 1024 * 1024
    stage3_cfg["pcap_source_selection_mode"] = selection_mode
    stage3_cfg["pcap_source_sample_n"] = source_sample_n
    stage3_cfg.setdefault("pcap_target_source", "pcap_conditioned")
    stage3_cfg.setdefault("pcap_ids_benign_path", "data/PCAPs/benign/benign.pcap")  # required for calibration
    stage3_cfg.setdefault("pcap_source_sample_seed", int(seed))
    # Reviewer-suite Stage3 chooses one semantically adjacent carrier from the
    # malicious pool, so fixed carrier paths inherited from base configs or
    # ablation patches must not bypass attack-aware ranking.
    stage3_cfg["pcap_path"] = ""


def reviewer_stage1_shared_root(out_dir: Path) -> Path:
    """Return the reviewer-suite Stage1 cache root for a generated run directory."""

    seed_dir: Path | None = None
    dataset_root: Path | None = None
    for parent in [out_dir, *out_dir.parents]:
        if parent.name.startswith("seed_"):
            seed_dir = parent
            for candidate in parent.parents:
                if candidate.name in {"main", "ablation", "ablations"}:
                    dataset_root = candidate.parent
                    break
            break
    if seed_dir is not None and dataset_root is not None:
        return dataset_root / "_shared" / "stage1" / seed_dir.name
    return out_dir.parent / "_shared" / "stage1"


def build_run_config(
    *,
    base_cfg: dict[str, Any],
    attack: str,
    eval_attack_label: str | None = None,
    semantic_attack_labels: list[str] | None = None,
    seed: int,
    out_dir: Path,
    profile: str = "paper",
    stage2_baselines_enabled: bool,
    stage3_baselines_enabled: bool,
    stage2_baselines: list[str],
    pcap_source_selection_mode: str = "best",
    pcap_source_sample_n: int = 1,
    patch: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = deepcopy(base_cfg)
    apply_speed_profile(cfg, profile)
    cfg.setdefault("project", {})
    cfg.setdefault("data", {})
    cfg.setdefault("stage1", {})
    cfg.setdefault("stage2", {})
    cfg.setdefault("stage3", {})
    cfg["ids_models"] = _rq1_ids_zoo(list(cfg.get("ids_models") or cfg.get("oracle_models") or []))
    cfg["oracle_models"] = [dict(m) for m in cfg["ids_models"]]  # deep copy to prevent aliasing

    benign_labels = [str(label) for label in cfg["data"].get("benign_labels", [])]
    cfg["project"]["seed"] = int(seed)
    cfg["project"]["out_dir"] = str(out_dir).replace("\\", "/")
    cfg["project"].setdefault("stage1_shared_root", str(reviewer_stage1_shared_root(out_dir)).replace("\\", "/"))
    resolved_eval_attack = str(eval_attack_label if eval_attack_label is not None else attack).strip()
    cfg["project"]["attack_type"] = str(attack)
    if resolved_eval_attack:
        cfg["project"]["eval_attack_label"] = resolved_eval_attack
    else:
        cfg["project"].pop("eval_attack_label", None)
    cfg["data"].pop("include_labels", None)
    if resolved_eval_attack:
        cfg["data"]["eval_attack_label"] = resolved_eval_attack
    else:
        cfg["data"].pop("eval_attack_label", None)
    cfg["data"]["benign_labels"] = benign_labels

    cfg["stage1"]["compare_baseline"] = str(cfg["stage1"].get("compare_baseline", True)).strip().lower() not in (
        "false",
        "no",
        "0",
        "",
    )
    cfg["stage1"]["compute_matrix"] = True
    cfg["stage1"]["ids_names"] = [str(item["name"]) for item in cfg["ids_models"]]
    cfg["stage1"]["oracle_names"] = list(cfg["stage1"]["ids_names"])
    cfg["stage1"]["force_retrain"] = False

    project_cfg = cfg.setdefault("project", {})
    if project_cfg.get("num_threads") is None:
        project_cfg["num_threads"] = 4
    if project_cfg.get("num_interop_threads") is None:
        project_cfg["num_interop_threads"] = 1

    stage2_cfg = cfg["stage2"]
    stage2_cfg["ids_name"] = str(
        stage2_cfg.get("ids_name") or stage2_cfg.get("oracle_name") or cfg["ids_models"][0]["name"]
    )
    stage2_cfg["oracle_name"] = stage2_cfg["ids_name"]
    stage2_cfg["save_samples"] = True
    if float(stage2_cfg.get("sample_pullback_alpha", 0.0) or 0.0) <= 0.0:
        stage2_cfg["sample_pullback_alpha"] = 0.10
    if float(stage2_cfg.get("sample_moment_alpha", 0.0) or 0.0) <= 0.0:
        stage2_cfg["sample_moment_alpha"] = 0.10
    pareto_cfg = stage2_cfg.setdefault("pareto_eval", {})
    pareto_cfg.setdefault("enable", True)
    pareto_cfg.setdefault("auto_select", True)
    pareto_selection = pareto_cfg.setdefault("selection", {})
    pareto_selection.setdefault("candidate_selection", "stage3_closed_loop")
    pareto_selection.setdefault("prefer_oracle", True)
    pareto_selection.setdefault("min_asr_oracle", 0.95)
    pareto_selection.setdefault("min_asr_surrogate", 0.90)
    pareto_selection.setdefault("per_sample_distance_weight", 0.10)
    pareto_selection.setdefault("per_sample_support_weight", 0.15)
    pareto_selection.setdefault("per_sample_remapability_weight", 0.30)
    pareto_selection.setdefault("per_sample_stage3_closed_loop_weight", 0.25)
    stage2_baseline_cfg = stage2_cfg.setdefault("baselines", {})
    stage2_baseline_cfg["enable"] = bool(stage2_baselines_enabled)
    stage2_baseline_cfg["methods"] = list(stage2_baselines)

    stage3_cfg = cfg["stage3"]
    stage3_cfg["pcap_dataset"] = str(cfg["data"].get("dataset", ""))
    stage3_cfg["pcap_attack_label"] = resolved_eval_attack or str(attack)
    if semantic_attack_labels:
        stage3_cfg["pcap_attack_labels"] = [str(label) for label in semantic_attack_labels if str(label).strip()]
    else:
        stage3_cfg["pcap_attack_labels"] = []
    stage3_cfg.setdefault("pcap_semantic_filter", True)
    stage3_cfg["main_ids_name"] = str(
        stage3_cfg.get("main_ids_name") or stage3_cfg.get("oracle_name") or stage2_cfg["ids_name"]
    )
    stage3_cfg["oracle_name"] = stage3_cfg["main_ids_name"]
    stage3_cfg["adv_samples_path"] = ""
    stage3_cfg["copy_adv_samples"] = True
    stage3_cfg["pcap_baseline_jobs"] = 1
    enforce_reviewer_stage3_policy(
        stage3_cfg,
        profile=profile,
        seed=seed,
        pcap_source_selection_mode=pcap_source_selection_mode,
        pcap_source_sample_n=pcap_source_sample_n,
    )
    stage3_cfg.setdefault("pcap_scan_min_prob", 0.5)
    stage3_cfg.setdefault("pcap_dst_port_policy", "flow_vocab_closest")
    stage3_cfg.setdefault("pcap_eval", True)
    stage3_cfg["pcap_eval_use_ids"] = True
    stage3_cfg["pcap_eval_use_oracle"] = True
    stage3_cfg["pcap_compare_baselines"] = bool(stage3_baselines_enabled)
    stage3_cfg.setdefault("pcap_feature_fail_closed", True)
    stage3_cfg.setdefault("pcap_feature_fail_on_partial_alignment", True)
    stage3_cfg["pcap_out_dir"] = str(out_dir / "stage3" / "pcap").replace("\\", "/")

    if patch:
        deep_update(cfg, deepcopy(patch))
        enforce_reviewer_stage3_policy(
            stage3_cfg,
            profile=profile,
            seed=seed,
            pcap_source_selection_mode=pcap_source_selection_mode,
            pcap_source_sample_n=pcap_source_sample_n,
        )
    return cfg


def main_ids_name(cfg: dict[str, Any]) -> str:
    stage3_cfg = cfg.get("stage3", {})
    stage2_cfg = cfg.get("stage2", {})
    if stage3_cfg.get("main_ids_name"):
        return str(stage3_cfg["main_ids_name"])
    if stage3_cfg.get("ids_name"):
        return str(stage3_cfg["ids_name"])
    if stage3_cfg.get("oracle_name"):
        return str(stage3_cfg["oracle_name"])
    if stage2_cfg.get("ids_name"):
        return str(stage2_cfg["ids_name"])
    if stage2_cfg.get("oracle_name"):
        return str(stage2_cfg["oracle_name"])
    ids_models = cfg.get("ids_models") or []
    if ids_models:
        return str(ids_models[0].get("name", "mlp_small"))
    oracle_models = cfg.get("oracle_models") or []
    if oracle_models:
        return str(oracle_models[0].get("name", "mlp_small"))
    return "mlp_small"


def main_oracle_name(cfg: dict[str, Any]) -> str:
    return main_ids_name(cfg)


def first_row(rows: list[dict[str, str]], *, key: str, value: str) -> dict[str, str]:
    return next((row for row in rows if str(row.get(key, "")).strip() == value), {})
